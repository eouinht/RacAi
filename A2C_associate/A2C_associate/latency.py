# latency.py
import numpy as np

C_LIGHT = 3e8  # m/s
eps = 1e-30

def build_latency_model(
    num_RUs, num_DUs, num_CUs, num_UEs,
    distances_RU_UE,                 # (num_RUs, num_UEs) in meters
    SLICE_PRESET,                    # dict: slice -> {R_min, packet_size_bits, cycles_per_packet, lambda_default_pps, ...}
    UE_slice_name,                   # list/array len=num_UEs, each in SLICE_PRESET.keys()
    DU_caps_cycles_per_s,            # (num_DUs,) DU capacity_j in cycles/s
    CU_caps_cycles_per_s,            # (num_CUs,) CU capacity_m in cycles/s
):

    # ---------------- Chuẩn hoá & kiểm tra đầu vào ----------------
    slice_names = list(SLICE_PRESET.keys())
    num_slices = len(slice_names)
    UE_slice_name = np.asarray(UE_slice_name)

    distances_RU_UE = np.asarray(distances_RU_UE, dtype=float)
    DU_caps = np.asarray(DU_caps_cycles_per_s, dtype=float).reshape(num_DUs)
    CU_caps = np.asarray(CU_caps_cycles_per_s, dtype=float).reshape(num_CUs)

    assert distances_RU_UE.shape == (num_RUs, num_UEs), "distances_RU_UE wrong shape"
    assert DU_caps.shape == (num_DUs,), "DU_caps_cycles_per_s wrong shape"
    assert CU_caps.shape == (num_CUs,), "CU_caps_cycles_per_s wrong shape"

    # ---------------- 1) Propagation (RU–UE): d/c ----------------
    propagation_delay = distances_RU_UE / C_LIGHT                  # (num_RUs, num_UEs)

    # ---------------- 2) Transmission (per-UE): S_pkt / R_min ----
    transmission_delay = np.empty(num_UEs, dtype=float)
    pkt_bits_arr = np.empty(num_UEs, dtype=float)
    lam_ue_pps   = np.empty(num_UEs, dtype=float)

    for k in range(num_UEs):
        s = UE_slice_name[k]
        preset = SLICE_PRESET[s]
        R_min    = float(preset["R_min"])               # bps
        pkt_bits = float(preset["packet_size_bits"])    # bits
        lam_pps  = float(preset["lambda_default_pps"])  # pkt/s

        # S_pkt / R_min (bảo vệ chia 0 → ∞ nếu R_min=0)
        transmission_delay[k] = (pkt_bits / R_min) if R_min > 0 else np.inf
        pkt_bits_arr[k] = pkt_bits
        lam_ue_pps[k]   = max(lam_pps, 0.0)

    # ---------------- 3) Processing time S = cycles / capacity ----
    # Service time theo (node, slice) — không phụ thuộc UE
    S_du = np.zeros((num_DUs, num_slices), dtype=float)
    S_cu = np.zeros((num_CUs, num_slices), dtype=float)

    safe_DU_caps = np.maximum(DU_caps, eps)
    safe_CU_caps = np.maximum(CU_caps, eps)

    for s_idx, s_name in enumerate(slice_names):
        cycles = float(SLICE_PRESET[s_name]["cycles_per_packet"])
        S_du[:, s_idx] = cycles / safe_DU_caps
        S_cu[:, s_idx] = cycles / safe_CU_caps

    # Broadcast sang (node, slice, UE)
    # (lưu ý: repeat tạo bản sao rõ ràng giúp dễ thao tác về sau)
    processing_delay_DU = np.repeat(S_du[:, :, None], num_UEs, axis=2)  # (J,S,K)
    processing_delay_CU = np.repeat(S_cu[:, :, None], num_UEs, axis=2)  # (M,S,K)

    # ---------------- 4) Queuing delay: M/M/1 ---------------------
    # Mask (S,K): UE k thuộc slice s_idx → 1, else 0
    slice_index_of_ue = np.array([slice_names.index(s) for s in UE_slice_name], dtype=int)  # (K,)
    mask_SK = np.zeros((num_slices, num_UEs), dtype=float)
    mask_SK[slice_index_of_ue, np.arange(num_UEs)] = 1.0

    # λ(S,K): chỉ gán vào slice của UE; các slice khác = 0
    lam_SK = mask_SK * lam_ue_pps[None, :]  # (S,K), λ ≥ 0

    # Mở rộng λ lên trục node
    lam_DU_JSK = np.repeat(lam_SK[None, :, :], num_DUs, axis=0)  # (J,S,K)
    lam_CU_MSK = np.repeat(lam_SK[None, :, :], num_CUs, axis=0)  # (M,S,K)

    # Service time (node, slice, UE) đã có sẵn ở processing_delay_* (chính là S)
    S_du_JSK = processing_delay_DU  # (J,S,K)
    S_cu_MSK = processing_delay_CU  # (M,S,K)

    # ρ = λ * S
    rho_DU = lam_DU_JSK * S_du_JSK
    rho_CU = lam_CU_MSK * S_cu_MSK

    # Wq = (rho / (1 - rho)) * S, với:
    #  - rho >= 1  → Wq = ∞ (bão hoà)
    #  - lambda == 0 (tức rho == 0) → Wq = 0 (không chờ)
    def _wq_mm1(rho, S):
        denom = np.maximum(1.0 - rho, eps)
        Wq = (rho / denom) * S
        Wq = np.where(rho < 1.0, Wq, np.inf)
        Wq = np.where(rho == 0.0, 0.0, Wq)
        return Wq

    queuing_delay_DU = _wq_mm1(rho_DU, S_du_JSK)  # (J,S,K)
    queuing_delay_CU = _wq_mm1(rho_CU, S_cu_MSK)  # (M,S,K)

    # ---------------- Sanity checks -------------------------------
    assert propagation_delay.shape == (num_RUs, num_UEs)
    assert transmission_delay.shape == (num_UEs,)
    assert processing_delay_DU.shape == (num_DUs, num_slices, num_UEs)
    assert processing_delay_CU.shape == (num_CUs, num_slices, num_UEs)
    assert queuing_delay_DU.shape == (num_DUs, num_slices, num_UEs)
    assert queuing_delay_CU.shape == (num_CUs, num_slices, num_UEs)

    return propagation_delay, transmission_delay, processing_delay_DU, processing_delay_CU, queuing_delay_DU, queuing_delay_CU
