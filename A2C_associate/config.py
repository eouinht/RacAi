import numpy as np
# =======================================================
# ================== Tham số mô phỏng ===================
# =======================================================
SEED = 42
# ---------------------- Quy mô mạng -----------------------
num_RUs = 5         # số RU (Radio Unit)
num_DUs = 3         # số DU (Distributed Unit)
num_CUs = 3         # số CU (Central Unit)
total_nodes = num_RUs + num_DUs + num_CUs

# ---------------------- RB & băng thông -------------------
# NR: SCS=30kHz, 1 RB = 12 subcarriers => 360 kHz
subcarrier_bandwidth_Hz = 60e3
num_subcarriers_per_RB  = 12
channel_bandwidth_Hz    = 100e6

#bandwidth_per_RB        = channel_bandwidth_Hz / num_RBs
bandwidth_per_RB        = num_subcarriers_per_RB * subcarrier_bandwidth_Hz

max_RBs_per_UE = 10

# ---------------------- Cấu hình dịch vụ / slice ----------------------
SLICE_PRESET = {
    'eMBB': {
        'type'        : 'eMBB',
        'R_min'       : 50e6,        # [bps]
        'SINR_min'    : 10,          # [dB]
        'eta_slice'   : 0.05,
        'weight_accept'    : 1.0,
        'weight_throughput'     : 1.2,
        'weight_latency'     : 0.8,
        'delay'             : 5e-3,         # [s]
        'packet_size_bits'  : 1500 * 8,
        'cycles_per_packet' : 4000.0,
        'lambda_default_pps': 100.0,
    },
    'uRLLC': {
        'type'        : 'uRLLC',
        'R_min'       : 5e6,
        'SINR_min'    : 20,          # [dB]
        'eta_slice'   : 0.08,
        'weight_accept'    : 1.0,
        'weight_throughput'     : 0.6,
        'weight_latency'     : 1.0,
        'delay'             : 1e-3,        # [s]
        'packet_size_bits'  : 128 * 8,
        'cycles_per_packet' : 2000.0,
        'lambda_default_pps': 500.0,
    },
}

# ---------------------- Công suất --------------------
# RU: công suất phát tối đa ~ 43 dBm ≈ 20 W
max_tx_power_dbm    = 43
max_tx_power_mw     = 10 ** (max_tx_power_dbm / 10)  # mW
max_tx_power_watts  = max_tx_power_mw / 1e3          # W

# ---------------------- Tài nguyên node mạng -----------------
P_i_random_list = [max_tx_power_watts]              # RU powers (W)
A_j_random_list = [8e9]                             # DU CPU (cycles/s)
A_m_random_list = [5e9]                             # CU CPU (cycles/s)
# -------------- Tài nguyên liên kết trong mạng ---------------
bw_ru_du_random_list = [5e9]                               # bps
bw_du_cu_random_list = [20e9]                              # bps

# ---------------------- Mức công suất rời rạc cho agent (RU) --------------
num_power_levels = 10
def generate_power_levels(P_max, num_power_levels):
    if num_power_levels <= 1:
        return [float(P_max)]
    arr = np.linspace(P_max / num_power_levels, P_max, num_power_levels, dtype=float)
    return [float(round(x, 6)) for x in arr]

P_ib_sk_val = generate_power_levels(max_tx_power_watts, num_power_levels)

# ---------------------- Mô hình CPU theo bit ----------------
k_DU = 5.0     # [cycles/bit] tại DU
k_CU = 3.0      # [cycles/bit] tại CU

