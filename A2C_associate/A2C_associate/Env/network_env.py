import numpy as np, copy
# Local modules - must exist
import RAN_topo as RAN_topo
import gen_RU_UE as gen_RU_UE
import wireless as wireless
import latency as latency
# =======================================================
# ====================== ENV ============================
# =======================================================
class NetworkEnv:
    def __init__(self, total_nodes, num_RUs, num_DUs, num_CUs, num_RBs, num_UEs, SLICE_PRESET,
                 P_i_random_list, A_j_random_list, A_m_random_list, bw_ru_du_random_list, bw_du_cu_random_list, bandwidth_per_RB, max_RBs_per_UE, P_ib_sk_val, k_DU, k_CU):

        self.w_acc = 25    # Trọng số accept
        self.w_thr = 1     # Trọng số throughtput
        self.w_lat = 50    # Trọng số latency

        # -------------------- Sizes & Presets --------------------
        self.total_nodes    = int(total_nodes)
        self.num_RUs        = int(num_RUs)
        self.num_DUs        = int(num_DUs)
        self.num_CUs        = int(num_CUs)
        self.num_RBs        = int(num_RBs)
        self.num_UEs        = int(num_UEs)
        self.SLICE_PRESET   = dict(SLICE_PRESET)
        self.slice_names = list(self.SLICE_PRESET.keys())
        self.max_RBs_per_UE = int(max_RBs_per_UE)

        # Ưu tiên slice: eMBB thiên throughput, uRLLC thiên latency
        self.slice_weight_accept = np.array(
            [float(self.SLICE_PRESET[name].get("weight_accept", 1.0)) for name in self.slice_names],
            dtype=float
        )
        self.slice_weight_throughput = np.array(
            [float(self.SLICE_PRESET[name].get("weight_throughput", 1.0)) for name in self.slice_names],
            dtype=float
        )
        self.slice_weight_latency = np.array(
            [float(self.SLICE_PRESET[name].get("weight_latency", 1.0)) for name in self.slice_names],
            dtype=float
        )

        # -------------------- Physics --------------------
        self.bandwidth_per_RB = float(bandwidth_per_RB)   # Hz per RB
        self.P_ib_sk_val      = list(P_ib_sk_val)         # codebook công suất (W)

        # -------------------- Resource Budgets (input topo) --------------------
        self.P_i_random_list = list(P_i_random_list)  # RU power budgets (W)
        self.A_j_random_list = list(A_j_random_list)  # DU CPU caps (cycles/s)
        self.A_m_random_list = list(A_m_random_list)  # CU CPU caps (cycles/s)
        self.bw_ru_du_random_list = np.array(bw_ru_du_random_list, dtype=float)  # (num_RUs, num_DUs)
        self.bw_du_cu_random_list = np.array(bw_du_cu_random_list, dtype=float)  # (num_DUs, num_CUs)

        # -------------------- CPU model (cycles/bit) --------------------
        self.k_DU = float(k_DU)
        self.k_CU = float(k_CU)

        # -------------------- Topology & Capacities --------------------
        G = RAN_topo.create_topo(self.num_RUs, self.num_DUs, self.num_CUs, self.P_i_random_list, self.A_j_random_list, self.A_m_random_list, self.bw_ru_du_random_list, self.bw_du_cu_random_list)
        self.RU_power_cap, self.DU_cap, self.CU_cap = RAN_topo.get_node_cap(G)  # [W], [cycles/s], [cycles/s]
        self.link_bw_ru_du_bps, self.link_bw_du_cu_bps = RAN_topo.get_links(G)                        # (RU,DU), (DU,CU) capacities
        # đồng bộ: dùng mặt nạ nhị phân 0/1
        self.l_ru_du = (np.array(self.link_bw_ru_du_bps, dtype=float) > 0).astype(int)
        self.l_du_cu = (np.array(self.link_bw_du_cu_bps, dtype=float) > 0).astype(int)

        # -------------------- Geometry & Channel --------------------
        coordinates_RU = gen_RU_UE.gen_coordinates_RU(self.num_RUs)
        coordinates_UE = gen_RU_UE.gen_coordinates_UE(self.num_UEs)
        self.distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, self.num_RUs, self.num_UEs)
        self.gain = wireless.channel_gain(self.distances_RU_UE, self.num_RUs, self.num_UEs, self.bandwidth_per_RB)  # linear

        # -------------------- UE Presets & Per-UE State --------------------
        self.UE_slice_name = gen_RU_UE.gen_UE_requirements(self.num_UEs, self.SLICE_PRESET)
        self.UE_requests = {
            i: {
                "id": int(i),
                "slice": str(self.UE_slice_name[i]),
                **copy.deepcopy(self.SLICE_PRESET[str(self.UE_slice_name[i])]),
                "status": {"active": 1, "accepted": None, "reason": None},
                "alloc": {
                    "RU": None, "DU": None, "CU": None,
                    "num_RB_alloc": 0, "power_alloc": 0.0,
                    "throughput_bps": 0.0, "delay_s": 0.0,
                    "delay_parts": None
                },
            } for i in range(self.num_UEs)
        }

        self.propagation_delay, self.transmission_delay, self.processing_delay_DU, self.processing_delay_CU, self.queuing_delay_DU, self.queuing_delay_CU = latency.build_latency_model(self.num_RUs, self.num_DUs, self.num_CUs, self.num_UEs, self.distances_RU_UE, self.SLICE_PRESET, self.UE_slice_name, np.asarray(self.DU_cap, dtype=float), np.asarray(self.CU_cap, dtype=float))

        # -------------------- Init dynamic resources --------------------
        self.reset_env()

    # ======================================================================
    # Basics
    # ======================================================================
    def reset_env(self):
        """Reset remaining resources & clear all UE allocations."""
        self.RB_remaining        = int(self.num_RBs)
        self.RU_power_remaining  = np.copy(self.RU_power_cap).astype(float)
        self.DU_remaining        = np.copy(self.DU_cap).astype(float)
        self.CU_remaining        = np.copy(self.CU_cap).astype(float)

        for i in range(self.num_UEs):
            ue = self.UE_requests[i]
            ue["status"].update({"active": 1, "accepted": None, "reason": None})
            ue["alloc"].update({
                "RU": None, "DU": None, "CU": None,
                "num_RB_alloc": 0, "power_alloc": 0.0,
                "throughput_bps": 0.0, "delay_s": 0.0,
                "delay_parts": None
            })

        self.pending_UEs = {i: {"active": 1} for i in range(self.num_UEs)}
        self.done = False
        return self.get_state()

    def get_state(self):
        """Snapshot tài nguyên & bảng UE (phẳng) cho agent."""
        state = {
            "RU_power_remaining": np.copy(self.RU_power_remaining),
            "DU_remaining":       np.copy(self.DU_remaining),
            "CU_remaining":       np.copy(self.CU_remaining),
            "RU_power_cap":       np.copy(self.RU_power_cap),
            "DU_cap":             np.copy(self.DU_cap),
            "CU_cap":             np.copy(self.CU_cap),
            "l_ru_du":            np.copy(self.l_ru_du),
            "l_du_cu":            np.copy(self.l_du_cu),
            "RB_remaining":       int(self.RB_remaining)
        }

        ue_info = []
        for _, ue in self.UE_requests.items():
            ue_info.append({
                "id": ue["id"],
                "slice": ue["slice"],
                "R_min": float(ue.get("R_min", 0.0)),
                "SINR_min": float(ue.get("SINR_min", 0.0)),
                "delay": float(ue.get("delay", 0.0)),
                "eta_slice": float(ue.get("eta_slice", 0.0)),
                "active": int(ue["status"]["active"]),
                "accepted": None if ue["status"]["accepted"] is None else bool(ue["status"]["accepted"]),
                "RU": None if ue["alloc"]["RU"] is None else int(ue["alloc"]["RU"]),
                "DU": None if ue["alloc"]["DU"] is None else int(ue["alloc"]["DU"]),
                "CU": None if ue["alloc"]["CU"] is None else int(ue["alloc"]["CU"]),
                "num_RB_alloc": int(ue["alloc"]["num_RB_alloc"]),
                "power_alloc": float(ue["alloc"]["power_alloc"]),
                "throughput_bps": float(ue["alloc"]["throughput_bps"]),
                "delay_s": float(ue["alloc"]["delay_s"]),
            })
        state["UE_requests"] = ue_info
        return state
    
    def compute_reward(self, UE, throughput_bps, delay_total, debug=True):
        s_name = UE["slice"]
        s_idx  = self.slice_names.index(s_name)

        ue_id = UE.get("id", UE.get("ue_id", UE.get("k", "NA")))

        R_min = float(UE.get("R_min", 1e-9))
        D_max = float(UE.get("delay", 1e-9))

        thr_util = throughput_bps / R_min
        lat_pen  = delay_total / D_max

        w_acc_s = float(self.slice_weight_accept[s_idx])
        w_thr_s = float(self.slice_weight_throughput[s_idx])
        w_lat_s = float(self.slice_weight_latency[s_idx])

        acc_term = self.w_acc * w_acc_s * 1
        thr_term = self.w_thr * w_thr_s * thr_util
        lat_term = self.w_lat * w_lat_s * lat_pen

        reward = acc_term + thr_term - lat_term

        """if debug:
            print(
                f"[UE={ue_id:>3} | {s_name:<6}] "
                f"thr={throughput_bps:.2e}, Rmin={R_min:.2e}, util={thr_util:6.3f} | "
                f"lat={delay_total:.2e}, Dmax={D_max:.2e}, pen={lat_pen:6.3f} | "
                f"acc={acc_term:6.3f}, thrR={thr_term:6.3f}, latP={lat_term:6.3f} | "
                f"reward={reward:7.3f}"
            )"""

        return float(reward)

    # ======================================================================
    # Feasibility
    # ======================================================================
    def check_feasible(self, UE_idx, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_level_alloc):
        """
        Returns:
            (True, (rate, cpu_DU_req, cpu_CU_req, delay_total, delay_parts))
            hoặc (False, reason).
        """
        eps = 1e-9

        # ---- Index & activity ----
        try:
            if not (0 <= UE_idx < self.num_UEs):   return False, "invalid UE_idx"
            if not (0 <= RU_choice < self.num_RUs): return False, "invalid RU_choice"
            if not (0 <= DU_choice < self.num_DUs): return False, "invalid DU_choice"
            if not (0 <= CU_choice < self.num_CUs): return False, "invalid CU_choice"
        except Exception as e:
            return False, f"index_validation_error: {e}"

        UE = self.UE_requests[UE_idx]
        if int(UE["status"]["active"]) != 1:
            return False, "UE_not_active"

        # ---- Slice constraints ----
        R_min        = float(UE.get("R_min", 0.0))
        SINR_min_dB  = float(UE.get("SINR_min", -1e9))
        SINR_min_lin = 10.0 ** (SINR_min_dB / 10.0)
        L_max        = float(UE.get("delay", np.inf))
        eta          = float(UE.get("eta_slice", 0.0))

        # ---- RB / Power availability ----
        if not isinstance(num_RB_alloc, (int, np.integer)):
            return False, "invalid num_RB_alloc_type"
        if not (1 <= num_RB_alloc <= self.max_RBs_per_UE):
            return False, f"RB_out_of_range ({num_RB_alloc})"
        if int(self.RB_remaining) < int(num_RB_alloc):
            return False, "insufficient RB remaining"

        if not isinstance(power_level_alloc, (int, float, np.floating)):
            return False, "invalid power_level_alloc_type"
        if not np.isfinite(power_level_alloc) or power_level_alloc <= 0:
            return False, "invalid power_level_alloc_value"
        if self.RU_power_remaining[RU_choice] + eps < float(power_level_alloc):
            return False, "insufficient RU power"

        # Optional: codebook công suất phải khớp
        if self.P_ib_sk_val and (power_level_alloc not in self.P_ib_sk_val):
            return False, "power_level_not_in_codebook"

        # ---- Topology links ----
        if self.l_ru_du.shape != (self.num_RUs, self.num_DUs):
            return False, "invalid l_ru_du shape"
        if self.l_du_cu.shape != (self.num_DUs, self.num_CUs):
            return False, "invalid l_du_cu shape"
        if self.l_ru_du[RU_choice, DU_choice] == 0:
            return False, "no RU-DU link"
        if self.l_du_cu[DU_choice, CU_choice] == 0:
            return False, "no DU-CU link"

        # ---- PHY throughput & SINR ----
        try:
            power_per_RB = float(power_level_alloc) / float(num_RB_alloc)
            SNR_per_RB   = power_per_RB * float(self.gain[RU_choice, UE_idx])  # linear SNR
            if not np.isfinite(SNR_per_RB) or SNR_per_RB < 0.0:
                return False, "invalid SNR value"
            if SNR_per_RB + eps < SINR_min_lin:
                return False, "SINR_below_min"
            data_rate = float(num_RB_alloc) * float(self.bandwidth_per_RB) * np.log2(1.0 + SNR_per_RB)  # bps
        except Exception as e:
            return False, f"throughput_calc_error: {e}"

        if data_rate + eps < R_min:
            return False, f"insufficient_throughput ({data_rate:.2f} < {R_min:.2f})"

        # ---- DU/CU CPU budgets (cycles/s) ----
        cpu_DU_req = self.k_DU * R_min * (1.0 + eta)
        cpu_CU_req = self.k_CU * R_min * (1.0 + eta)
        if self.DU_remaining[DU_choice] + eps < cpu_DU_req:
            return False, "insufficient DU resource"
        if self.CU_remaining[CU_choice] + eps < cpu_CU_req:
            return False, "insufficient CU resource"

        # ---- Latency (dựa vào mô hình tiền ánh xạ đã build) ----
        try:
            s_name = UE["slice"]
            s_idx  = self.slice_names.index(s_name)

            prop = float(self.propagation_delay[RU_choice, UE_idx])
            tx   = float(self.transmission_delay[UE_idx])
            proc_du = float(self.processing_delay_DU[DU_choice, s_idx, UE_idx])
            proc_cu = float(self.processing_delay_CU[CU_choice, s_idx, UE_idx])
            q_du    = float(self.queuing_delay_DU[DU_choice, s_idx, UE_idx])
            q_cu    = float(self.queuing_delay_CU[CU_choice, s_idx, UE_idx])

            L_total = prop + tx + proc_du + proc_cu + q_du + q_cu
            L_parts = {
                "propagation_delay": prop,
                "transmission_delay": tx,
                "processing_delay_DU": proc_du,
                "processing_delay_CU": proc_cu,
                "queuing_delay_DU": q_du,
                "queuing_delay_CU": q_cu,
            }
        except Exception as e:
            return False, f"latency_indexing_error: {e}"

        if not np.isfinite(L_total):
            return False, "latency_nan_or_inf"
        if L_total > L_max:
            return False, f"latency_violation ({L_total:.6f}s > {L_max:.6f}s)"

        return True, (data_rate, cpu_DU_req, cpu_CU_req, float(L_total), L_parts)

    # ======================================================================
    # Commit (apply resources & persist)
    # ======================================================================
    def update_network(self, UE_idx, RU_choice, DU_choice, CU_choice,
                       num_RB_alloc, power_level_alloc,
                       throughput_bps, cpu_DU_req, cpu_CU_req, delay_total, delay_parts):
        """Apply the feasible allocation; mutate resources & UE record."""
        self.RU_power_remaining[RU_choice] -= float(power_level_alloc)
        self.DU_remaining[DU_choice]       -= float(cpu_DU_req)
        self.CU_remaining[CU_choice]       -= float(cpu_CU_req)
        self.RB_remaining                  -= int(num_RB_alloc)

        UE = self.UE_requests[UE_idx]
        UE["status"].update({"active": 0, "accepted": True, "reason": "accepted_success"})
        UE["alloc"].update({
            "RU": int(RU_choice), "DU": int(DU_choice), "CU": int(CU_choice),
            "num_RB_alloc": int(num_RB_alloc), "power_alloc": float(power_level_alloc),
            "throughput_bps": float(throughput_bps), "delay_s": float(delay_total),
            "delay_parts": delay_parts
        })

    # ======================================================================
    # RL step
    # ======================================================================
    def step(self, action):
        """
        action = (UE_id, accept_flag, ru_sel, du_sel, cu_sel, num_RB_alloc, power_alloc)
        returns: (state, reward, done, info)
        """
        if not (isinstance(action, (list, tuple, np.ndarray)) and len(action) == 7):
            return self.get_state(), -0, self.check_done(), {"msg": "action length must be 7"}

        try:
            UE_idx       = int(action[0])
            accept_flag  = int(action[1])
            RU_choice    = int(action[2])
            DU_choice    = int(action[3])
            CU_choice    = int(action[4])
            num_RB_alloc = int(action[5])
            power_alloc  = float(action[6])
        except Exception as e:
            return self.get_state(), 0, self.check_done(), {"msg": f"invalid action types: {e}"}

        # Mark pending mask
        if UE_idx in self.pending_UEs:
            self.pending_UEs[UE_idx]["active"] = 0

        UE = self.UE_requests[UE_idx] if 0 <= UE_idx < self.num_UEs else None
        if UE is None:
            return self.get_state(), 0, self.check_done(), {"msg": "invalid UE_idx"}
        if int(UE["status"]["active"]) != 1:
            return self.get_state(), 5, self.check_done(), {"msg": "UE already inactive"}

        # ----- Reject path -----
        if accept_flag == 0:
            UE["status"].update({"active": 0, "accepted": False, "reason": "rejected_by_agent"})
            reward = 0
            return self.get_state(), 0, self.check_done(), {"success": False, "reason": "reject", "reward": reward}

        # ----- Accept path → feasibility + commit -----
        feasible, msg = self.check_feasible(UE_idx, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc)
        if not feasible:
            UE["status"].update({"active": 0, "accepted": False, "reason": str(msg)})
            reward = 0
            return self.get_state(), reward, self.check_done(), {"success": False, "reason": msg, "reward": reward}

        throughput_bps, cpu_DU_req, cpu_CU_req, delay_total, delay_parts = \
            float(msg[0]), float(msg[1]), float(msg[2]), float(msg[3]), msg[4]

        self.update_network(UE_idx, RU_choice, DU_choice, CU_choice,
                            num_RB_alloc, power_alloc, throughput_bps,
                            cpu_DU_req, cpu_CU_req, delay_total, delay_parts)

        # ======== REWARD (throughput ↑, delay ↓, accept) ========
        reward = self.compute_reward(UE, throughput_bps, delay_total)

        info = {
            "success": True, "reason": "accepted_success",
            "throughput_bps": throughput_bps, "reward": reward,
            "num_RB_alloc": num_RB_alloc, "power_level_alloc": power_alloc,
            "cpu_DU_req": cpu_DU_req, "cpu_CU_req": cpu_CU_req,
            "delay_s": delay_total, "delay_parts": delay_parts,
        }
        return self.get_state(), reward, self.check_done(), info

    # ======================================================================
    # Termination
    # ======================================================================
    def check_done(self):
        """Episode ends if no pending UE remains or any global resource depleted."""
        if sum(self.pending_UEs[i]["active"] for i in self.pending_UEs) == 0: return True
        if int(self.RB_remaining) <= 0: return True
        if np.all(self.RU_power_remaining <= 0): return True
        if np.all(self.DU_remaining <= 0): return True
        if np.all(self.CU_remaining <= 0): return True
        return False
