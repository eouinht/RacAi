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
    def __init__(self, 
                 total_nodes, 
                 num_RUs, 
                 num_DUs, 
                 num_CUs, 
                 num_RBs, 
                 num_UEs, 
                 SLICE_PRESET,
                 P_i_random_list,
                 A_j_random_list,
                 A_m_random_list, 
                 bw_ru_du_random_list, 
                 bw_du_cu_random_list, 
                 bandwidth_per_RB, 
                 max_RBs_per_UE, 
                 P_ib_sk_val, 
                 k_DU, 
                 k_CU,
                 dynamic_mode=False,
                 min_ues = 45,
                 max_ues= 55,
                 mobility_step = 15.0,
                 prb_cap_per_ru = None
                 ):

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

        # ------------------- Dynamic mode config ----------------------------
        self.dynamic_mode = bool(dynamic_mode)
        self.min_active_ues = min_ues
        self.max_active_ues = max_ues
        self.mobility_step = float(mobility_step)
        self.time_step = 0
        
        if prb_cap_per_ru is None:
            base = self.num_RBs // self.num_RUs
            extra = self.num_RBs % num_RUs
            self.prb_cap_per_ru = np.array(
                [base + (1 if i < extra else 0) for i in range(self.num_RUs)],
                dtype=int
            )
        else:
            self.prb_cap_per_ru = np.array(prb_cap_per_ru, dtype=int)
            if self.prb_cap_per_ru.shape[0] != self.num_RUs:
                raise ValueError("prb_cap_per_ru must have  length num_RUs")
            
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
        G = RAN_topo.create_topo(
            self.num_RUs, 
            self.num_DUs, 
            self.num_CUs, 
            self.P_i_random_list, 
            self.A_j_random_list, 
            self.A_m_random_list, 
            self.bw_ru_du_random_list, 
            self.bw_du_cu_random_list
            )
        self.RU_power_cap, self.DU_cap, self.CU_cap = RAN_topo.get_node_cap(G)  # [W], [cycles/s], [cycles/s]
        self.link_bw_ru_du_bps, self.link_bw_du_cu_bps = RAN_topo.get_links(G)                        # (RU,DU), (DU,CU) capacities
        
        # đồng bộ: dùng mặt nạ nhị phân 0/1
        self.l_ru_du = (np.array(self.link_bw_ru_du_bps, dtype=float) > 0).astype(int)
        self.l_du_cu = (np.array(self.link_bw_du_cu_bps, dtype=float) > 0).astype(int)

        # -------------------- Geometry & Channel --------------------
        self.coordinates_RU = gen_RU_UE.gen_coordinates_RU(self.num_RUs)
        self.coordinates_UE = gen_RU_UE.gen_coordinates_UE(self.num_UEs)
        self.distances_RU_UE = gen_RU_UE.calculate_distances(self.coordinates_RU, 
                                                             self.coordinates_UE, 
                                                             self.num_RUs, 
                                                             self.num_UEs)
        self.gain = wireless.channel_gain(self.distances_RU_UE, 
                                          self.num_RUs, 
                                          self.num_UEs,
                                          self.bandwidth_per_RB)  # linear

        # -------------------- UE Presets & Per-UE State --------------------
        self.UE_slice_name = gen_RU_UE.gen_UE_requirements(self.num_UEs, 
                                                           self.SLICE_PRESET)
        self.UE_requests = {i: self._build_ue_record(i, self.coordinates_UE[i], self.UE_slice_name[i])
                            for i in range(self.num_UEs)}

        self.propagation_delay, self.transmission_delay, self.processing_delay_DU, self.processing_delay_CU, \
            self.queuing_delay_DU, self.queuing_delay_CU = latency.build_latency_model(
                self.num_RUs,
                self.num_DUs,
                self.num_CUs,
                self.num_UEs,
                self.distances_RU_UE,
                self.SLICE_PRESET,
                self.UE_slice_name,
                np.asarray(self.DU_cap, dtype=float),
                np.asarray(self.CU_cap, dtype=float),
            )
        # -------------------- Init dynamic resources --------------------
        self.reset_env()
        
    def _build_ue_record(self,
                         ue_id,
                         pos_xy,
                         slice_name):
        
        pkt_bits = float(self.SLICE_PRESET[str(slice_name)]["packet_size_bits"])
        base_lambda = float(self.SLICE_PRESET[str(slice_name)]["lambda_default_pps"])
        lam = np.random.uniform(0.8 * base_lambda, 1.2 * base_lambda)
        
        return{
            "id": int(ue_id),
            "slice": str(slice_name),
            **copy.deepcopy(self.SLICE_PRESET[str(slice_name)]),
            "status": {
                "active": 1, 
                "accepted": None, 
                "reason": None,
                "served": 0,     #da duoc cap tai nguyen hay chua
                "is_new": 1,
                "is_ho_candidate": 0,
            },
            "mobility": {
                "x": float(pos_xy[0]),
                "y": float(pos_xy[1]),
                "vx": 0.0,
                "vy": 0.0,
            },   
            "traffic": {
                "arrival_step":0,
                "remaining_time": 0,    # Thoi gian song cu UE nay
                "packet_size_bits": pkt_bits,
                "lambda_pps": lam,
                "arrival_rate_bps": lam*pkt_bits,
                "queue_bits": 0.0,
            },
            "alloc": {
                "RU": None, 
                "DU": None, 
                "CU": None,
                "num_RB_alloc": 0,
                "power_alloc": 0.0,
                "throughput_bps": 0.0, 
                "delay_s": 0.0,
                "delay_parts": None
            },
            "ho_info": {
                "last_ho_step": -1, # lần cuối HO là khi nào
                "num_ho": 0,        # Số lần HO 
            },
        }
        
    def update_traffic(self, step_duration_s=1.0):
        for ue_id, ue in self.UE_requests.items():
            if int(ue["status"]["active"]) != 1:
                continue

            lam = float(ue["traffic"]["lambda_pps"])
            pkt_bits = float(ue["traffic"]["packet_size_bits"])

            # số packet đến trong step
            num_packets = np.random.poisson(lam * step_duration_s)
            new_bits = num_packets * pkt_bits

            ue["traffic"]["queue_bits"] += float(new_bits)
            
    def _remaining_time(self):
        return int(np.random.normal(25,46))
    
    def _check_prb_consistency(self, context=""):
        total_prb = int(np.sum(self.PRB_remaining_per_RU))
        
        if int(self.RB_remaining) != total_prb:
            raise RuntimeError(
                f"[PRB ERROR] {context} mismatch: RB_remaining={self.RB_remaining}, "
                f"sum(PRB_remaining_per_RU)={total_prb}, PRB_remaining_per_RU={self.PRB_remaining_per_RU}"
            )
        if np.any(self.PRB_remaining_per_RU < 0):
            raise RuntimeError(
                f"[PRB ERROR] {context} negative PRB per RU: {self.PRB_remaining_per_RU}"
            )


    def _refresh_channel_and_latency(self):
        
        self.distances_RU_UE = gen_RU_UE.calculate_distances(self.coordinates_RU,
                                                             self.coordinates_UE,
                                                             self.num_RUs,
                                                             self.num_UEs)
        self.gain = wireless.channel_gain(self.distances_RU_UE,
                                          self.num_RUs,
                                          self.num_UEs,
                                          self.bandwidth_per_RB)
        self.propagation_delay, self.transmission_delay, self.processing_delay_DU, self.processing_delay_CU, \
            self.queuing_delay_DU, self.queuing_delay_CU = latency.build_latency_model(
                self.num_RUs,
                self.num_DUs,
                self.num_CUs,
                self.num_UEs,
                self.distances_RU_UE,
                self.SLICE_PRESET,
                self.UE_slice_name,
                np.asarray(self.DU_cap, dtype=float),
                np.asarray(self.CU_cap, dtype=float),
            )
    # ======================================================================
    # Basics
    # ======================================================================
    def reset_env(self):
        """Reset remaining resources & clear all UE allocations."""
        
        self.PRB_remaining_per_RU = np.copy(self.prb_cap_per_ru).astype(int)
        print(f"check prb_per_rus{self.PRB_remaining_per_RU}")
        self.RB_remaining        = int(np.sum(self.PRB_remaining_per_RU))
        self.RU_power_remaining  = np.copy(self.RU_power_cap).astype(float)
        self.DU_remaining        = np.copy(self.DU_cap).astype(float)
        self.CU_remaining        = np.copy(self.CU_cap).astype(float)

        for i in range(self.num_UEs):
            ue = self.UE_requests[i]
            ue["status"].update({
                "active": 1, 
                "accepted": None, 
                "reason": None,
                "served": 0,
                "is_new": 1,
                "is_ho_candidate": 0})
            
            ue["traffic"].update({
                "arrival_step":0,
                "remaining_time": 0,
            })
            ue["alloc"].update({
                "RU": None, 
                "DU": None, 
                "CU": None,
                "num_RB_alloc": 0, 
                "power_alloc": 0.0,
                "throughput_bps": 0.0, 
                "delay_s": 0.0,
                "delay_parts": None
            })
            ue["ho_info"].update({
                "last_ho_step": -1,
                "num_ho":0})
            
        if self.dynamic_mode:
            self._reset_dynamic_activity()
        else:
            self.pending_UEs = {i: {"active": 1} for i in range(self.num_UEs)} # Danh sach UE chờ đc xử lý

            
        self.done = False
        self.time_step = 0
        self._check_prb_consistency("after reset")
        return self.get_state()
    
    def _reset_dynamic_activity(self):
        start_active = min(50, self.max_active_ues)
        # start_active = min(start_active, self.num_UEs)
        active_ids = set(np.random.choice(self.num_UEs, size=start_active, replace=False).tolist())

        self.pending_UEs = {}
        for i in range(self.num_UEs):
            ue = self.UE_requests[i]
            is_active = 1 if i in active_ids else 0
            
            ue["status"]["active"] = is_active
            ue["status"]["is_new"] = is_active
            ue["status"]["accepted"] = None
            ue["status"]["served"] = 0
            ue["status"]["is_ho_candidate"] = 0

            if is_active:
                ue["traffic"]["arrival_step"] = 0
                ue["traffic"]["remaining_time"] = self._remaining_time()
            else:
                ue["traffic"]["arrival_step"] = 0
                ue["traffic"]["remaining_time"] = 0
                
            self.pending_UEs[i] = {"active": is_active}
    
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
            "RB_remaining":       int(self.RB_remaining),
            "PRB_remaining_per_RU": np.copy(self.PRB_remaining_per_RU),
            "time_step": int(self.time_step),
            "num_active_ues": int(sum(self.UE_requests[i]["status"]["active"] for i in range(self.num_UEs))),
     
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
                "served": int(ue["status"].get("served", 0)),
                "is_new": int(ue["status"].get("is_new", 0)),
                "is_ho_candidate": int(ue["status"].get("is_ho_candidate", 0)),
                "remaining_time": int(ue["traffic"].get("remaining_time", 0)),
                "x": float(ue["mobility"]["x"]),
                "y": float(ue["mobility"]["y"]),
                "RU": None if ue["alloc"]["RU"] is None else int(ue["alloc"]["RU"]),
                "DU": None if ue["alloc"]["DU"] is None else int(ue["alloc"]["DU"]),
                "CU": None if ue["alloc"]["CU"] is None else int(ue["alloc"]["CU"]),
                "num_RB_alloc": int(ue["alloc"]["num_RB_alloc"]),
                "power_alloc": float(ue["alloc"]["power_alloc"]),
                "throughput_bps": float(ue["alloc"]["throughput_bps"]),
                "delay_s": float(ue["alloc"]["delay_s"]),
                "queue_bits": float(ue["traffic"].get("queue_bits", 0.0)),
                "lambda_pps": float(ue["traffic"].get("lambda_pps", 0.0)),
                "arrival_rate_bps": float(ue["traffic"].get("arrival_rate_bps", 0.0)),
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
    # tính toán resource (feasibility + requirement) là:
    def check_feasible(self, 
                       UE_idx, 
                       RU_choice, 
                       DU_choice, 
                       CU_choice, 
                       num_RB_alloc, 
                       power_level_alloc,
                       step_duration_s=0.1):
        """
        Returns:
            (True, (rate, cpu_DU_req, cpu_CU_req, delay_total, delay_parts))
            hoặc (False, reason).
        """
        eps = 1e-9

        # ---- Index & activity ----
        try:
            if not (0 <= UE_idx < self.num_UEs):   
                return False, "invalid UE_idx"
            if not (0 <= RU_choice < self.num_RUs): 
                return False, "invalid RU_choice"
            if not (0 <= DU_choice < self.num_DUs): 
                return False, "invalid DU_choice"
            if not (0 <= CU_choice < self.num_CUs): 
                return False, "invalid CU_choice"
            
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

        queue_bits = float(UE["traffic"].get("queue_bits", 0.0))
        arrival_rate_bps = float(UE["traffic"].get("arrival_rate_bps", 0.0))
        pkt_bits = float(UE["traffic"].get("packet_size_bits", 0.0))
        cycles_per_packet = float(UE.get("cycles_per_packet", 0.0))
        required_rate = max(
            R_min,
            arrival_rate_bps,
            queue_bits / max(step_duration_s, eps)
        )
        
            # ---- RB / Power availability ----
        if not isinstance(num_RB_alloc, (int, np.integer)):
            return False, "invalid num_RB_alloc_type"
        if not (1 <= num_RB_alloc <= self.max_RBs_per_UE):
            return False, f"RB_out_of_range ({num_RB_alloc})"
        
        if int(self.RB_remaining) < int(num_RB_alloc):
            return False, "insufficient RB remaining"
        if int(self.PRB_remaining_per_RU[RU_choice]) < int(num_RB_alloc):
            return False, "insufficient PRB at selected RU"

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

        if data_rate + eps < required_rate:
            return False, (
            f"insufficient_throughput "
            f"({data_rate:.2f} < required {required_rate:.2f}, "
            f"R_min={R_min:.2f}, queue_bits={queue_bits:.2f}, arrival_rate={arrival_rate_bps:.2f})"
        )
            
        
        # ---- DU/CU CPU budgets ----
        try:
            if pkt_bits > 0 and cycles_per_packet > 0:
                packet_rate_served = data_rate / pkt_bits
                cpu_req = packet_rate_served * cycles_per_packet * (1.0 + eta)

                # chia mềm DU/CU theo hệ số k_DU, k_CU
                cpu_DU_req = float(self.k_DU) * cpu_req
                cpu_CU_req = float(self.k_CU) * cpu_req
            else:
                # fallback về logic cũ nếu thiếu field
                cpu_DU_req = self.k_DU * required_rate * (1.0 + eta)
                cpu_CU_req = self.k_CU * required_rate * (1.0 + eta)
        except Exception as e:
            return False, f"cpu_calc_error: {e}"

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

    def check_feadible_handover(self,
                                UE_idx,
                                target_RU,
                                target_DU,
                                target_CU,
                                target_num_RB,
                                target_power):
        UE = self.UE_requests[UE_idx]
        old_ru = UE["alloc"]["RU"]
        old_du = UE["alloc"]["DU"]
        old_cu = UE["alloc"]["CU"]
        old_prb = int(UE["alloc"]["num_RB_alloc"])
        old_ptx = float(UE["alloc"]["power_alloc"])
        old_du_req = self.k_DU*float(UE.get("R_min", 0.0))*(1.0 + float(UE.get("eta_slice", 0.0)))
        old_cu_req = self.k_CU * float(UE.get("R_min", 0.0)) * (1.0 + float(UE.get("eta_slice", 0.0)))

        if old_ru is not None:
            self.RB_remaining += old_prb
            self.PRB_remaining_per_RU[old_ru] += old_prb
            self.RU_power_remaining[old_ru] += old_ptx
        if old_du is not None:
            self.DU_remaining[old_du] += old_du_req
        if old_cu is not None:
            self.CU_remaining[old_cu] += old_cu_req
            
        feasible, msg = self.check_feasible(
            UE_idx, 
            target_RU, 
            target_DU, 
            target_CU, 
            target_num_RB,
            target_power
        )
        
        if old_ru is not None:
            self.RB_remaining -= old_prb
            self.PRB_remaining_per_RU[old_ru] -= old_prb
            self.RU_power_remaining[old_ru] -= old_ptx
        if old_du is not None:
            self.DU_remaining[old_du] -= old_du_req
        if old_cu is not None:
            self.CU_remaining[old_cu] -= old_cu_req
            
        return feasible, msg

    # ======================================================================
    # Commit (apply resources & persist)
    # ======================================================================
    def update_network(self, UE_idx, RU_choice, DU_choice, CU_choice,
                       num_RB_alloc, power_level_alloc,
                       throughput_bps, cpu_DU_req, cpu_CU_req, 
                       delay_total, delay_parts):
        
        """Apply the feasible allocation; mutate resources & UE record."""
        self.RU_power_remaining[RU_choice] -= float(power_level_alloc)
        self.DU_remaining[DU_choice]       -= float(cpu_DU_req)
        self.CU_remaining[CU_choice]       -= float(cpu_CU_req)
        self.RB_remaining                  -= int(num_RB_alloc)
        self.PRB_remaining_per_RU[RU_choice]          -= int(num_RB_alloc)
        
        UE = self.UE_requests[UE_idx]
        UE["status"].update({
            "active": 1 if self.dynamic_mode else 0, 
            "accepted": True, 
            "reason": "accepted_success",
            "served": 1,
            "is_new": 0})
        
        UE["alloc"].update({
            "RU": int(RU_choice), 
            "DU": int(DU_choice), 
            "CU": int(CU_choice),
            "num_RB_alloc": int(num_RB_alloc), 
            "power_alloc": float(power_level_alloc),
            "throughput_bps": float(throughput_bps), 
            "delay_s": float(delay_total),
            "delay_parts": delay_parts
        })
        
        served_bits = float(throughput_bps) * 1.0
        UE["traffic"]["queue_bits"] = max(
            0.0,
            float(UE["traffic"].get("queue_bits", 0.0)) - served_bits
        )
        
        self._check_prb_consistency(f"after update UE {UE_idx}")
        
    def release_ue(self, UE_idx, reason="departed"):
        UE = self.UE_requests[UE_idx]
        old_ru = UE["alloc"]["RU"]
        old_du = UE["alloc"]["DU"]
        old_cu = UE["alloc"]["CU"]
        old_prb = int(UE["alloc"]["num_RB_alloc"])
        old_ptx = float(UE["alloc"]["power_alloc"])
        eta = float(UE.get("eta_slice", 0.0))
        R_min = float(UE.get("R_min", 0.0))
        old_du_req = self.k_DU * R_min * (1.0 + eta)
        old_cu_req = self.k_CU * R_min * (1.0 + eta)
        
        if old_ru is not None:
            self.RB_remaining += old_prb
            self.PRB_remaining_per_RU[old_ru] += old_prb
            self.RU_power_remaining[old_ru] += old_ptx
        if old_du is not None:
            self.DU_remaining[old_du] += old_du_req
        if old_cu is not None:
            self.CU_remaining[old_cu] += old_cu_req
            
        UE["status"].update({
            "active": 0,
            "accepted": None,
            "reason": reason,
            "served": 0,
            "is_new": 0,
            "is_ho_candidate": 0,
        })
        UE["traffic"]["remaining_time"] = 0
        UE["alloc"].update({
            "RU": None,
            "DU": None,
            "CU": None,
            "num_RB_alloc": 0,
            "power_alloc": 0.0,
            "throughput_bps": 0.0,
            "delay_s": 0.0,
            "delay_parts": None,
        })
        self.pending_UEs[UE_idx]["active"] = 0
        self._check_prb_consistency(f"after release UE {UE_idx}")

    def apply_handover(self,
                       UE_idx,
                       target_RU,
                       target_DU,
                       target_CU,
                       target_num_RB,
                       target_power,
                       throughput_bps,
                       cpu_DU_req,
                       cpu_CU_req,
                       delay_total,
                       delay_parts):   
        
        self.release_ue(UE_idx=UE_idx, reason="handover_release")
        
        self.UE_requests[UE_idx]["status"]["active"] = 1
        
        self.UE_requests[UE_idx]["traffic"]["remaining_time"] = max(
            1, int(self.UE_requests[UE_idx]["traffic"].get("remaining_time", 1))
        )
        self.update_network(
            UE_idx, 
            target_RU, 
            target_DU, 
            target_CU,
            target_num_RB, 
            target_power,
            throughput_bps, 
            cpu_DU_req, 
            cpu_CU_req,
            delay_total, 
            delay_parts
        )
        self.UE_requests[UE_idx]["ho_info"]["last_ho_step"] = int(self.time_step)
        self.UE_requests[UE_idx]["ho_info"]["num_ho"] += 1
        self._check_prb_consistency(f"after apply handover UE {UE_idx}")

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
            UE["status"].update({"accepted": False, 
                                 "reason": "rejected_by_agent",
                                 "is_new": 0})
            
            if not self.dynamic_mode:
                UE["status"]["active"] = 0
            return self.get_state(), 0, self.check_done(), {"success": False, 
                                                            "reason": "reject", 
                                                            "reward": 0}


        # ----- Accept path → feasibility + commit -----
        feasible, msg = self.check_feasible(UE_idx, 
                                            RU_choice, 
                                            DU_choice, 
                                            CU_choice, 
                                            num_RB_alloc,
                                            power_alloc)
        if not feasible:
            UE["status"].update({"accepted": False, 
                                 "reason": str(msg),
                                 "is_new": 0})
            reward = 0
            return self.get_state(), reward, self.check_done(), {"success": False, 
                                                                 "reason": msg, 
                                                                 "reward": reward}

        throughput_bps, cpu_DU_req, cpu_CU_req, delay_total, delay_parts = \
            float(msg[0]), float(msg[1]), float(msg[2]), float(msg[3]), msg[4]

        self.update_network(UE_idx, RU_choice, DU_choice, CU_choice,
                            num_RB_alloc, power_alloc, throughput_bps,
                            cpu_DU_req, cpu_CU_req, delay_total, delay_parts)

        # ======== REWARD (throughput ↑, delay ↓, accept) ========
        reward = self.compute_reward(UE, throughput_bps, delay_total)

        info = {
            "success": True, 
            "reason": "accepted_success",
            "throughput_bps": throughput_bps, 
            "reward": reward,
            "num_RB_alloc": num_RB_alloc, 
            "power_level_alloc": power_alloc,
            "cpu_DU_req": cpu_DU_req, 
            "cpu_CU_req": cpu_CU_req,
            "delay_s": delay_total, 
            "delay_parts": delay_parts,
        }
        return self.get_state(), reward, self.check_done(), info

    # Cap nhat vi tri UE theo thoi gian
    def move_active_ues(self):
        for ue_id, ue in self.UE_requests.items():
            if int(ue["status"]["active"]) != 1:
                continue
            dx = np.random.uniform(-self.mobility_step, self.mobility_step)
            dy = np.random.uniform(-self.mobility_step, self.mobility_step)
            x = ue["mobility"]["x"] + dx 
            y = ue["mobility"]["y"] + dy
            ue["mobility"].update({"x": x,
                                   "y": y,
                                   "vx": dx,
                                   "vy": dy
                                   })
            self.coordinates_UE[ue_id] = (x, y)
    
    # Tim UE het thoi gian song de loai khoi mang 
    def sample_departures(self):
        departed_ids = []
        for ue_id, ue in self.UE_requests.items():
            if int(ue["status"]["active"]) != 1:
                continue
            ue["traffic"]["remaining_time"] -= 1
            if int(ue["traffic"]["remaining_time"]) <= 0:
                departed_ids.append(ue_id)
        return departed_ids
    
    # Add new ue vao mang khi UE khong du nguong min
    def add_new_ues(self, target_active_use=None):
        current_total = int(sum(
            self.UE_requests[i]["status"]["active"] for i in range(self.num_UEs)
        ))
        if current_total >= self.max_active_ues:
            return []
        
        inactive_ids = [
            i for i in range(self.num_UEs)
            if int(self.UE_requests[i]["status"]["active"]) == 0
        ]

        if len(inactive_ids) == 0:
            return []

        if current_total <= self.min_active_ues:
            num_new = target_active_use - current_total
        else:
            num_new = int((target_active_use - current_total)/2)
        
        # num_new = np.random.randint(1, max_new_per_step + 1)
        # num_new = min(num_new, len(inactive_ids))
        num_new = min(num_new, len(inactive_ids), self.max_active_ues - current_total)
        # print(f"num new: {num}")
        if num_new <= 0:
            return []    
        new_positions = gen_RU_UE.gen_coordinates_UE(num_new)
        new_slices = gen_RU_UE.gen_UE_requirements(num_new, self.SLICE_PRESET)
        chosen_ids = inactive_ids[:num_new]

        for local_idx, ue_id in enumerate(chosen_ids):
            self.coordinates_UE[ue_id] = new_positions[local_idx]
            self.UE_slice_name[ue_id] = new_slices[local_idx]
            self.UE_requests[ue_id] = self._build_ue_record(
                ue_id,
                new_positions[local_idx],
                new_slices[local_idx]
            )
            self.UE_requests[ue_id]["traffic"]["arrival_step"] = int(self.time_step)
            self.UE_requests[ue_id]["traffic"]["remaining_time"] = self._remaining_time()
            self.pending_UEs[ue_id]["active"] = 1
           

        return chosen_ids
    
    def get_filter_ues(self, sinr_margin_db=1.0):
        stable_ues = []
        ho_candidates = []
        new_ue_candidates = []

        for ue_id, ue in self.UE_requests.items():
            if int(ue["status"]["active"]) != 1:
                continue

            if int(ue["status"].get("is_new", 0)) == 1 or ue["alloc"]["RU"] is None:
                new_ue_candidates.append(ue_id)
                continue

            current_ru = int(ue["alloc"]["RU"])
            current_gain = float(self.gain[current_ru, ue_id])
            best_ru = int(np.argmax(self.gain[:, ue_id]))
            best_gain = float(self.gain[best_ru, ue_id])
            gain_ratio_db = 10.0 * np.log10((best_gain + 1e-12) / (current_gain + 1e-12))

            overloaded = self.PRB_remaining_per_RU[current_ru] <= 2
            if best_ru != current_ru and (gain_ratio_db >= sinr_margin_db or overloaded):
                ue["status"]["is_ho_candidate"] = 1
                ho_candidates.append(ue_id)
            else:
                ue["status"]["is_ho_candidate"] = 0
                stable_ues.append(ue_id)

        return stable_ues, ho_candidates, new_ue_candidates
    
    def run_stable_compaction(self, stable_ues):
        compact_map = {}
        for ue_id in stable_ues:
            ue = self.UE_requests[ue_id]
            compact_map[ue_id] = {
                "RU": ue["alloc"]["RU"],
                "DU": ue["alloc"]["DU"],
                "CU": ue["alloc"]["CU"],
                "num_RB_alloc": ue["alloc"]["num_RB_alloc"],
                "power_alloc": ue["alloc"]["power_alloc"],
            }
        return compact_map
    
    def advance_time(self, target_active_ues=None):
        self.time_step += 1

        departed_ues = self.sample_departures()
        for ue_id in departed_ues:
            self.release_ue(ue_id, reason="departed")

        self.move_active_ues()
        self.update_traffic()
        new_ue_ids = self.add_new_ues(target_active_ues)
        self._refresh_channel_and_latency()

        stable_ues, ho_candidates, new_ue_candidates = self.get_filter_ues()
        compact_alloc_map = self.run_stable_compaction(stable_ues)

        info = {
            "time_step": int(self.time_step),
            "departed_ues": departed_ues,
            "new_ue_ids": new_ue_ids,
            "stable_ues": stable_ues,
            "ho_candidates": ho_candidates,
            "new_ue_candidates": new_ue_candidates,
            "compact_alloc_map": compact_alloc_map,
        }
        return self.get_state(), info
    # ======================================================================
    # Termination
    # ======================================================================
    def check_done(self):
        """Episode ends if no pending UE remains or any global resource depleted."""
        if self.dynamic_mode:
            if int(sum(self.UE_requests[i]["status"]["active"] for i in range(self.num_UEs))) <= 0:
                return True
        else:
            if sum(self.pending_UEs[i]["active"] for i in self.pending_UEs) == 0:
                return True
        if int(self.RB_remaining) <= 0:
            return True
        if np.any(self.PRB_remaining_per_RU < 0):
            return True
        if np.all(self.RU_power_remaining <= 0):
            return True
        if np.all(self.DU_remaining <= 0):
            return True
        if np.all(self.CU_remaining <= 0):
            return True
        return False
