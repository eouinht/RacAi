import numpy as np
import cvxpy as cp
import time
import traceback as tb
import numpy as np
import random

SOLVER = cp.MOSEK
mosek_params = {
    # Hiệu năng & song song
    "MSK_IPAR_NUM_THREADS": 0,        # 0 = dùng tất cả CPU
    "MSK_IPAR_LOG": 0,                # tắt log để giảm IO

    # Heuristic & ổn định MIP
    "MSK_IPAR_MIO_HEURISTIC_LEVEL": 3,     # heuristic mạnh hơn
    "MSK_IPAR_MIO_LOCAL_BRANCH_NUMBER": 20,
    "MSK_IPAR_MIO_ROOT_OPTIMIZER": 4,      # concurrent root
    "MSK_IPAR_PRESOLVE_USE": 1,
    "MSK_DPAR_MIO_TOL_ABS_GAP": 0.0,
    "MSK_DPAR_MIO_TOL_REL_GAP": 5e-2,      # cho phép 5% để dừng sớm
    "MSK_DPAR_MIO_MAX_TIME": 900.0,        # 15 phút
    "MSK_IPAR_MIO_SEED": 1,                # tái lập
}


def Long_Doraemon(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, bandwidth_per_RB, gain, slice_mapping, max_tx_power_watts, delay_max, propagation_delay, transmission_delay, processing_delay_DU, processing_delay_CU, queuing_delay_DU, queuing_delay_CU, w_acc, w_thr, w_lat, slice_weight_accept, slice_thr_alpha, slice_lat_alpha):
    
    try:
        epsilon = 1e-9
        inv_log2 = 1.0 / np.log(2.0)

        # -------------------- Cấu hình công suất phân bổ --------------------
        long_P_ib_sk = np.max(max_tx_power_watts) / num_UEs 

        # 2) Hệ số thông lượng tuyến tính theo z
        coef_rate = bandwidth_per_RB * np.log1p(gain * long_P_ib_sk) * inv_log2  # (R,B,S,K)

        # -------------------- Khởi tạo biến nhị phân --------------------
        long_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        long_z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"long_z_ib_sk({i},{b},{s},{k})")

        long_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        long_phi_j_sk = np.empty((num_DUs, num_slices, num_UEs), dtype=object)
        long_phi_m_sk = np.empty((num_CUs, num_slices, num_UEs), dtype=object)

        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    long_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"long_phi_i_sk({i},{s},{k})")
                for j in range(num_DUs):
                    long_phi_j_sk[j, s, k] = cp.Variable(boolean=True, name=f"long_phi_j_sk({j},{s},{k})")
                for m in range(num_CUs):
                    long_phi_m_sk[m, s, k] = cp.Variable(boolean=True, name=f"long_phi_m_sk({m},{s},{k})")

        long_pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="long_pi_sk")

        constraints = []

        # -------------------- Ràng buộc 1 RB chỉ gán cho 1 UE --------------------
        for b in range(num_RBs):
            constraints.append(cp.sum([long_z_ib_sk[i, b, s, k] for i in range(num_RUs) for s in range(num_slices) for k in range(num_UEs)]) <= 1)

        # -------------------- Ràng buộc QoS (Data rate) --------------------
        long_R_expr = [[None for _ in range(num_UEs)] for __ in range(num_slices)]
        long_total_R_expr = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([coef_rate[i, b, s, k] * long_z_ib_sk[i, b, s, k]
                               for i in range(num_RUs) for b in range(num_RBs)])
                constraints.append(R_sk >= R_min[k] * long_pi_sk[s, k])
                long_R_expr[s][k] = R_sk
                long_total_R_expr += R_sk


        # -------------------- Tổng công suất của mỗi RU --------------------
        for i in range(num_RUs):
            total_power = cp.sum([long_z_ib_sk[i, b, s, k] * long_P_ib_sk
                                for b in range(num_RBs)
                                for s in range(num_slices)
                                for k in range(num_UEs)])
            constraints.append(total_power <= max_tx_power_watts[i])

        # -------------------- Tài nguyên DU / CU --------------------
        for j in range(num_DUs):
            total_du = cp.sum([long_phi_j_sk[j, s, k] * D_j[k]
                               for s in range(num_slices)
                               for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])

        for m in range(num_CUs):
            total_cu = cp.sum([long_phi_m_sk[m, s, k] * D_m[k]
                               for s in range(num_slices)
                               for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])

        # -------------------- Ánh xạ RU–DU–CU --------------------
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(cp.sum([long_phi_i_sk[i, s, k] for i in range(num_RUs)]) == long_pi_sk[s, k])
                constraints.append(cp.sum([long_phi_j_sk[j, s, k] for j in range(num_DUs)]) == long_pi_sk[s, k])
                constraints.append(cp.sum([long_phi_m_sk[m, s, k] for m in range(num_CUs)]) == long_pi_sk[s, k])
        
        # -------------------- Chuyển đổi z_ib_sk ↔ phi_i_sk --------------------
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([long_z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= long_phi_i_sk[i, s, k])
                    constraints.append(long_phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # -------------------- Liên kết topo RU–DU–CU --------------------
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    for j in range(num_DUs):
                        constraints.append(long_phi_j_sk[j, s, k] <= l_ru_du[i, j] - long_phi_i_sk[i, s, k] + 1)
                for j in range(num_DUs):
                    for m in range(num_CUs):
                        constraints.append(long_phi_m_sk[m, s, k] <= l_du_cu[j, m] - long_phi_j_sk[j, s, k] + 1)

        # -------------------- Slice mapping --------------------
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(long_pi_sk[s, k] <= slice_mapping[s, k])
        
        # ======= TÍNH DELAY (đúng shapes mô hình latency) =======
        long_UE_delay = np.empty(num_UEs, dtype=object)
        for k in range(num_UEs):

            prop_k = cp.sum([
                propagation_delay[i, k] * cp.sum([long_phi_i_sk[i, s, k] for s in range(num_slices)])
                for i in range(num_RUs)
            ])

            tx_k = transmission_delay[k] * cp.sum(long_pi_sk[:, k])

            proc_du_k = cp.sum([
                processing_delay_DU[j, s, k] * long_phi_j_sk[j, s, k]
                for j in range(num_DUs) for s in range(num_slices)
            ])

            proc_cu_k = cp.sum([
                processing_delay_CU[m, s, k] * long_phi_m_sk[m, s, k]
                for m in range(num_CUs) for s in range(num_slices)
            ])

            # 4) Queuing: DU + CU
            queue_du_k = cp.sum([
                queuing_delay_DU[j, s, k] * long_phi_j_sk[j, s, k]
                for j in range(num_DUs) for s in range(num_slices)
            ])
            queue_cu_k = cp.sum([
                queuing_delay_CU[m, s, k] * long_phi_m_sk[m, s, k]
                for m in range(num_CUs) for s in range(num_slices)
            ])

            long_UE_delay[k] = prop_k + tx_k + proc_du_k + proc_cu_k + queue_du_k + queue_cu_k

            for s in range(num_slices):
                constraints.append(long_UE_delay[k] <= delay_max[s] + (1 - long_pi_sk[s, k]) * 1e3)


        w_acc_col = slice_weight_accept.reshape((num_slices, 1))
        accept_term = cp.sum(cp.multiply(w_acc_col, long_pi_sk))

        thr_term = 0
        lat_pen  = 0
        for s in range(num_slices):
            thr_term += slice_thr_alpha[s] * cp.sum([long_R_expr[s][k] / max(R_min[k], 1e-9) for k in range(num_UEs)])
            lat_pen  += slice_lat_alpha[s] * cp.sum(cp.hstack([long_UE_delay[k] / max(delay_max[s], 1e-9) for k in range(num_UEs)]))
        
        # -------------------- Mục tiêu tối ưu --------------------
        objective = cp.Maximize(w_acc * accept_term  + w_thr * thr_term - w_lat * lat_pen)

        # -------------------- Giải bài toán --------------------
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, warm_start=True)

        # -------------------- Kết quả --------------------
        long_objective = objective.value
        long_total_pi_sk = cp.sum(long_pi_sk).value
        long_total_z_ib_sk = cp.sum([long_z_ib_sk[i, b, s, k]
            for i in range(num_RUs)
            for b in range(num_RBs)
            for s in range(num_slices)
            for k in range(num_UEs)
        ]).value
        long_total_P_ib_sk = long_total_z_ib_sk * long_P_ib_sk
        long_total_R_sk = float(long_total_R_expr.value)
        long_total_delay = np.sum([L.value for L in long_UE_delay])

        print("✅ MILP solved successfully.")
        return long_pi_sk, long_z_ib_sk, long_P_ib_sk, long_phi_i_sk, long_phi_j_sk, long_phi_m_sk, long_objective, long_total_pi_sk, long_total_R_sk, long_UE_delay, long_total_delay, long_total_z_ib_sk, long_total_P_ib_sk

    except cp.SolverError:
        print("❌ Solver error: non-feasible problem.")
        return (None,) * 13
    except Exception as e:
        print(f"⚠️ Error in MILP solver: {e}")
        return (None,) * 13
    


def Short_Doraemon(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, bandwidth_per_RB, gain, slice_mapping, max_tx_power_watts, delay_max, propagation_delay, transmission_delay, processing_delay_DU, processing_delay_CU, queuing_delay_DU, queuing_delay_CU, w_acc, w_thr, w_lat, slice_weight_accept, slice_thr_alpha, slice_lat_alpha, arr_long_pi_sk, arr_long_z_ib_sk, arr_long_phi_i_sk, arr_long_phi_j_sk, arr_long_phi_m_sk, long_UE_delay):
    try:
        # --------- 1) Chuẩn hóa input cố định thành float mask 0/1 ----------
        short_pi_sk    = np.asarray(arr_long_pi_sk,    dtype=float)
        short_z_ib_sk     = np.asarray(arr_long_z_ib_sk,  dtype=float)
        short_phi_i_sk = np.asarray(arr_long_phi_i_sk, dtype=float)
        short_phi_j_sk = np.asarray(arr_long_phi_j_sk, dtype=float)
        short_phi_m_sk = np.asarray(arr_long_phi_m_sk, dtype=float)
        short_UE_delay_list = []
        for k in range(num_UEs):
            v = long_UE_delay[k]
            v = getattr(v, "value", v)  # nếu là Expression thì lấy .value
            short_UE_delay_list.append(float(v))
        short_UE_delay = np.array(short_UE_delay_list, dtype=float)   # (K,)

        # ---------- 2) Biến quyết định: P_{i,b,s,k} (chỉ tạo khi z=1) ----------
        short_P_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_P_ib_sk[i,b,s,k] = cp.Variable(nonneg=True, name=f"P({i},{b},{s},{k})")
                        

        constraints = []
        # --------- 3) QoS (Data rate) – DCP-correct ----------
        R_expr = [[None for _ in range(num_UEs)] for _ in range(num_slices)]
        short_total_R_sk = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([
                    bandwidth_per_RB * cp.log1p(gain[i, b, s, k] * short_P_ib_sk[i, b, s, k] * short_z_ib_sk[i, b, s, k]) / np.log(2)
                    for i in range(num_RUs) for b in range(num_RBs)
                ])
                constraints.append(R_sk >= R_min[k] * short_pi_sk[(s, k)])
                R_expr[s][k] = R_sk
                short_total_R_sk += R_sk

        # --------- 4) Ngân sách công suất theo RU ----------
        for i in range(num_RUs):
            power_i = cp.sum([short_z_ib_sk[i, b, s, k] * short_P_ib_sk[i, b, s, k] 
                      for b in range(num_RBs) 
                      for k in range(num_UEs) 
                      for s in range(num_slices)])
            # Thêm từng ràng buộc riêng lẻ
            constraints.append(power_i <= max_tx_power_watts[i])

        # --------- 5) Latency & DU/CU cho điểm số ----------

        w_acc_col = slice_weight_accept.reshape((num_slices, 1))
        accept_term = cp.sum(cp.multiply(w_acc_col, short_pi_sk))

        thr_term = 0
        lat_pen  = 0
        for s in range(num_slices):
            thr_term += slice_thr_alpha[s] * cp.sum([R_expr[s][k] / max(R_min[k], 1e-9) for k in range(num_UEs)])
            lat_pen  += slice_lat_alpha[s] * cp.sum([short_UE_delay[k] / max(delay_max[s], 1e-9) for k in range(num_UEs)])
        
        # -------------------- Mục tiêu tối ưu --------------------
        objective = cp.Maximize(w_acc * accept_term + w_thr * thr_term - w_lat * lat_pen)

        # --------- 7) Giải bài toán & kiểm tra status ----------
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, warm_start=True)

        # --------- 8) Kết quả ----------
        short_objective = objective.value
        short_total_pi_sk = cp.sum(short_pi_sk).value
        short_total_z_ib_sk = cp.sum([short_z_ib_sk[i, b, s, k]
            for i in range(num_RUs)
            for b in range(num_RBs)
            for s in range(num_slices)
            for k in range(num_UEs)
        ])

        short_total_P_ib_sk = cp.sum([short_z_ib_sk[i, b, s, k] * short_P_ib_sk[i, b, s, k]
            for i in range(num_RUs)
            for b in range(num_RBs)
            for s in range(num_slices)
            for k in range(num_UEs)
        ]).value
        short_total_R_sk = short_total_R_sk.value
        short_total_delay = np.sum(short_UE_delay)


        return short_pi_sk, short_z_ib_sk, short_P_ib_sk, short_phi_i_sk, short_phi_j_sk, short_phi_m_sk, short_objective, short_total_pi_sk, short_total_R_sk, short_UE_delay, short_total_delay, short_total_z_ib_sk, short_total_P_ib_sk

    except cp.SolverError:
        print("❌ Solver error: non-feasible problem.")
        return (None,) * 13
    except Exception as e:
        print(f"⚠️ Error in Short_Doraemon: {e}")
        return (None,) * 13
    



