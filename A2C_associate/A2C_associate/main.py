import os
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path

from config import *
from Env.network_env import NetworkEnv
from model.ppo_graphSAGE_MLP_agent import PPOAgent, FullPolicy, train_agent, evaluate_agent, save_checkpoint, load_checkpoint
# Thêm import ở đầu file main
from baseline import run_all_baselines

import solving_MILP_2
import other_function

# -----------------------------
# 1) HUẤN LUYỆN PPO (train)
# -----------------------------
def train_ppo(env: NetworkEnv, results_dir: str, ckpt_path: str):
    """
    Huấn luyện PPO và lưu checkpoint cuối.
    Trả về đường dẫn checkpoint.
    """
    policy = FullPolicy(
        num_RUs=num_RUs,
        num_DUs=num_DUs,
        num_CUs=num_CUs,
        max_RBs_per_UE=max_RBs_per_UE,
        num_power_levels=num_power_levels,
    )

    agent = PPOAgent(
        policy=policy,
        total_nodes=total_nodes,
        num_RUs=num_RUs,
        num_DUs=num_DUs,
        num_CUs=num_CUs,
        k_DU=k_DU,
        k_CU=k_CU,
        P_ib_sk_val=P_ib_sk_val,
        max_RBs_per_UE=max_RBs_per_UE,
    )

    print("🚀 Bắt đầu huấn luyện PPO agent ...")
    agent_trained = train_agent(env, agent, results_dir)

    save_checkpoint(agent_trained, ckpt_path)
    print(f"💾 Đã lưu checkpoint tại: {ckpt_path}")
    return ckpt_path


# =====================================================
# 2) EVALUATION FUNCTION
# =====================================================
def evaluate_ppo(env, ckpt_path: str, results_dir: str, num_runs: int = 20) -> None:
    """
    Load checkpoint và đánh giá PPO agent nhiều lần để thống kê.
    Lưu các file thống kê vào {results_dir}/evaluation_agent_PPO/.
    """
    print("🔍 Đang đánh giá PPO agent (load từ checkpoint cuối)...")

    # Khởi tạo lại agent & policy đồng nhất với lúc train
    policy = FullPolicy(
        num_RUs=num_RUs,
        num_DUs=num_DUs,
        num_CUs=num_CUs,
        max_RBs_per_UE=max_RBs_per_UE,
        num_power_levels=num_power_levels,
    )
    agent = PPOAgent(
        policy=policy,
        total_nodes=total_nodes,
        num_RUs=num_RUs,
        num_DUs=num_DUs,
        num_CUs=num_CUs,
        k_DU=k_DU,
        k_CU=k_CU,
        P_ib_sk_val=P_ib_sk_val,
        max_RBs_per_UE=max_RBs_per_UE,
    )

    if not os.path.exists(ckpt_path) or not load_checkpoint(agent, ckpt_path, strict=True):
        print(f"⚠️  Không thể load checkpoint tại: {ckpt_path}. Dừng đánh giá.")
        return
    else:
        print(f"📦 Loaded checkpoint từ {ckpt_path} để đánh giá")

    # Chuẩn bị thư mục lưu
    eval_dir = Path(results_dir) / "evaluation_agent_PPO"
    eval_dir.mkdir(parents=True, exist_ok=True)

    reward_file     = eval_dir / "total_reward_PPO.txt"
    accept_file     = eval_dir / "total_accept_PPO.txt"
    throughput_file = eval_dir / "total_throughput_PPO.txt"
    latency_file    = eval_dir / "total_latency_PPO.txt"
    time_file       = eval_dir / "evaluation_time_PPO.txt"

    # Ghi file kết quả
    with open(reward_file, "w") as f_rew, \
         open(accept_file, "w") as f_acc, \
         open(throughput_file, "w") as f_thr, \
         open(latency_file, "w") as f_lat, \
         open(time_file, "w") as f_time:

        total_eval_time = 0.0

        # Chạy nhiều lần evaluation
        for run in range(1, max(1, num_runs) + 1):
            start_time = time.time()

            total_rew, total_acc, total_thr, total_lat, _ = evaluate_agent(
                env, agent, render=False
            )

            elapsed = time.time() - start_time
            total_eval_time += elapsed

            f_rew.write(f"{total_rew:.6f}\n")
            f_acc.write(f"{total_acc}\n")
            f_thr.write(f"{total_thr:.6f}\n")
            f_lat.write(f"{total_lat:.6f}\n")
            f_time.write(f"{elapsed:.3f}\n")

            for f in (f_rew, f_acc, f_thr, f_lat, f_time):
                f.flush()

            print(f"✅ Evaluation {run}/{num_runs} hoàn tất ({elapsed:.2f}s)")

        avg_time = total_eval_time / max(1, num_runs)
        print(f"⏱️ Tổng thời gian {num_runs} lần: {total_eval_time:.2f}s (TB {avg_time:.2f}s/lần)")
        print(f"📄 Kết quả lưu tại: {eval_dir}")


# =========================================================
# 3) BUILD INPUT MILP (trả về tuple biến)
# =========================================================
def _build_milp_inputs_from_env(env):
    """
    Trích xuất toàn bộ dữ liệu cần thiết từ NetworkEnv để truyền trực tiếp
    vào hàm MILP.
    """
    # --------------------- Kích thước mạng ---------------------
    num_UEs = env.num_UEs
    num_RUs = env.num_RUs
    num_DUs = env.num_DUs
    num_CUs = env.num_CUs
    num_RBs = env.num_RBs

    # --------------------- Slice info ---------------------
    slice_names = list(map(str, env.SLICE_PRESET.keys()))
    num_slices = len(slice_names)
    name_to_sid = {name: s for s, name in enumerate(slice_names)}

    # --------------------- Mapping UE → Slice (hằng) ---------------------
    slice_mapping = np.zeros((num_slices, num_UEs), dtype=int)
    for k in range(num_UEs):
        sname = str(env.UE_slice_name[k])
        slice_mapping[name_to_sid[sname], k] = 1

    # --------------------- QoS yêu cầu ---------------------
    R_min = np.array(
        [float(env.SLICE_PRESET[str(env.UE_slice_name[k])]["R_min"]) for k in range(num_UEs)],
        dtype=float
    )  # (K,)
    delay_max = np.array(
        [float(env.SLICE_PRESET[name]["delay"]) for name in slice_names],
        dtype=float
    )  # (S,)
    eta_list = np.array(
        [float(env.SLICE_PRESET[str(env.UE_slice_name[k])].get("eta_slice", 0.0)) for k in range(num_UEs)],
        dtype=float
    )

    # --------------------- DU / CU compute demand ---------------------
    D_j = env.k_DU * R_min * (1.0 + eta_list)  # (K,)
    D_m = env.k_CU * R_min * (1.0 + eta_list)  # (K,)

    # --------------------- Node capacity ---------------------
    A_j = np.array(env.DU_cap, dtype=float)  # (J,)
    A_m = np.array(env.CU_cap, dtype=float)  # (M,)

    # --------------------- Gain tensor ---------------------
    g_ik = np.array(env.gain, dtype=float)  # (R,K)
    gain = np.repeat(g_ik[:, np.newaxis, np.newaxis, :], repeats=num_RBs, axis=1)  # (R,B,1,K)
    gain = np.repeat(gain, repeats=num_slices, axis=2)                             # (R,B,S,K)

    # --------------------- Tham số vật lý ---------------------
    max_tx_power_watts = np.array(env.RU_power_cap, dtype=float)  # (R,)
    bandwidth_per_RB = float(env.bandwidth_per_RB)

    # --------------------- Liên kết topo (nhị phân 0/1) ---------------------
    l_ru_du = (np.array(env.l_ru_du, dtype=float) > 0).astype(int)  # (R,J)
    l_du_cu = (np.array(env.l_du_cu, dtype=float) > 0).astype(int)  # (J,M)

    # --------------------- Latency tensors ---------------------
    propagation_delay   = np.array(env.propagation_delay, dtype=float)      # (R,K)
    transmission_delay  = np.array(env.transmission_delay, dtype=float)     # (K,)
    processing_delay_DU = np.array(env.processing_delay_DU, dtype=float)    # (J,S,K)
    processing_delay_CU = np.array(env.processing_delay_CU, dtype=float)    # (M,S,K)
    queuing_delay_DU    = np.array(env.queuing_delay_DU, dtype=float)       # (J,S,K)
    queuing_delay_CU    = np.array(env.queuing_delay_CU, dtype=float)       # (M,S,K)

    # --------------------- Slice weights (đọc từ env) ---------------------
    slice_weight_accept = np.array(env.slice_weight_accept, dtype=float)  # (S,)
    slice_weight_throughput     = np.array(env.slice_weight_throughput, dtype=float)      # (S,)
    slice_weight_latency     = np.array(env.slice_weight_latency, dtype=float)      # (S,)

    # --------------------- Trọng số objective toàn cục ---------------------
    w_acc = float(env.w_acc)
    w_thr = float(env.w_thr)
    w_lat = float(env.w_lat)

    # --------------------- Trả về toàn bộ biến ---------------------
    return num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, bandwidth_per_RB, gain, slice_mapping, max_tx_power_watts, delay_max, propagation_delay, transmission_delay, processing_delay_DU, processing_delay_CU, queuing_delay_DU, queuing_delay_CU, w_acc, w_thr, w_lat, slice_weight_accept, slice_weight_throughput, slice_weight_latency


def save_result_MILP(out_dir, objective, accept, throughput, latency, power, runtime, tag=""):
    """Ghi kết quả ra các file .txt theo format thống nhất."""
    objective_file  = os.path.join(out_dir, "total_objective_MILP.txt")
    accept_file     = os.path.join(out_dir, "total_accept_MILP.txt")
    throughput_file = os.path.join(out_dir, "total_throughput_MILP.txt")
    latency_file    = os.path.join(out_dir, "total_latency_MILP.txt")
    power_file      = os.path.join(out_dir, "total_power_MILP.txt")
    time_file       = os.path.join(out_dir, "evaluation_time_MILP.txt")

    with open(objective_file, "w") as f_obj, \
         open(accept_file, "w") as f_acc, \
         open(throughput_file, "w") as f_thr, \
         open(latency_file, "w") as f_lat, \
         open(power_file, "w") as f_pow, \
         open(time_file, "w") as f_time:

        f_obj.write(f"{objective:.6f}\n")
        f_acc.write(f"{int(accept)}\n")
        f_thr.write(f"{throughput:.6f}\n")
        f_lat.write(f"{latency:.6e}\n")
        f_pow.write(f"{power:.6f}\n")
        f_time.write(f"{runtime:.3f}\n")

# =========================================================
# 4) GIẢI MILP
# =========================================================
def solve_milp(env, results_dir):
    num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, bandwidth_per_RB, gain, slice_mapping, max_tx_power_watts, delay_max, propagation_delay, transmission_delay, processing_delay_DU, processing_delay_CU, queuing_delay_DU, queuing_delay_CU,w_acc, w_thr, w_lat, slice_weight_accept, slice_thr_alpha, slice_lat_alpha = _build_milp_inputs_from_env(env)
    # ========== 2. Tạo thư mục MILP_results ==========
    milp_dir   = os.path.join(results_dir, "MILP_results")
    long_dir   = os.path.join(milp_dir, "long")
    short_dir  = os.path.join(milp_dir, "short")
    Doraemon_dir = os.path.join(milp_dir, "Doraemon")
    os.makedirs(long_dir,  exist_ok=True)
    os.makedirs(short_dir, exist_ok=True)
    os.makedirs(Doraemon_dir,  exist_ok=True)

    print("\n  Đang giải MILP Long_Doraemon ...")
    long_time_start = time.time()
    long_pi_sk, long_z_ib_sk, long_P_ib_sk, long_phi_i_sk, long_phi_j_sk, long_phi_m_sk, long_objective, long_total_pi_sk, long_total_R_sk, long_UE_delay, long_total_delay, long_total_z_ib_sk, long_total_P_ib_sk = solving_MILP_2.Long_Doraemon(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, bandwidth_per_RB, gain, slice_mapping, max_tx_power_watts, delay_max, propagation_delay, transmission_delay, processing_delay_DU, processing_delay_CU, queuing_delay_DU, queuing_delay_CU, w_acc, w_thr, w_lat, slice_weight_accept, slice_thr_alpha, slice_lat_alpha)

    if long_objective is None:
        print("❌ MILP Long Doraemon không khả thi.")
        return None
    long_time_end = time.time()
    long_time = long_time_end - long_time_start
    print(f"✅ Long_Doraemon hoàn tất trong {long_time:.2f}s")
    # Lưu file LONG
    save_result_MILP(
        out_dir=long_dir,
        objective=float(long_objective),
        accept=float(long_total_pi_sk),
        throughput=float(long_total_R_sk),
        latency=float(long_total_delay),
        power=float(long_total_P_ib_sk),
        runtime=float(long_time),
        tag="Long_"
    )
    

    # Chuyển nghiệm Long sang numpy 0/1
    arr_long_pi_sk    = other_function.extract_optimization_results(long_pi_sk)
    arr_long_z_ib_sk  = other_function.extract_optimization_results(long_z_ib_sk)
    arr_long_phi_i_sk = other_function.extract_optimization_results(long_phi_i_sk)
    arr_long_phi_j_sk = other_function.extract_optimization_results(long_phi_j_sk)
    arr_long_phi_m_sk = other_function.extract_optimization_results(long_phi_m_sk)
    
    print("  Đang giải MILP Short_Doraemon ...")
    short_time_start = time.time()
    short_pi_sk, short_z_ib_sk, short_P_ib_sk, short_phi_i_sk, short_phi_j_sk, short_phi_m_sk, short_objective, short_total_pi_sk, short_total_R_sk, short_UE_delay, short_total_delay, short_total_z_ib_sk, short_total_P_ib_sk = solving_MILP_2.Short_Doraemon(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs,D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, bandwidth_per_RB, gain, slice_mapping, max_tx_power_watts, delay_max, propagation_delay, transmission_delay, processing_delay_DU, processing_delay_CU, queuing_delay_DU, queuing_delay_CU, w_acc, w_thr, w_lat, slice_weight_accept, slice_thr_alpha, slice_lat_alpha, arr_long_pi_sk, arr_long_z_ib_sk, arr_long_phi_i_sk, arr_long_phi_j_sk, arr_long_phi_m_sk, long_UE_delay)
    short_time_end = time.time()
    short_time = short_time_end - short_time_start
    Doraemon_time = short_time_end - long_time_start

    print(f"✅ Hoàn tất Long + Short Doraemon trong {Doraemon_time:.2f} giây")
    # Lưu file SHORT
    save_result_MILP(
        out_dir=short_dir,
        objective=float(short_objective),
        accept=float(short_total_pi_sk),
        throughput=float(short_total_R_sk),
        latency=float(short_total_delay),
        power=float(short_total_P_ib_sk),
        runtime=float(short_time),
        tag="Short_"
    )

    save_result_MILP(
        out_dir=Doraemon_dir,
        objective=float(short_objective),
        accept=float(short_total_pi_sk),
        throughput=float(short_total_R_sk),
        latency=float(short_total_delay),
        power=float(short_total_P_ib_sk),
        runtime=float(Doraemon_time),
        tag="Doraemon_"
    )


    # ========== 6. Log tổng hợp ==========
    print("\n📊 Kết quả:")
    print(f"  - Long Power: {long_total_P_ib_sk:.6f}")
    print(f"  - Short Power: {short_total_P_ib_sk:.6f}")
    print(f"  - Long Throughtput: {long_total_R_sk:.6f}")
    print(f"  - Short Throughtput: {short_total_R_sk:.6f}")
    print(f"  - Short Accept: {short_total_pi_sk:.6f}")
    print(f"  - total z_ib_sk: {long_total_z_ib_sk:.6f}")
    print(f"  - Short Latency: {short_total_delay:.6f}")
    print(f"  - Long Objective:  {long_objective:.6f}")
    print(f"  - Short Objective: {short_objective:.6f}")
    print(f"  - Thời gian Long:  {long_time:.2f}s")
    print(f"  - Thời gian Short: {short_time:.2f}s")
    print(f"📂 LONG  → {long_dir}")
    print(f"📂 SHORT → {short_dir}")
    print(f"📂 Doraemon → {Doraemon_dir}")



# -----------------------------------------
# 5) MAIN: tách train / eval rõ ràng
# -----------------------------------------


def main_1():
    

    # dir
    os.makedirs(results_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_root, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    num_UEs = 20      
    num_RBs = 25 # Số RB (Resource Block)

    print(f"\n==============================")
    print(f"🚀 Chạy thử với {num_UEs} UEs")
    print(f"==============================")

    # Tạo thư mục riêng cho từng cấu hình UE
        

        # --- Khởi tạo môi trường ---
    env = NetworkEnv(
        total_nodes=total_nodes,
        num_RUs=num_RUs,
        num_DUs=num_DUs,
        num_CUs=num_CUs,
        num_RBs=num_RBs,
        num_UEs=num_UEs,
        SLICE_PRESET=SLICE_PRESET,
        P_i_random_list=P_i_random_list,
        A_j_random_list=A_j_random_list,
        A_m_random_list=A_m_random_list,
        bw_ru_du_random_list=bw_ru_du_random_list,
        bw_du_cu_random_list=bw_du_cu_random_list,
        bandwidth_per_RB=bandwidth_per_RB,
        max_RBs_per_UE=max_RBs_per_UE,
        P_ib_sk_val=P_ib_sk_val,
        k_DU=k_DU,
        k_CU=k_CU,
    )

    # --- (1) Train PPO ---
    #ckpt_path = os.path.join(results_dir, "checkpoint_PPO.pt")

    # --- MILP ---
    #solve_milp(env, results_dir)
    
    # --- (3) Run baselines ---
    run_all_baselines(env, results_dir, episodes=20)

    # --- (2) Evaluate PPO ---
    #train_ppo(env, results_dir, ckpt_path)
    #evaluate_ppo(env, ckpt_path, results_dir, num_runs=20)

    

def main_2():
    os.makedirs(results_root, exist_ok=True)

    num_UEs_list = [30, 40, 50]  
    #num_UEs_list = [50]  
    num_RBs = 135 # Số RB (Resource Block)    

    for num_UEs in num_UEs_list:
        print(f"\n==============================")
        print(f"🚀 Chạy thử với {num_UEs} UEs")
        print(f"==============================")

        # Tạo thư mục riêng cho từng cấu hình UE
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(results_root, f"run_{num_UEs}UE_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        # --- Khởi tạo môi trường ---
        env = NetworkEnv(
            total_nodes=total_nodes,
            num_RUs=num_RUs,
            num_DUs=num_DUs,
            num_CUs=num_CUs,
            num_RBs=num_RBs,
            num_UEs=num_UEs,
            SLICE_PRESET=SLICE_PRESET,
            P_i_random_list=P_i_random_list,
            A_j_random_list=A_j_random_list,
            A_m_random_list=A_m_random_list,
            bw_ru_du_random_list=bw_ru_du_random_list,
            bw_du_cu_random_list=bw_du_cu_random_list,
            bandwidth_per_RB=bandwidth_per_RB,
            max_RBs_per_UE=max_RBs_per_UE,
            P_ib_sk_val=P_ib_sk_val,
            k_DU=k_DU,
            k_CU=k_CU,
        )

        # --- (1) Train PPO ---
        ckpt_path = os.path.join(results_dir, "checkpoint_PPO.pt")
        train_ppo(env, results_dir, ckpt_path)

        # --- (2) Evaluate PPO ---
        evaluate_ppo(env, ckpt_path, results_dir, num_runs=20)

        # --- (3) Run baselines ---
        run_all_baselines(env, results_dir, episodes=20)

def main_3():
    # seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # dir
    
    os.makedirs(results_root, exist_ok=True)

    num_UEs = 20
    num_RBs = 25 # Số RB (Resource Block)    

    print(f"\n==============================")
    print(f"🚀 Chạy thử với {num_UEs} UEs")
    print(f"==============================")

    # Tạo thư mục riêng cho từng cấu hình UE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_root, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # --- Khởi tạo môi trường ---
    env = NetworkEnv(
        total_nodes=total_nodes,
        num_RUs=num_RUs,
        num_DUs=num_DUs,
        num_CUs=num_CUs,
        num_RBs=num_RBs,
        num_UEs=num_UEs,
        SLICE_PRESET=SLICE_PRESET,
        P_i_random_list=P_i_random_list,
        A_j_random_list=A_j_random_list,
        A_m_random_list=A_m_random_list,
        bw_ru_du_random_list=bw_ru_du_random_list,
        bw_du_cu_random_list=bw_du_cu_random_list,
        bandwidth_per_RB=bandwidth_per_RB,
        max_RBs_per_UE=max_RBs_per_UE,
        P_ib_sk_val=P_ib_sk_val,
        k_DU=k_DU,
        k_CU=k_CU,
    )


    # --- (3) Run baselines ---
    run_all_baselines(env, results_dir, episodes=20)


results_root = "./results_final_6"

if __name__ == "__main__":
    # seed
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    #main_1()
    main_2()
    #main_3()