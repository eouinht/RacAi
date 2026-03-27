# ===========================
# main_a2c.py  (RUN ONLY A2C)
# ===========================
import os
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path

from config import *                 # dùng đúng các biến config cũ của bạn
from Env.network_env import NetworkEnv

# >>> A2C baseline (MLP-only, no GraphSAGE)
from model.a2c_mlp_agent import (
    A2CAgent, A2CPolicyMLP,
    train_agent_a2c, evaluate_agent_a2c,
    save_checkpoint_a2c, load_checkpoint_a2c
)

# ===========================
# Helpers
# ===========================
def set_seed(seed: int = 2):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_env(num_UEs: int, num_RBs: int) -> NetworkEnv:
    """
    Build NetworkEnv giống hệt style main cũ (chỉ đổi num_UEs/num_RBs theo kịch bản).
    Các biến còn lại lấy từ config.py của bạn.
    """
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
    return env


# ===========================
# Train / Eval (A2C)
# ===========================
def train_a2c(env: NetworkEnv, results_dir: str, ckpt_path: str) -> str:
    """
    Train A2C_MLP và lưu checkpoint.
    """
    policy = A2CPolicyMLP(
        num_RUs=num_RUs,
        num_DUs=num_DUs,
        num_CUs=num_CUs,
        max_RBs_per_UE=max_RBs_per_UE,
        num_power_levels=num_power_levels,
    )

    agent = A2CAgent(
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

    print("🚀 Train A2C_MLP ...")
    agent_trained = train_agent_a2c(env, agent, results_dir)

    save_checkpoint_a2c(agent_trained, ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}")
    return ckpt_path


def evaluate_a2c(env: NetworkEnv, ckpt_path: str, results_dir: str, num_runs: int = 20) -> None:
    """
    Load checkpoint và chạy evaluate nhiều lần, lưu log vào results_dir/evaluation_agent_A2C/
    """
    print("🔍 Evaluate A2C_MLP ...")

    policy = A2CPolicyMLP(
        num_RUs=num_RUs,
        num_DUs=num_DUs,
        num_CUs=num_CUs,
        max_RBs_per_UE=max_RBs_per_UE,
        num_power_levels=num_power_levels,
    )
    agent = A2CAgent(
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

    if not os.path.exists(ckpt_path) or not load_checkpoint_a2c(agent, ckpt_path, strict=True):
        print(f"⚠️  Cannot load checkpoint: {ckpt_path}")
        return

    eval_dir = Path(results_dir) / "evaluation_agent_A2C"
    eval_dir.mkdir(parents=True, exist_ok=True)

    reward_file     = eval_dir / "total_reward_A2C.txt"
    accept_file     = eval_dir / "total_accept_A2C.txt"
    throughput_file = eval_dir / "total_throughput_A2C.txt"
    latency_file    = eval_dir / "total_latency_A2C.txt"
    time_file       = eval_dir / "evaluation_time_A2C.txt"

    total_eval_time = 0.0

    with open(reward_file, "w") as f_rew, \
         open(accept_file, "w") as f_acc, \
         open(throughput_file, "w") as f_thr, \
         open(latency_file, "w") as f_lat, \
         open(time_file, "w") as f_time:

        for run in range(1, max(1, num_runs) + 1):
            t0 = time.time()

            total_rew, total_acc, total_thr, total_lat, _ = evaluate_agent_a2c(
                env, agent, render=False
            )

            elapsed = time.time() - t0
            total_eval_time += elapsed

            f_rew.write(f"{total_rew:.6f}\n")
            f_acc.write(f"{total_acc}\n")
            f_thr.write(f"{total_thr:.6f}\n")
            f_lat.write(f"{total_lat:.6f}\n")
            f_time.write(f"{elapsed:.3f}\n")

            for f in (f_rew, f_acc, f_thr, f_lat, f_time):
                f.flush()

            print(f"✅ Eval {run}/{num_runs} done | time={elapsed:.2f}s | reward={total_rew:.3f} | acc={total_acc}")

    avg_time = total_eval_time / max(1, num_runs)
    print(f"📄 Saved eval logs to: {eval_dir}")
    print(f"⏱️ Total eval time: {total_eval_time:.2f}s | Avg/run: {avg_time:.2f}s")


# ===========================
# Scenarios (same as old)
# ===========================
def main_1():
    """
    Scenario like old main_1:
      - num_UEs = 20
      - num_RBs = 25
      - Train + Eval
    """
    set_seed(42)

    os.makedirs(results_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_root, f"run_A2C_20UE_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    num_UEs = 20
    num_RBs = 25

    print("\n==============================")
    print(f"🚀 A2C scenario: {num_UEs} UEs | {num_RBs} RBs")
    print("==============================")

    env = build_env(num_UEs=num_UEs, num_RBs=num_RBs)

    ckpt_path = os.path.join(results_dir, "checkpoint_A2C.pt")
    train_a2c(env, results_dir, ckpt_path)
    evaluate_a2c(env, ckpt_path, results_dir, num_runs=20)


def main_2():
    """
    Scenario like old main_2:
      - num_UEs_list = [30, 40, 50]
      - num_RBs = 135
      - Train + Eval for each num_UEs
    """
    set_seed(42)

    os.makedirs(results_root, exist_ok=True)
    num_UEs_list = [30, 40, 50]
    num_RBs = 135

    for num_UEs in num_UEs_list:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(results_root, f"run_A2C_{num_UEs}UE_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        print("\n==============================")
        print(f"🚀 A2C scenario: {num_UEs} UEs | {num_RBs} RBs")
        print("==============================")

        env = build_env(num_UEs=num_UEs, num_RBs=num_RBs)

        ckpt_path = os.path.join(results_dir, "checkpoint_A2C.pt")
        train_a2c(env, results_dir, ckpt_path)
        evaluate_a2c(env, ckpt_path, results_dir, num_runs=20)



# ===========================
# Entry
# ===========================
# Nếu code cũ của bạn set results_root ở cuối file, bạn có thể giữ giống:
results_root = "./results"
# Nếu config.py đã có results_root thì dòng dưới không cần.
try:
    results_root
except NameError:
    results_root = "./results"


if __name__ == "__main__":
    # chạy giống cách bạn hay chạy ở code cũ:
    main_1()
    main_2()
