import random
import numpy as np

from config import *
from Env.network_env import NetworkEnv

def run_admission(env, new_ues):
    accepted = 0
    fail_reasons = {}

    for ue_id in new_ues:
        placed = False

        # 1. RU theo gain (tốt → kém)
        ru_order = np.argsort(env.gain[:, ue_id])[::-1]

        for ru in ru_order:

            # 2. chỉ lấy DU nối được với RU
            valid_dus = np.where(env.l_ru_du[ru] == 1)[0]
            # print(f"RU {ru} -> DU {valid_dus}")
            
            if len(valid_dus) == 0:
                continue

            for du in valid_dus:

                # 3. chỉ lấy CU nối được với DU
                valid_cus = np.where(env.l_du_cu[du] == 1)[0]
                
                # print(f"DU {du} -> CU {valid_cus}")
                if len(valid_cus) == 0:
                    continue

                for cu in valid_cus:

                    # 4. thử resource (ít → nhiều)
                    for prb in range(1, env.max_RBs_per_UE + 1):

                        for ptx in env.P_ib_sk_val:

                            ok, msg = env.check_feasible(ue_id, 
                                                         int(ru), 
                                                         int(du), 
                                                         int(cu), 
                                                         prb, 
                                                         ptx
                            )

                            if ok:
                                thr, cpu_du, cpu_cu, delay_total, delay_parts = msg

                                env.update_network(
                                    ue_id,
                                    int(ru),
                                    int(du),
                                    int(cu),
                                    prb,
                                    ptx,
                                    thr,
                                    cpu_du,
                                    cpu_cu,
                                    delay_total,
                                    delay_parts,
                                )

                                accepted += 1
                                placed = True
                                break

                        if placed:
                            break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break

        if not placed:
            fail_reasons["not_feasible"] = fail_reasons.get("not_feasible", 0) + 1

    return accepted, fail_reasons
            
def build_env(num_ues=50, num_rbs=135, dynamic_mode=True):
    env = NetworkEnv(
        total_nodes=total_nodes,
        num_RUs=num_RUs,
        num_DUs=num_DUs,
        num_CUs=num_CUs,
        num_RBs=num_rbs,
        num_UEs=num_ues,
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
        dynamic_mode=dynamic_mode,
        min_ues=45,
        max_ues=55,
        mobility_step=10.0,
    )
    return env


def print_stats(env, state, info):
    served_active = sum(
        1 for ue in env.UE_requests.values()
        if ue["status"]["active"] == 1 and ue["status"]["served"] == 1
    )

    print(f"time_step           : {info['time_step']}")
    print(f"active_ues          : {state['num_active_ues']}")
    print(f"served_active       : {served_active}")
    print(f"new_ues             : {len(info['new_ue_candidates'])}")
    print(f"stable_ues          : {len(info['stable_ues'])}")
    print(f"ho_candidates       : {len(info['ho_candidates'])}")
    print(f"RB_remaining        : {state['RB_remaining']}")
    print(f"PRB_remaing_per_RU  : {state['PRB_remaining_per_RU']}")
    print("-" * 50)

def print_ue_path(env):
    print("\n===== UE PATHS =====")
    
    for ue_id, ue in env.UE_requests.items():
        if int(ue["status"]["active"]) != 1 or int(ue["status"]["served"]) != 1:
            continue

        ru = ue["alloc"]["RU"]
        du = ue["alloc"]["DU"]
        cu = ue["alloc"]["CU"]
        prb = ue["alloc"]["num_RB_alloc"]
        ptx = ue["alloc"]["power_alloc"]

        print(
            f"UE {ue_id} --> RU {ru} --> DU {du} --> CU {cu} | "
            f"PRB={prb}, PTX={ptx:.6f}"
        )
    print("-" * 60)

def print_traffic(env):
    print("\n===== UE TRAFFIC =====")
    for ue_id, ue in env.UE_requests.items():
        if int(ue["status"]["active"]) != 1:
            continue
        print(
            f"UE {ue_id:>2} | slice={ue['slice']:<5} | "
            f"queue={ue['traffic']['queue_bits']:.1f} bits | "
            f"lambda={ue['traffic']['lambda_pps']:.2f} pps | "
            f"rate={ue['traffic']['arrival_rate_bps']:.2f} bps"
        )
    print("-" * 60)
    
def main():
    seed = 2
    random.seed(seed)
    np.random.seed(seed)

    env = build_env(num_ues=50, num_rbs=135, dynamic_mode=True)

    print("=== RESET ===")
    state = env.reset_env()
    print("num_active_ues:", state["num_active_ues"])
    print("RB_remaining :", state["RB_remaining"])
    
    print("PRB_remaining_per_RU   :", state["PRB_remaining_per_RU"])
    print("-" * 50)

    for t in range(10):
        target = int(np.random.randint(45, 56))
        print(f"\n=== STEP {t+1} | target={target} ===")
        state, info = env.advance_time(target_active_ues=target)
        accepted, fail = run_admission(env, info["new_ue_candidates"])

        state = env.get_state()
        
        

        print_stats(env, state, info)
        # print_ue_path(env)
        # print_traffic(env)
        # print(f"accepted this step: {accepted}")
        # print(f"fail reasons: {fail}")
        

    print("Smoke test done.")


if __name__ == "__main__":
    main()