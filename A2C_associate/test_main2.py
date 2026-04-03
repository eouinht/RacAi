

import os
import random
import numpy as np
from datetime import datetime

from config import *
from Env.network_env import NetworkEnv
from baseline import run_all_baselines


def set_seed(seed=2):
    random.seed(seed)
    np.random.seed(seed)


def create_env(num_UEs=30, num_RBs=135):
    return NetworkEnv(
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


def test_compute_prb_requirement(env):
    UE_idx = 0
    RU_choice = 0
    DU_choice = 0
    CU_choice = 0
    power = P_ib_sk_val[0]

    min_prb, status, details = env.compute_prb_requirement(
        UE_idx=UE_idx,
        RU_choice=RU_choice,
        DU_choice=DU_choice,
        CU_choice=CU_choice,
        power_level_alloc=power,
        step_duration_s=0.1
    )

    assert status in ["feasible", "no_feasible_prb_found"], f"Invalid status {status}"
    if status == "feasible":
        assert min_prb >= 1 and min_prb <= env.max_RBs_per_UE
    print("test_compute_prb_requirement PASSED", status, min_prb)
    return min_prb, status, details


def test_check_feasible(env, num_RB_alloc, power):
    f, msg = env.check_feasible(
        UE_idx=0,
        RU_choice=0,
        DU_choice=0,
        CU_choice=0,
        num_RB_alloc=num_RB_alloc,
        power_level_alloc=power,
        step_duration_s=0.1
    )

    if f:
        print("test_check_feasible PASSED: feasible", msg)
    else:
        print("test_check_feasible PASSED: infeasible (expected in some cases)", msg)
    return f, msg


def test_get_filter_ues(env):
    stable, ho, new_u = env.get_filter_ues(sinr_margin_db=1.0)
    assert isinstance(stable, list) and isinstance(ho, list) and isinstance(new_u, list)
    print(f"test_get_filter_ues PASSED: stable={len(stable)} ho={len(ho)} new={len(new_u)}")
    return stable, ho, new_u


def test_allocate_from_filter(env, max_allocs=5):
    stable, ho, new_u = env.get_filter_ues(sinr_margin_db=1.0)
    selection = ho + new_u
    if not selection:
        print("test_allocate_from_filter: no HO/new candidates to allocate")
        return []

    allocated = []
    for ue_id in selection[:max_allocs]:
        ue = env.UE_requests[ue_id]

        # choose RU with highest gain for this UE and any available DU/CU path
        best_ru = int(np.argmax(env.gain[:, ue_id]))
        dus = [d for d in range(env.num_DUs) if env.l_ru_du[best_ru, d] == 1 and env.DU_remaining[d] > 0]
        if not dus:
            continue
        du = dus[0]

        cus = [c for c in range(env.num_CUs) if env.l_du_cu[du, c] == 1 and env.CU_remaining[c] > 0]
        if not cus:
            continue
        cu = cus[0]

        power = float(env.P_ib_sk_val[0])
        min_prb, status, details = env.compute_prb_requirement(
            UE_idx=ue_id,
            RU_choice=best_ru,
            DU_choice=du,
            CU_choice=cu,
            power_level_alloc=power,
            step_duration_s=0.1
        )

        if status == "feasible":
            f, msg = env.check_feasible(
                UE_idx=ue_id,
                RU_choice=best_ru,
                DU_choice=du,
                CU_choice=cu,
                num_RB_alloc=min_prb,
                power_level_alloc=power,
                step_duration_s=0.1
            )
            if f:
                data_rate, cpu_DU_req, cpu_CU_req, delay_total, delay_parts = msg
                env.update_network(ue_id, best_ru, du, cu, min_prb, power, data_rate, cpu_DU_req, cpu_CU_req, delay_total, delay_parts)
                allocated.append(ue_id)
                print(f"Allocated UE {ue_id} as {'HO' if ue_id in ho else 'NEW'} with {min_prb} PRBs")

    return allocated


def test_advance_time_and_departures(env):
    env.advance_time(target_active_ues=env.max_active_ues)
    departed = env.sample_departures()
    if departed:
        print(f"test_advance_time_and_departures PASSED: departed {len(departed)} UEs")
    else:
        print("test_advance_time_and_departures PASSED: no departures this step")


def test_remove_and_add_ues(env):
    if len(env.UE_requests) == 0:
        raise AssertionError("No UEs in env")

    first_id = 0
    env.release_ue(first_id, reason="test_departed")
    assert env.UE_requests[first_id]["status"]["active"] == 0
    print("test_remove_and_add_ues PASSED: removed UE", first_id)

    added = env.add_new_ues(target_active_use=env.min_active_ues + 1)
    print("test_remove_and_add_ues PASSED: added UEs", added)


def test_full_pipeline():
    set_seed(2)
    env = create_env(num_UEs=50, num_RBs=135)

    min_prb, status, details = test_compute_prb_requirement(env)

    if status == "feasible":
        power = P_ib_sk_val[0]
        feasible, msg = test_check_feasible(env, num_RB_alloc=min_prb, power=power)
        if feasible:
            data_rate, cpu_DU_req, cpu_CU_req, delay_total, delay_parts = msg
            env.update_network(0, 0, 0, 0, min_prb, power, data_rate, cpu_DU_req, cpu_CU_req, delay_total, delay_parts)
            print("Committed allocation for UE 0")

    stable, ho, new_u = test_get_filter_ues(env)

    test_advance_time_and_departures(env)
    test_remove_and_add_ues(env)

    print("Running quick baseline test (3 episodes)...")
    run_all_baselines(env, "./results_test_proceed", episodes=3)
    print("test_full_pipeline completed successfully")

def test_dynamic_steps(env, steps=10):
    for t in range(steps):
        target = int(np.random.randint(45, 56))
        print(f"\n=== STEP {t+1} | target_active_ues={target} ===")
        state, info = env.advance_time(target_active_ues=target)

        # Optional: after advance, run filtering and report
        stable, ho, new_u = env.get_filter_ues(sinr_margin_db=1.0)
        print(
            f"advance_time info: departed={info['departed_ues'][:5]} (n={len(info['departed_ues'])}), "
            f"new={info['new_ue_ids'][:5]} (n={len(info['new_ue_ids'])}), "
            f"stable={len(stable)}, ho={len(ho)}, new={len(new_u)}"
        )

        # allocate candidates from new/ho with current filter state
        allocated = test_allocate_from_filter(env, max_allocs=5)
        print(f"Allocated UEs this step: {allocated}")

        # Re-run filtering after allocation to update groups
        stable, ho, new_u = env.get_filter_ues(sinr_margin_db=1.0)
        print(f"post-allocation stable={len(stable)}, ho={len(ho)}, new={len(new_u)}")

        # refresh state snapshot
        state = env.get_state()
        print(
            f"state after step: RB_rem={state['RB_remaining']}, "
            f"active_ues={state['num_active_ues']}, "
            f"RU_power={state['RU_power_remaining']}"
        )
        
if __name__ == "__main__":
    print("Starting rewritten full env test script\n")
    env = create_env(num_UEs=50, num_RBs=135)
    test_full_pipeline()
    print("\nNow running dynamic state updates to verify next-state behavior:")
    test_dynamic_steps(env, steps=10)
    print("All done")
