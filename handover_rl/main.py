from pathlib import Path

from env import TraceDrivenHandoverEnv
from models import UEAction
from parser import NS3TraceParser


HO_MARGIN_DB = 0.0


def choose_best_ru_by_sinr(ue_state: dict, rus_state: dict, ho_margin_db: float = HO_MARGIN_DB) -> int:
    serving_ru = ue_state["serving_ru"]
    air_metrics = ue_state.get("air_metrics", {})

    current_sinr = None
    if serving_ru in air_metrics:
        current_sinr = air_metrics[serving_ru].get("sinr_db")

    best_ru = serving_ru
    best_sinr = current_sinr if current_sinr is not None else float("-inf")

    for ru_id, metric in air_metrics.items():
        sinr = metric.get("sinr_db")
        if sinr is None:
            continue
        
        cell_type = rus_state.get(ru_id, {}).get("cell_type", "unknown")
        if cell_type == "small":
            sinr += 1.0
        elif cell_type == "macro":
            sinr += 0.0
            
        if sinr > best_sinr:
            best_sinr = sinr
            best_ru = ru_id

    if best_ru == serving_ru:
        return serving_ru

    if current_sinr is None:
        return best_ru

    if best_sinr >= current_sinr + ho_margin_db:
        return best_ru

    return serving_ru


def print_state_summary(state: dict) -> None:
    print(f"State t={state['t']}")
    for ue_id, ue_state in state["ues"].items():
        print(
            f"  UE {ue_id}: "
            f"serving_ru={ue_state['serving_ru']}, "
            f"sinr_db={ue_state['sinr_db']}, "
            f"rsrp_dbm={ue_state['rsrp_dbm']}, "
            f"tput_mbps={ue_state['tput_mbps']}, "
            f"queue_bytes={ue_state['queue_bytes']}, "
            f"latency_ms={ue_state['latency_ms']}"
        )


def print_action_summary(state: dict, actions: dict) -> None:
    print("Actions:")
    for ue_id, action in actions.items():
        serving_ru = state["ues"][ue_id]["serving_ru"]
        air_metrics = state["ues"][ue_id].get("air_metrics", {})
        current_sinr = air_metrics.get(serving_ru, {}).get("sinr_db")
        target_sinr = air_metrics.get(action.target_ru, {}).get("sinr_db")

        print(
            f"  UE {ue_id}: "
            f"serving_ru={serving_ru} (sinr={current_sinr}) "
            f"-> target_ru={action.target_ru} (sinr={target_sinr})"
        )


def print_step_info(step_info: dict) -> None:
    reward_info = step_info.get("reward_info", {})
    print("Step info:")
    print(f"  handover_types={step_info.get('handover_types', {})}")
    print(
        "  reward_info="
        f"total_tput_mbps={reward_info.get('total_tput_mbps')}, "
        f"total_queue_bytes={reward_info.get('total_queue_bytes')}, "
        f"avg_queue_mb={reward_info.get('avg_queue_mb')}, "
        f"avg_delay_penalty_ms={reward_info.get('avg_delay_penalty_ms')}, "
        f"total_handover_cost={reward_info.get('total_handover_cost')}, "
        f"handover_count={reward_info.get('handover_count')}"
    )


def main() -> None:
    parser = NS3TraceParser()
    trace_path = Path(__file__).with_name("nr_stream.jsonl")
    trace = parser.parse_file(trace_path)

    env = TraceDrivenHandoverEnv(trace)

    state, info = env.reset()
    print("Reset info:", info)
    print_state_summary(state)
    print("-" * 80)

    done = False
    step_count = 0

    while not done:
        actions = {}

        for ue_id, ue_state in state["ues"].items():
            target_ru = choose_best_ru_by_sinr(ue_state, state["rus"])

            actions[ue_id] = UEAction(
                target_ru=target_ru,
                prb_alloc=0.0,
                ptx_alloc=0.0,
                du_alloc=0.0,
                cu_alloc=0.0,
            )

        print(f"Decision at t={state['t']}")
        print_action_summary(state, actions)

        next_state, reward, terminated, truncated, step_info = env.step(actions)

        print(f"Result after step to t={next_state['t']}: reward={reward:.3f}")
        print_step_info(step_info)
        print_state_summary(next_state)
        print("-" * 80)

        state = next_state
        step_count += 1
        done = terminated or truncated

    print("Finished after", step_count, "steps")


if __name__ == "__main__":
    main()