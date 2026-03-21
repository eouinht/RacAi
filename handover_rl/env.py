from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from models import (
    CellAirMetric,
    EnvConfig,
    HandoverCosts,
    RewardWeights,
    TimeStep,
    Topology,
    TraceBundle,
    UEAction,
)
from reward_engine import RewardEngine
from state_builder import StateBuilder


class TraceDrivenHandoverEnv:
    """
    Trace-guided pseudo-dynamics env:
    - dùng trace làm bối cảnh mobility / arrivals / air metrics
    - action HO sẽ rewrite next snapshot trước khi tính reward
    """

    def __init__(
        self,
        trace_bundle: TraceBundle,
        env_config: Optional[EnvConfig] = None,
        reward_weights: Optional[RewardWeights] = None,
        ho_costs: Optional[HandoverCosts] = None,
    ) -> None:
        self.trace_bundle = trace_bundle
        self.env_config = env_config or EnvConfig()
        self.state_builder = StateBuilder()
        self.reward_engine = RewardEngine(
            weights=reward_weights or RewardWeights(),
            ho_costs=ho_costs or HandoverCosts(),
            delay_threshold_ms=self.env_config.delay_threshold_ms,
        )
        self._idx = 0

    @property
    def topology(self) -> Topology:
        return self.trace_bundle.topology

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.trace_bundle.steps:
            raise ValueError("Trace bundle has no time-step snapshots.")

        self._idx = 0
        snapshot = self.trace_bundle.steps[self._idx]
        state = self.state_builder.build(snapshot, self.topology)
        info = {
            "t": snapshot.t,
            "num_ues": len(snapshot.ue_metrics),
        }
        return state, info

    def step(
        self,
        actions: Dict[int, UEAction],
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._idx >= len(self.trace_bundle.steps) - 1:
            current = self.trace_bundle.steps[self._idx]
            state = self.state_builder.build(current, self.topology)
            return state, 0.0, True, False, {"reason": "end_of_trace"}

        prev_snapshot = self.trace_bundle.steps[self._idx]
        self._validate_actions(prev_snapshot, actions)

        self._idx += 1
        next_snapshot_trace = self.trace_bundle.steps[self._idx]

        sim_next_snapshot = self._apply_actions_to_next_snapshot(
            prev_snapshot=prev_snapshot,
            next_snapshot=next_snapshot_trace,
            actions=actions,
        )

        reward, reward_info = self.reward_engine.compute(
            prev_snapshot=prev_snapshot,
            next_snapshot=sim_next_snapshot,
            actions=actions,
            topology=self.topology,
        )

        state = self.state_builder.build(sim_next_snapshot, self.topology)
        terminated = self._idx >= len(self.trace_bundle.steps) - 1
        truncated = False

        info = {
            "t": sim_next_snapshot.t,
            "reward_info": reward_info,
            "handover_types": self._classify_actions(prev_snapshot, actions),
        }

        return state, reward, terminated, truncated, info

    def _validate_actions(
        self,
        snapshot: TimeStep,
        actions: Dict[int, UEAction],
    ) -> None:
        for ue_id in snapshot.ue_metrics:
            if ue_id not in actions:
                raise ValueError(f"Missing action for ue_id={ue_id}")

            action = actions[ue_id]
            if action.target_ru not in self.topology.rus:
                raise ValueError(
                    f"Invalid target_ru={action.target_ru} for ue_id={ue_id}"
                )

    def _classify_actions(
        self,
        snapshot: TimeStep,
        actions: Dict[int, UEAction],
    ) -> Dict[int, str]:
        result: Dict[int, str] = {}

        for ue_id, ue in snapshot.ue_metrics.items():
            action = actions.get(ue_id)
            if action is None:
                continue

            ho_type = self.topology.classify_handover(
                ue.serving_ru,
                action.target_ru,
            )
            result[ue_id] = ho_type.name

        return result

    def _apply_actions_to_next_snapshot(
        self,
        prev_snapshot: TimeStep,
        next_snapshot: TimeStep,
        actions: Dict[int, UEAction],
    ) -> TimeStep:
        sim_snapshot = deepcopy(next_snapshot)

        # 1) tính load mới theo target RU agent chọn
        ru_load = self._compute_ru_load_after_actions(prev_snapshot, actions)

        # 2) rewrite từng UE theo action
        for ue_id, sim_ue in sim_snapshot.ue_metrics.items():
            prev_ue = prev_snapshot.ue_metrics.get(ue_id)
            action = actions.get(ue_id)

            if prev_ue is None or action is None:
                continue

            target_ru = action.target_ru
            old_ru = prev_ue.serving_ru
            did_ho = target_ru != old_ru

            # serv_cell mới
            sim_ue.serving_ru = target_ru
            sim_ue.du_id = self.topology.get_du(target_ru)
            sim_ue.cu_id = self.topology.get_cu(target_ru)

           
            air_metric = sim_ue.air_metrics.get(target_ru)
            if air_metric is None:
                target_sinr = float(sim_ue.sinr_db or -5.0)
                target_rsrp = float(sim_ue.rsrp_dbm or -110.0)
                missing_air_penalty = True
            else:
                target_sinr = float(air_metric.sinr_db or -5.0)
                target_rsrp = float(air_metric.rsrp_dbm or -110.0)
                missing_air_penalty = False

            sim_ue.sinr_db = target_sinr
            sim_ue.rsrp_dbm = target_rsrp

            # ước lượng throughput theo SINR + load cell
            cell_load = max(1, ru_load.get(target_ru, 1))
            cell_type = self.topology.rus[target_ru].cell_type
            estimated_tput = self._estimate_throughput_mbps(
                sinr_db=target_sinr,
                cell_load=cell_load,
                cell_type=cell_type,
                did_ho=did_ho,
                missing_air_penalty=missing_air_penalty,
            )

            prev_queue = float(prev_ue.bsr_bytes or 0.0)
            arrival_bytes = float(sim_ue.payload_arrival_bytes or 0.0)

            estimated_queue = self._estimate_queue_bytes(
                prev_queue_bytes=prev_queue,
                arrival_bytes=arrival_bytes,
                throughput_mbps=estimated_tput,
            )

            estimated_latency = self._estimate_latency_ms(
                queue_bytes=estimated_queue,
                throughput_mbps=estimated_tput,
                did_ho=did_ho,
                missing_air_penalty=missing_air_penalty,
            )

            sim_ue.tput_mbps = estimated_tput
            sim_ue.bsr_bytes = estimated_queue
            sim_ue.latency_ms = estimated_latency

        return sim_snapshot

    def _compute_ru_load_after_actions(
        self,
        prev_snapshot: TimeStep,
        actions: Dict[int, UEAction],
    ) -> Dict[int, int]:
        ru_load: Dict[int, int] = {ru_id: 0 for ru_id in self.topology.rus}

        for ue_id, prev_ue in prev_snapshot.ue_metrics.items():
            action = actions.get(ue_id)
            target_ru = prev_ue.serving_ru if action is None else action.target_ru
            ru_load[target_ru] = ru_load.get(target_ru, 0) + 1

        return ru_load

    def _estimate_throughput_mbps(
        self,
        sinr_db: float,
        cell_load: int,
        cell_type: str,
        did_ho: bool,
        missing_air_penalty: bool,
    ) -> float:
        # base rate thô theo SINR
        if sinr_db < 0.0:
            base_rate = 2.0
        elif sinr_db < 5.0:
            base_rate = 8.0
        elif sinr_db < 10.0:
            base_rate = 18.0
        elif sinr_db < 15.0:
            base_rate = 32.0
        else:
            base_rate = 50.0

        if cell_type == "macro":
            base_rate *= 0.9
        elif cell_type == "small":
            base_rate *= 1.1
            
        # chia tải
        tput = base_rate / max(1, cell_load)

        # penalty ngắn hạn do HO interruption
        if did_ho:
            tput *= 0.75

        # penalty nếu thiếu air metric cho target RU
        if missing_air_penalty:
            tput *= 0.60

        return max(0.1, tput)

    def _estimate_queue_bytes(
        self,
        prev_queue_bytes: float,
        arrival_bytes: float,
        throughput_mbps: float,
        dt_s: float = 1.0,
    ) -> float:
        served_bytes = throughput_mbps * 1e6 / 8.0 * dt_s
        new_queue = max(0.0, prev_queue_bytes + arrival_bytes - served_bytes)
        return new_queue

    def _estimate_latency_ms(
        self,
        queue_bytes: float,
        throughput_mbps: float,
        did_ho: bool,
        missing_air_penalty: bool,
    ) -> float:
        # latency cơ sở
        latency_ms = 3.0

        # queueing delay gần đúng
        service_bps = max(throughput_mbps, 0.1) * 1e6
        queue_delay_ms = (queue_bytes * 8.0 / service_bps) * 1000.0
        latency_ms += queue_delay_ms

        # interruption / reconfiguration penalty khi HO
        if did_ho:
            latency_ms += 8.0

        if missing_air_penalty:
            latency_ms += 5.0

        return latency_ms