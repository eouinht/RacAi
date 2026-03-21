from __future__ import annotations

from typing import Any, Dict

from models import TimeStep, Topology


class StateBuilder:
    """
    Giai đoạn đầu dùng dict-based state để dễ debug.
    Sau này có thể thêm hàm convert sang numpy / torch tensor.
    """

    def build(self, snapshot: TimeStep, topology: Topology) -> Dict[str, Any]:
        ue_states: Dict[int, Dict[str, Any]] = {}
        ru_load_count: Dict[int, int] = {ru_id: 0 for ru_id in topology.rus}

        for ue_id, ue in snapshot.ue_metrics.items():
            ru_load_count[ue.serving_ru] = ru_load_count.get(ue.serving_ru, 0) + 1

            ue_states[ue_id] = {
                "serving_ru": ue.serving_ru,
                "du_id": ue.du_id,
                "cu_id": ue.cu_id,
                "x": ue.x,
                "y": ue.y,
                "sinr_db": ue.sinr_db,
                "rsrp_dbm": ue.rsrp_dbm,
                "pathloss_db": ue.pathloss_db,
                "tput_mbps": ue.tput_mbps,
                "queue_bytes": ue.bsr_bytes,
                "latency_ms": ue.latency_ms,
                "mcs": ue.mcs,
                "cqi": ue.cqi,
                "slice_type": int(ue.slice_type),
                "traffic_class": ue.traffic_class.value,
                "payload_arrival_bytes": ue.payload_arrival_bytes,
                "control_demand": ue.control_demand,
                "candidate_cells": list(ue.candidate_cells),
                "air_metrics": {
                    ru_id: {
                        "rsrp_dbm": metric.rsrp_dbm,
                        "sinr_db": metric.sinr_db,
                    }
                    for ru_id, metric in ue.air_metrics.items()
                },
            }

        ru_states: Dict[int, Dict[str, Any]] = {}
        for ru_id, ru in topology.rus.items():
            ru_states[ru_id] = {
                "du_id": ru.du_id,
                "cu_id": ru.cu_id,
                "cell_type": ru.cell_type,
                "x": ru.x,
                "y": ru.y,
                "total_prb": ru.total_prb,
                "total_ptx": ru.total_ptx,
                "connected_ue_count": ru_load_count.get(ru_id, 0),
                "neighbors": topology.neighbors.get(ru_id, []),
            }

        return {
            "t": snapshot.t, 
            "ues": ue_states, 
            "rus": ru_states
            }