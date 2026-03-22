from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ObsAdapterConfig:
    n_max: int = 16
    m_max: int = 16
    include_serving_ru_feature: bool = True
    include_cell_type_flags: bool = True
    queue_log_scale: bool = True
    delay_clip_ms: float = 1000.0
    sinr_clip_db: float = 40.0


class ObservationAdapter:
    """
    Chuyển state dict của env -> observation dict:
    {
        "ue_matrix": [N_max, F_ue],
        "cell_matrix": [M_max, F_cell],
        "ue_mask": [N_max],
        "cell_mask": [M_max],
    }
    """

    def __init__(self, cfg: ObsAdapterConfig) -> None:
        self.cfg = cfg

    def _safe_float(self, x, default: float = 0.0) -> float:
        if x is None:
            return default
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    def _log1p_nonneg(self, x: float) -> float:
        return float(np.log1p(max(x, 0.0)))

    def _best_neighbor_metrics(self, ue_state: dict) -> Dict[str, float]:
        serving_ru = int(ue_state["serving_ru"])
        serving_sinr = self._safe_float(ue_state.get("sinr_db"), 0.0)
        serving_rsrp = self._safe_float(ue_state.get("rsrp_dbm"), -140.0)

        best_neighbor_sinr = serving_sinr
        best_neighbor_rsrp = serving_rsrp

        air = ue_state.get("air_metrics", {})
        for ru_id, metric in air.items():
            sinr = metric.get("sinr_db")
            rsrp = metric.get("rsrp_dbm")

            if sinr is not None and float(sinr) > best_neighbor_sinr:
                best_neighbor_sinr = float(sinr)

            if ru_id != serving_ru and rsrp is not None and float(rsrp) > best_neighbor_rsrp:
                best_neighbor_rsrp = float(rsrp)

        return {
            "best_neighbor_sinr": best_neighbor_sinr,
            "best_neighbor_rsrp": best_neighbor_rsrp,
            "sinr_gap": best_neighbor_sinr - serving_sinr,
            "rsrp_gap": best_neighbor_rsrp - serving_rsrp,
        }

    def _build_ue_feature(self, ue_state: dict) -> np.ndarray:
        sinr = np.clip(self._safe_float(ue_state.get("sinr_db"), 0.0), -self.cfg.sinr_clip_db, self.cfg.sinr_clip_db)
        queue_bytes = self._safe_float(ue_state.get("queue_bytes"), 0.0)
        latency_ms = np.clip(self._safe_float(ue_state.get("latency_ms"), 0.0), 0.0, self.cfg.delay_clip_ms)
        slice_type = self._safe_float(ue_state.get("slice_type"), 0.0)
        tput_mbps = self._safe_float(ue_state.get("tput_mbps"), 0.0)
        serving_ru = self._safe_float(ue_state.get("serving_ru"), 0.0)

        ho_ctx = self._best_neighbor_metrics(ue_state)

        if self.cfg.queue_log_scale:
            queue_feature = self._log1p_nonneg(queue_bytes)
        else:
            queue_feature = queue_bytes

        features: List[float] = [
            sinr,
            queue_feature,
            latency_ms,
            slice_type,
            tput_mbps,
            ho_ctx["best_neighbor_sinr"],
            ho_ctx["sinr_gap"],
            ho_ctx["rsrp_gap"],
        ]

        if self.cfg.include_serving_ru_feature:
            features.append(serving_ru)

        return np.asarray(features, dtype=np.float32)

    def _build_cell_feature(self, ru_id: int, ru_state: dict) -> np.ndarray:
        total_prb = ru_state.get("total_prb")
        connected_ue_count = self._safe_float(ru_state.get("connected_ue_count"), 0.0)

        if total_prb is None:
            # fallback heuristic nếu trace chưa có total_prb
            util_prb = min(connected_ue_count / 10.0, 1.0)
        else:
            total_prb_f = max(self._safe_float(total_prb, 1.0), 1.0)
            util_prb = min(connected_ue_count / total_prb_f, 1.0)

        # Hiện chưa có usage DU/CU thật trong state -> dùng proxy đơn giản
        # Khi env giàu hơn có thể thay bằng giá trị thật.
        util_du = min(connected_ue_count / 10.0, 1.0)
        util_cu = min(connected_ue_count / 10.0, 1.0)

        cell_type = ru_state.get("cell_type", "unknown")
        is_small = 1.0 if cell_type == "small" else 0.0
        is_macro = 1.0 if cell_type == "macro" else 0.0

        features: List[float] = [
            util_prb,
            util_du,
            util_cu,
        ]

        if self.cfg.include_cell_type_flags:
            features.extend([is_small, is_macro])

        return np.asarray(features, dtype=np.float32)

    def adapt(self, state: dict) -> Dict[str, np.ndarray]:
        ue_ids = sorted(state["ues"].keys())
        ru_ids = sorted(state["rus"].keys())

        # suy ra số chiều động
        dummy_ue = self._build_ue_feature(state["ues"][ue_ids[0]]) if ue_ids else np.zeros((9,), dtype=np.float32)
        dummy_cell = self._build_cell_feature(ru_ids[0], state["rus"][ru_ids[0]]) if ru_ids else np.zeros((5,), dtype=np.float32)

        ue_feat_dim = int(dummy_ue.shape[0])
        cell_feat_dim = int(dummy_cell.shape[0])

        ue_matrix = np.zeros((self.cfg.n_max, ue_feat_dim), dtype=np.float32)
        cell_matrix = np.zeros((self.cfg.m_max, cell_feat_dim), dtype=np.float32)
        ue_mask = np.zeros((self.cfg.n_max,), dtype=np.float32)
        cell_mask = np.zeros((self.cfg.m_max,), dtype=np.float32)

        for i, ue_id in enumerate(ue_ids[: self.cfg.n_max]):
            ue_matrix[i] = self._build_ue_feature(state["ues"][ue_id])
            ue_mask[i] = 1.0

        for j, ru_id in enumerate(ru_ids[: self.cfg.m_max]):
            cell_matrix[j] = self._build_cell_feature(ru_id, state["rus"][ru_id])
            cell_mask[j] = 1.0

        return {
            "ue_matrix": ue_matrix,
            "cell_matrix": cell_matrix,
            "ue_mask": ue_mask,
            "cell_mask": cell_mask,
        }

    def get_feature_dims(self, state: dict) -> Dict[str, int]:
        obs = self.adapt(state)
        return {
            "ue_feat_dim": int(obs["ue_matrix"].shape[-1]),
            "cell_feat_dim": int(obs["cell_matrix"].shape[-1]),
        }