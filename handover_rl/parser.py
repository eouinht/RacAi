from __future__ import annotations

from json import JSONDecoder
from pathlib import Path
from typing import Any, Dict, List, Optional

from enums import SliceType, TrafficClass
from models import CellAirMetric, CU, DU, RU, TimeStep, Topology, TraceBundle, UEMetrics


class NS3TraceParser:
    """
    Parser cho trace ns-3.
    
    """

    def parse_file(self, path: str | Path) -> TraceBundle:
        bundle = TraceBundle()
        path = Path(path)

        text = path.read_text(encoding="utf-8")
        for record in self._iter_json_objects(text):
            self._consume_record(record, bundle)

        bundle.steps.sort(key=lambda step: step.t)
        return bundle

    def _iter_json_objects(self, text: str):
        decoder = JSONDecoder()
        idx = 0
        n = len(text)
        while idx < n:
            while idx < n and text[idx].isspace():
                idx += 1
            if idx >= n:
                break
            record, next_idx = decoder.raw_decode(text, idx)
            yield record
            idx = next_idx

    def _consume_record(self, record: Dict[str, Any], bundle: TraceBundle) -> None:
        rec_type = record.get("type")
        t = record.get("t")

        if rec_type == "config":
            bundle.config = record
            return

        if rec_type == "topology":
            bundle.topology = self._parse_topology(record)
            return

        if rec_type == "summary" or t == "summary":
            bundle.summary = record
            return

        if isinstance(t, int) and rec_type != "config" and rec_type != "topology":
            bundle.steps.append(self._parse_step(record, bundle.topology))

    def _parse_topology(self, record: Dict[str, Any]) -> Topology:
        topology = Topology()

        nodes = record.get("nodes") or record.get("cells") or []
        for node in nodes:
            ru_id = int(node.get("ru", node.get("ru_id", node.get("cell", -1))))
            du_id = int(node.get("du", node.get("du_id", ru_id)))
            cu_id = int(node.get("cu", node.get("cu_id", 0)))

            topology.rus[ru_id] = RU(
                ru_id=ru_id,
                du_id=du_id,
                cu_id=cu_id,
                cell_type=str(node.get("type", node.get("cell_type", "unknown"))),
                x=float(node.get("x", 0.0)),
                y=float(node.get("y", 0.0)),
                total_prb=self._to_optional_float(node.get("total_prb")),
                total_ptx=self._to_optional_float(node.get("total_ptx")),
            )

            if du_id not in topology.dus:
                topology.dus[du_id] = DU(du_id=du_id, cu_id=cu_id)
            if cu_id not in topology.cus:
                topology.cus[cu_id] = CU(cu_id=cu_id)

        if not topology.neighbors:
            ru_ids = sorted(topology.rus.keys())
            topology.neighbors = {ru_id: [x for x in ru_ids if x != ru_id] for ru_id in ru_ids}

        for item in record.get("neighbors", []):
            ru_id = int(item.get("ru_id", item.get("ru", -1)))
            nbrs = item.get("neighbor_ru_ids", item.get("neighbors", []))
            topology.neighbors[ru_id] = [int(x) for x in nbrs]

        return topology

    def _parse_step(self, record: Dict[str, Any], topology: Topology) -> TimeStep:
        snapshot = TimeStep(t=int(record["t"]), raw=record)

        for ue_block in self._discover_ue_blocks(record):
            ue = self._parse_ue_block(ue_block, topology, record)
            snapshot.ue_metrics[ue.ue_id] = ue

        return snapshot

    def _discover_ue_blocks(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        conn = record.get("conn", {})

        for key, value in record.items():
            if not (key.startswith("ue") and isinstance(value, dict) and key[2:].isdigit()):
                continue

            ue_id = int(key[2:])
            candidate = dict(value)
            candidate.setdefault("ue_id", ue_id)

            conn_info = conn.get(key, {}) if isinstance(conn, dict) else {}
            if isinstance(conn_info, dict):
                candidate.setdefault("ru", conn_info.get("ru"))
                candidate.setdefault("du", conn_info.get("du"))
                candidate.setdefault("cu", conn_info.get("cu"))

            candidate.setdefault("x", record.get("ue_x"))
            candidate.setdefault("y", record.get("ue_y"))
            out.append(candidate)

        return out

    def _parse_ue_block(
        self,
        ue_block: Dict[str, Any],
        topology: Topology,
        full_record: Dict[str, Any],
    ) -> UEMetrics:
        ue_id = int(ue_block.get("ue_id", -1))
        
        serving_ru = int(
            ue_block.get(
                "serving_cell",
                ue_block.get("serving_ru", ue_block.get("ru", full_record.get("serving_cell", -1))),
            )
        )

        du_id = self._to_optional_int(ue_block.get("du"))
        if du_id is None and serving_ru in topology.rus:
            du_id = topology.get_du(serving_ru)

        cu_id = self._to_optional_int(ue_block.get("cu"))
        if cu_id is None and serving_ru in topology.rus:
            cu_id = topology.get_cu(serving_ru)

        ue = UEMetrics(
            ue_id=ue_id,
            serving_ru=serving_ru,
            du_id=du_id,
            cu_id=cu_id,
            x=self._to_optional_float(ue_block.get("x")),
            y=self._to_optional_float(ue_block.get("y")),
            sinr_db=self._to_optional_float(ue_block.get("sinr_dB")),
            rsrp_dbm=self._to_optional_float(ue_block.get("rsrp_dBm")),
            pathloss_db=self._to_optional_float(ue_block.get("pathloss_dB")),
            tput_mbps=self._to_optional_float(ue_block.get("tput_Mbps")),
            bsr_bytes=float(ue_block.get("bsr_B", 0.0) or 0.0),
            latency_ms=self._to_optional_float(ue_block.get("latency_ms")),
            mcs=self._to_optional_int(ue_block.get("mcs")),
            cqi=self._to_optional_int(ue_block.get("cqi")),
            ho_src=self._to_optional_int(ue_block.get("ho_src")),
            ho_dst=self._to_optional_int(ue_block.get("ho_dst")),
            slice_type=self._parse_slice_type(ue_block.get("slice_type")),
            traffic_class=self._parse_traffic_class(ue_block.get("traffic_class")),
            payload_arrival_bytes=float(ue_block.get("payload_arrival_B", 0.0) or 0.0),
            control_demand=float(ue_block.get("control_demand", 0.0) or 0.0),
        )

        ue.air_metrics = self._extract_air_metrics(full_record)
        ue.candidate_cells = sorted(ue.air_metrics.keys())
        return ue

    def _extract_air_metrics(self, record: Dict[str, Any]) -> Dict[int, CellAirMetric]:
        out: Dict[int, CellAirMetric] = {}
        air = record.get("air", {})
        if not isinstance(air, dict):
            return out

        for key, value in air.items():
            if not (key.startswith("cell") and isinstance(value, dict)):
                continue
            suffix = key.replace("cell", "")
            try:
                cell_id = int(suffix)
            except ValueError:
                continue
            ru_id = max(0, cell_id - 1)
            out[ru_id] = CellAirMetric(
                ru_id=ru_id,
                rsrp_dbm=self._to_optional_float(value.get("rsrp_dBm")),
                sinr_db=self._to_optional_float(value.get("sinr_dB")),
            )
        return out

    @staticmethod
    def _parse_slice_type(value: Any) -> SliceType:
        if isinstance(value, str) and value.lower() == "urllc":
            return SliceType.URLLC
        if value == 1:
            return SliceType.URLLC
        return SliceType.EMBB

    @staticmethod
    def _parse_traffic_class(value: Any) -> TrafficClass:
        if isinstance(value, str) and value.lower() == "control":
            return TrafficClass.CONTROL
        return TrafficClass.PAYLOAD

    @staticmethod
    def _to_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None