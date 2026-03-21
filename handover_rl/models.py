from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from enums import HandoverType, SliceType, TrafficClass

@dataclass
class RU:
    ru_id: int
    du_id: int
    cu_id: int 
    cell_type: str = "unknown"
    x: float = 0.0
    y: float = 0.0
    total_prb: Optional[float] = None
    total_ptx: Optional[float] = None 

@dataclass
class DU:
    du_id: int
    cu_id: int
    capacity: Optional[float] = None

@dataclass
class CU:
    cu_id: int
    capacity: Optional[float] = None

@dataclass
class Topology:
    rus: Dict[int, RU]= field(default_factory=dict)
    dus: Dict[int, DU] = field(default_factory=dict)
    cus: Dict[int, CU] = field(default_factory=dict)
    neighbors: Dict[int, List[int]] = field(default_factory=dict)
    
    def get_du(self, ru_id:int) -> Optional[int]:
        ru = self.rus.get(ru_id)
        return None if ru is None else ru.du_id
    
    def get_cu(self, ru_id:int) -> Optional[int]:
        ru = self.rus.get(ru_id)
        return None if ru is None else ru.cu_id
    
    def classify_handover(self, source_ru:int, target_ru:int) -> HandoverType:
        if source_ru == target_ru:
            return HandoverType.NO_HO
        
        src_ru = self.rus[source_ru]
        tgt_ru = self.rus[target_ru]
        
        if src_ru.du_id == tgt_ru.du_id and src_ru.cu_id == tgt_ru.cu_id:
            return HandoverType.INTRA_DU_INTRA_CU
        
        if src_ru.du_id != tgt_ru.du_id and src_ru.cu_id == tgt_ru.cu_id:
            return HandoverType.INTER_DU_INTRA_CU

        return HandoverType.INTER_CU
    
    
@dataclass
class CellAirMetric:
    ru_id: int
    rsrp_dbm: Optional[float] = None
    sinr_db: Optional[float] = None

@dataclass
class UEMetrics:
    ue_id: int
    serving_ru: int
    du_id: Optional[int] = None
    cu_id: Optional[int] = None 
    x: Optional[float] = None
    y: Optional[float] = None
    sinr_db: Optional[float] = None 
    rsrp_dbm: Optional[float] = None 
    pathloss_db: Optional[float] = None 
    tput_mbps: Optional[float] = None 
    bsr_bytes: float = 0.0 
    latency_ms: Optional[float] = None 
    mcs: Optional[int] = None    
    cqi: Optional[int] = None
    ho_src: Optional[int] = None
    ho_dst: Optional[int] = None
    slice_type: SliceType = SliceType.EMBB
    traffic_class: TrafficClass= TrafficClass.PAYLOAD
    payload_arrival_bytes:float = 0.0
    control_demand: float = 0.0
    candidate_cells: List[int] = field(default_factory=list)
    air_metrics:Dict[int, CellAirMetric]= field(default_factory=dict)
    
@dataclass
class TimeStep:
    t: int
    ue_metrics: Dict[int, UEMetrics] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class TraceBundle:
    config: Dict[str, Any] = field(default_factory=dict)
    topology: Topology = field(default_factory=Topology)
    steps: List[TimeStep] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UEAction:
    target_ru: int
    prb_alloc: float = 0.0
    ptx_alloc: float = 0.0
    du_alloc: float = 0.0
    cu_alloc: float = 0.0
    
@dataclass
class RewardWeights:
    throughput: float = 1.0
    delay: float = 1.0
    queue: float = 0.1
    handover: float = 1.0

@dataclass
class HandoverCosts:
    intra_du_intra_cu: float = 1.0
    inter_du_intra_cu: float = 3.0
    inter_cu: float = 6.0
    
    def get(self, ho_type: HandoverType) -> float:
        if ho_type == HandoverType.NO_HO:
            return 0.0
        if ho_type == HandoverType.INTRA_DU_INTRA_CU:
            return self.intra_du_intra_cu
        if ho_type == HandoverType.INTER_DU_INTRA_CU:
            return self.inter_du_intra_cu
        return self.inter_cu

@dataclass
class EnvConfig:
    delay_threshold_ms: float = 10.0
    default_slice_type: SliceType = SliceType.EMBB
    default_traffic_class: TrafficClass = TrafficClass.PAYLOAD
    