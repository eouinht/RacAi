import numpy as np
from typing import Dict 

from .LatencyModel import estimate_latency_state

def estimate_target_throughput_bps(
    target_gain: float,
    allocated_prb: float,
    power_alloc_w: float,
    rb_bandwidth_hz: float
)-> float:

    """
    Input:
        target_gain: float
        allocated_prb: float
        power_alloc_w: float
        rb_bandwidth_hz: float

    Output:
        throughput_bps: float
    """
    snr_per_rb = (power_alloc_w/max(allocated_prb, 1e-9)*target_gain)
    throughput_bps = allocated_prb * rb_bandwidth_hz * np.log2(1.0 + snr_per_rb)
    return float(max(throughput_bps, 0.0))

def estimate_required_prb_for_target(
    r_min_bps: float,
    target_gain: float,
    power_alloc_w: float,
    rb_bandwidth_hz: float,
    max_prb_search: int = 100
) -> float:
    """
    Input:
        r_min_bps: float
        target_gain: float
        power_alloc_w: float
        rb_bandwidth_hz: float

    Output:
        required_prb: float
    """
    
    candiate_prb = np.arange(1, max_prb_search + 1, dtype=np.float64)
    snr_per_rb = (power_alloc_w / candiate_prb) * target_gain
    rate = candiate_prb * rb_bandwidth_hz * np.log2(1.0 + snr_per_rb)
    
    feasible = np.where(rate >= r_min_bps)[0]
    if feasible.size > 0:
        return float(candiate_prb[feasible[0]])
    return float(candiate_prb[-1])
    
def check_prb_fesibility(
    ue_id: int,
    source_ru: int,
    target_ru: int,
    required_prb: float,
    resource_state: Dict,
    ru_prb_cap: float,
) -> Dict:
    """

    Output:
        dict:
            feasible: bool
            local_free_prb: float
            extra_needed_from_pool: float
    """
    current_prb = resource_state["ue_alloceted_prb"][ue_id]
    effective_pool_free = resource_state["prb_pool_free"] + current_prb
    
    local_free_prb = float(resource_state["ru_free_prb"][target_ru])
    
    if required_prb <= local_free_prb:
        return{
            "feasible": True,
            "local_free_prb": local_free_prb,
            "extra_needed_from_pool": 0.0
        }
    
    extra_needed = required_prb - local_free_prb
    local_cap_room = max(ru_prb_cap - resource_state["ru_prb_allocated"][target_ru], 0.0)
    
    feasible = (extra_needed <= effective_pool_free) and (extra_needed <= local_cap_room)
    
    return{
        "feasible": bool(feasible),
        "local_free_prb": local_free_prb,
        "extra_needed_from_pool": float(max(extra_needed,0.0))
    }

def check_du_cu_capacity(
    target_ru: int,
    topology: Dict,
    du_cpu_used: np.ndarray,
    cu_cpu_used: np.ndarray,
    du_cpu_required: float,
    cu_cpu_required: float
) -> Dict:
    """
    Input:
    

    Output:
        dict:
            feasible: bool
            target_du: int
            target_cu: int
    """
    
    target_du = int(topology["ru_to_du"][target_ru])
    target_cu = int(topology["du_to_cu"][target_du])
    
    du = du_cpu_used[target_du] + du_cpu_required <= topology["du_cpu_cap"][target_du]
    cu = cu_cpu_used[target_cu] + cu_cpu_required <= topology["cu_cpu_cap"][target_du]
    
    return {
        "feasible": bool(du and cu),
        "target_du": target_du,
        "target_cu": target_cu
    }
    
def check_latency_feasibility(
    target_distance_m: float,
    packet_size_bits: float,
    throughput_bps: float,
    arrival_rate_packets_per_s: float,
    du_cpu_required: float,
    du_cpu_capacity: float,
    cu_cpu_required: float,
    cu_cpu_capacity: float,
    du_cpu_used: float,
    cu_cpu_used: float,
    delay_max_s: float,
    ho_delay_s: float = 0.0
) -> Dict:
    """
    Output:
        dict:
            feasible: bool
            total_latency_s: float
    """
    du_service_rate = (du_cpu_capacity - du_cpu_used) / du_cpu_required
    cu_service_rate = (cu_cpu_capacity - cu_cpu_used) / cu_cpu_required

    
   
    latency_state = estimate_latency_state(
        serving_distance_m=np.array([target_distance_m]),
        packet_size_bits=np.array([packet_size_bits]),
        throughput_bps=np.array([throughput_bps]),
        arrival_rate_packets_per_s=np.array([arrival_rate_packets_per_s]),
        du_cpu_required=np.array([du_cpu_required]),
        du_cpu_capacity=np.array([du_cpu_capacity]),
        cu_cpu_required=np.array([cu_cpu_required]),
        cu_cpu_capacity=np.array([cu_cpu_capacity]),
        du_service_rate_packets_per_s=np.array([du_service_rate]),
        cu_service_rate_packets_per_s=np.array([cu_service_rate]),
        ho_delay_s=np.array([ho_delay_s])
    )
    
    total_latency_s = latency_state["total_latency_s"][0]
    feasible = total_latency_s <= delay_max_s
    
    return{
        "feasible": bool(feasible),
        "total_latency_s": total_latency_s
    }
    
def check_handover_feasibility(
    ue_id: int,
    source_ru:int,
    target_ru: int,
    radio_state: Dict,
    topology: Dict,
    resource_state: Dict,
    rb_bandwidth_hz: float,
    r_min_bps: np.ndarray,
    delay_max_s: np.ndarray,
    packet_size_bits: np.ndarray,
    lambda_arrival_bps: np.ndarray,
    du_cpu_required: np.ndarray,
    cu_cpu_required: np.ndarray,
    du_cpu_used: np.ndarray,
    cu_cpu_used: np.ndarray,
    ru_prb_cap: float,
    ho_delay_s: float = 0.0
)-> Dict:
    """
    Check whether UE can be handed over to target RU.

    Output:
        result: dict
            feasible: bool
            reason: str
            required_prb: float
            estimated_throughput_bps: float
            estimated_latency_s: float
            target_du: int
            target_cu: int
    """
    
    target_gain = float(radio_state["gain"][ue_id, target_ru])
    target_distance_m, = float(radio_state["distance_m"][ue_id, target_ru])
    power_alloc_w = float(resource_state["ue_power_alloc_w"][ue_id])
    
    required_prb = estimate_required_prb_for_target(
        r_min_bps,
        target_gain,
        power_alloc_w,
        rb_bandwidth_hz, 
        ru_prb_cap
    )
    
    prb_check = check_prb_fesibility(
        ue_id,
        source_ru,
        target_ru,
        required_prb,
        resource_state["ru_free_prb"],
        resource_state["prb_pool_free"],
        resource_state["ru_prb_allocated"],
        ru_prb_cap
    )
    
    if not prb_check["feasible"]:
        return {
            "feasible": False,
            "reason": "insufficient_prb",
            "required_prb": required_prb,
            "estimated_throughput_bps": 0.0,
            "estimated_latency_s": np.inf,
            "target_du": -1,
            "target_ru": -1
        }
    
    throughput_bps = estimate_target_throughput_bps(
        target_gain,
        required_prb,
        power_alloc_w,
        rb_bandwidth_hz

    )
    
    if throughput_bps < r_min_bps[ue_id]:
        return {
            "feasible": False,
            "reason": "insufficient_throughput",
            "required_prb": required_prb,
            "estimated_throughput_bps": throughput_bps,
            "estimated_latency_s": np.inf,
            "target_du": -1,
            "target_cu": -1,
        }
        
    path_check = check_du_cu_capacity(
        target_ru,
        topology,
        du_cpu_used,
        cu_cpu_used,
        du_cpu_required[ue_id],
        cu_cpu_required[ue_id]
    )
    
    if not path_check["feasible"]:
        return {
            "feasible": False,
            "reason": "insufficient_du_cu_capacity",
            "required_prb": required_prb,
            "estimated_throughput_bps": throughput_bps,
            "estimated_latency_s": np.inf,
            "target_du": path_check["target_du"],
            "target_cu": path_check["target_cu"],
        }
        
    arrival_rate_pps = lambda_arrival_bps[ue_id]/packet_size_bits[ue_id]
    
    target_du = path_check["target_du"]
    target_cu = path_check["target_cu"]
    
    du_service_rate_pps = max(arrival_rate_pps, 100.0)
    cu_service_rate_pps = max(arrival_rate_pps, 100.0)
    
    latency_check = check_latency_feasibility(
        target_distance_m,
        packet_size_bits[ue_id],
        throughput_bps,
        arrival_rate_pps,
        du_cpu_required[ue_id],
        topology["du_cpu_cap"][path_check["target_du"]],
        cu_cpu_required[ue_id],
        topology["cu_cpu_cap"][path_check["target_cu"]],
        du_service_rate_pps[path_check["target_du"]],
        cu_service_rate_pps[path_check["target_cu"]],
        delay_max_s[ue_id],
        ho_delay_s
    )
    
    if not latency_check["feasible"]:
        return {
            "feasible": False,
            "reason": "latency_violation",
            "required_prb": required_prb,
            "estimated_throughput_bps": throughput_bps,
            "estimated_latency_s": latency_check["total_latency_s"],
            "target_du": target_du,
            "target_cu": target_cu,
        }
        
    return {
            "feasible": False,
            "reason": "latency_violation",
            "required_prb": required_prb,
            "estimated_throughput_bps": throughput_bps,
            "estimated_latency_s": latency_check["total_latency_s"],
            "target_du": target_du,
            "target_cu": target_cu,
        }