import numpy as np
from typing import Dict

from .HandoverFeasibleChecker import check_handover_feasibility

def apply_single_handover(
    ue_id: int,
    source_ru: int,
    target_ru: int,
    required_prb: float,
    resource_state: Dict,
    du_cpu_used: np.ndarray,
    cu_cpu_used: np.ndarray,
    du_cpu_required: float,
    cu_cpu_required: float,
    topology: Dict
)-> Dict:
    """
    Apply HO: update resource + CPU + serving

    Output:
        updated resource_state, du_cpu_used, cu_cpu_used
    """
    old_prb = resource_state["ue_allocated_prb"][ue_id]
    
    resource_state["ue_allocated_prb"]["ue_id"] = required_prb
    resource_state["ru_used_prb"] -= old_prb
     
    # allocated to target ru 
    resource_state["ru_used_prb"][target_ru] += required_prb
    
    # update free prb
    resource_state["ru_free_prb"] = (resource_state["ru_prb_allocated"] - resource_state["ru_used_prb"])
    
    # update cpu load
    source_du = topology["ru_to_du"][source_ru]
    source_cu = topology["du_to_cu"][source_du]
    
    target_du = topology["ru_to_du"][target_ru]
    target_cu = topology["du_to_cu"][target_du]
    
    du_cpu_used[source_du] -= du_cpu_required
    cu_cpu_required[source_cu] -= cu_cpu_required
    
    du_cpu_used[target_du] += du_cpu_required
    cu_cpu_used[target_cu] += cu_cpu_required
    
    return {
        "resource_state": resource_state,
        "du_cpu_used": du_cpu_used,
        "cu_cpu_used": cu_cpu_used,
    }
    
def process_candidate_ues(
    candidate_mask: np.ndarray,
    radio_state: Dict,
    topology: Dict,
    resource_state: Dict,
    r_min_bps: np.ndarray,
    delay_max_s: np.ndarray,
    packet_size_bits: np.ndarray,
    lambda_arrival_bps: np.ndarray,
    du_cpu_required: np.ndarray,
    cu_cpu_required: np.ndarray,
    du_cpu_used: np.ndarray,
    cu_cpu_used: np.ndarray,
    rb_bandwidth_hz: float,
    ru_prb_cap: float,
)-> Dict:
    """
    Loop through candidate UEs and apply HO if feasible.

    Output:
        updated state
    """
    
    serving_ru = radio_state["serving_ru"].copy()
    best_neighbor_ru = radio_state["best_neighbor_ru"]
    
    for ue_id in np.where(candidate_mask)[0]:
        source_ru = int(serving_ru[ue_id])
        target_ru = int(best_neighbor_ru[ue_id])
        
        if target_ru == source_ru:
            continue
            
        result = check_handover_feasibility(
            ue_id=ue_id,
            target_ru=target_ru,
            radio_state=radio_state,
            topology=topology,
            resource_state=resource_state,
            rb_bandwidth_hz=rb_bandwidth_hz,
            r_min_bps=r_min_bps,
            delay_max_s=delay_max_s,
            packet_size_bits=packet_size_bits,
            lambda_arrival_bps=lambda_arrival_bps,
            du_cpu_required=du_cpu_required,
            cu_cpu_required=cu_cpu_required,
            du_cpu_used=du_cpu_used,
            cu_cpu_used=cu_cpu_used,
            ru_prb_cap=ru_prb_cap
        )
        
        if not result["feasible"]:
            continue
        
        # === APPLY HO ===
        apply_out = apply_single_handover(
            ue_id=ue_id,
            source_ru=source_ru,
            target_ru=target_ru,
            required_prb=result["required_prb"],
            resource_state=resource_state,
            du_cpu_used=du_cpu_used,
            cu_cpu_used=cu_cpu_used,
            du_cpu_required=du_cpu_required[ue_id],
            cu_cpu_required=cu_cpu_required[ue_id],
            topology=topology,
        )
        resource_state = apply_out["resource_state"]
        du_cpu_used = apply_out["du_cpu_used"]
        cu_cpu_used = apply_out["cu_cpu_used"]
        serving_ru[ue_id] = target_ru
        
    return {
    "serving_ru": serving_ru,
    "resource_state": resource_state,
    "du_cpu_used": du_cpu_used,
    "cu_cpu_used": cu_cpu_used,
    }