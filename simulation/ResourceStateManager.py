import numpy as np
from typing import Dict


# ================================
# INIT RESOURCE STATE
# ================================

def init_resource_state(
    serving_ru: np.ndarray,
    prb_total: int,
    ru_prb_cap: int,
    n_ru: int,
) -> Dict:
    """
    Initialize resource allocation based on UE distribution

    Input:
        serving_ru: (N,) int
        prb_total: int
        ru_prb_cap: int
        n_ru: int

    Output:
        resource_state: dict
    """

    total_ue = serving_ru.shape[0]

    # number of UE per RU
    ue_count_per_ru = np.bincount(serving_ru, minlength=n_ru).astype(np.float64)

    # PRB per UE
    prb_per_ue = prb_total / max(total_ue, 1)

    # initial RU allocation
    ru_prb_allocated = prb_per_ue * ue_count_per_ru

    # apply RU cap
    ru_prb_allocated = np.minimum(ru_prb_allocated, ru_prb_cap)

    # UE-level PRB (equal inside RU)
    ue_allocated_prb = prb_per_ue * np.ones(total_ue, dtype=np.float64)

    # recompute actual RU usage
    ru_used_prb = np.zeros(n_ru, dtype=np.float64)
    
    for ru in range(n_ru):
        mask = serving_ru == ru
        ru_used_prb[ru] = np.sum(ue_allocated_prb[mask])

    # remaining PRB in each RU
    ru_available_prb = ru_prb_allocated - ru_used_prb

    # global pool remaining
    total_allocated = np.sum(ru_prb_allocated)
    prb_pool_free = prb_total - total_allocated

    return {
        "ru_prb_allocated": ru_prb_allocated,   # (M,)
        "ru_used_prb": ru_used_prb,             # (M,)
        "ru_available_prb": ru_available_prb,             # (M,)
        "ue_allocated_prb": ue_allocated_prb,   # (N,)
        "prb_pool_free": prb_pool_free,         # scalar
        "ue_count_per_ru": ue_count_per_ru,     # (M,)
    }


# ================================
# UPDATE RU USAGE
# ================================

def update_ru_usage(
    serving_ru: np.ndarray,
    ue_allocated_prb: np.ndarray,
    n_ru: int,
) -> Dict:
    """
    Recompute RU usage after allocation changes

    Output:
        ru_used_prb: (M,)
    """
    ru_used_prb = np.zeros(n_ru, dtype=np.float64)

    for ru in range(n_ru):
        mask = serving_ru == ru
        ru_used_prb[ru] = np.sum(ue_allocated_prb[mask])

    return ru_used_prb


# ================================
# RELEASE EXTRA PRB
# ================================

def release_unused_prb(
    ru_prb_allocated: np.ndarray,
    ru_used_prb: np.ndarray,
) -> Dict:
    """
    Release unused PRB from RU back to pool

    Output:
        ru_prb_allocated_new
        released_prb_total
    """
    ru_available_prb = ru_prb_allocated - ru_used_prb

    # release all free PRB
    released_prb_total = np.sum(np.maximum(ru_available_prb, 0.0))

    ru_prb_allocated_new = ru_used_prb.copy()

    return {
        "ru_prb_allocated": ru_prb_allocated_new,
        "released_prb": released_prb_total,
    }


# ================================
# REQUEST EXTRA PRB FOR RU
# ================================

def request_prb_for_ru(
    ru_id: int,
    required_prb: float,
    ru_prb_allocated: np.ndarray,
    ru_prb_cap: float,
    prb_pool_free: float,
) -> Dict:
    """
    Try to allocate more PRB from global pool to a RU

    Output:
        success: bool
        new_ru_prb_allocated
        new_prb_pool_free
    """

    current = ru_prb_allocated[ru_id]

    # max RU can take
    max_additional = ru_prb_cap - current

    # actual request
    request = min(required_prb, max_additional)

    if request <= prb_pool_free:
        ru_prb_allocated[ru_id] += request
        prb_pool_free -= request
        success = True
    else:
        success = False

    return {
        "success": success,
        "ru_prb_allocated": ru_prb_allocated,
        "prb_pool_free": prb_pool_free,
    }


# ================================
# COMPUTE UE PRB FROM RU
# ================================

def redistribute_prb_within_ru(
    serving_ru: np.ndarray,
    ru_prb_allocated: np.ndarray,
    n_ru: int,
) -> np.ndarray:
    """
    Equal-share inside RU (baseline)

    Output:
        ue_allocated_prb: (N,)
    """
    N = serving_ru.shape[0]

    ue_allocated_prb = np.zeros(N, dtype=np.float64)

    for ru in range(n_ru):
        mask = serving_ru == ru
        n_ue = np.sum(mask)

        if n_ue > 0:
            prb_per_ue = ru_prb_allocated[ru] / n_ue
            ue_allocated_prb[mask] = prb_per_ue

    return ue_allocated_prb