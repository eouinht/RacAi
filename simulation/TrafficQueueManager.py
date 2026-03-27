import numpy as np
from typing import Tuple


def sinr_db_to_linear(
    sinr_db: np.ndarray,
) -> np.ndarray:
    """
    Input:
        sinr_db: (N,) float

    Output:
        sinr_linear: (N,) float
    """
    return np.power(10.0, sinr_db / 10.0).astype(np.float64)


def estimate_ue_throughput_bps(
    serving_sinr_db: np.ndarray,
    allocated_bandwidth_hz: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """
    Shannon-like surrogate throughput:
        R = eta * BW_alloc * log2(1 + SINR)

    Input:
        serving_sinr_db: (N,) float
        allocated_bandwidth_hz: (N,) float
        eta: (N,) float

    Output:
        throughput_bps: (N,) float
    """
    sinr_linear = sinr_db_to_linear(serving_sinr_db)
    spectral_eff = np.log2(1.0 + sinr_linear)
    throughput_bps = eta * allocated_bandwidth_hz * spectral_eff
    return np.maximum(throughput_bps, 0.0).astype(np.float64)


def estimate_equal_share_bandwidth_hz(
    serving_ru: np.ndarray,
    total_bandwidth_hz: float,
    n_ru: int,
) -> np.ndarray:
    """
    Equal-share surrogate:
    each UE in the same RU shares that RU bandwidth equally.

    Input:
        serving_ru: (N,) int
        total_bandwidth_hz: float
        n_ru: int

    Output:
        allocated_bandwidth_hz: (N,) float
    """
    ue_count_per_ru = np.bincount(serving_ru, minlength=n_ru).astype(np.float64)
    allocated_bandwidth_hz = total_bandwidth_hz / np.maximum(ue_count_per_ru[serving_ru], 1.0)
    return allocated_bandwidth_hz.astype(np.float64)


def generate_traffic_arrival_bits(
    lambda_arrival_bps: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Simple fluid arrival model:
        arrival_bits = lambda_arrival_bps * dt

    Input:
        lambda_arrival_bps: (N,) float
        dt: float

    Output:
        arrival_bits: (N,) float
    """
    arrival_bits = lambda_arrival_bps * dt
    return np.maximum(arrival_bits, 0.0).astype(np.float64)


def update_queue_backlog_bits(
    queue_bits: np.ndarray,
    arrival_bits: np.ndarray,
    served_bits: np.ndarray,
) -> np.ndarray:
    """
    Queue update:
        q_next = max(0, q + arrival - served)

    Input:
        queue_bits: (N,) float
        arrival_bits: (N,) float
        served_bits: (N,) float

    Output:
        new_queue_bits: (N,) float
    """
    new_queue_bits = queue_bits + arrival_bits - served_bits
    return np.maximum(new_queue_bits, 0.0).astype(np.float64)


def estimate_queue_delay_s(
    queue_bits: np.ndarray,
    throughput_bps: np.ndarray,
) -> np.ndarray:
    """
    Queueing delay surrogate:
        delay = queue / throughput

    Input:
        queue_bits: (N,) float
        throughput_bps: (N,) float

    Output:
        delay_s: (N,) float
    """
    delay_s = queue_bits / np.maximum(throughput_bps, 1e-9)
    return np.maximum(delay_s, 0.0).astype(np.float64)


def check_qos_violation(
    throughput_bps: np.ndarray,
    delay_s: np.ndarray,
    r_min_bps: np.ndarray,
    delay_max_s: np.ndarray,
) -> np.ndarray:
    """
    Input:
        throughput_bps: (N,) float
        delay_s: (N,) float
        r_min_bps: (N,) float
        delay_max_s: (N,) float

    Output:
        qos_violation: (N,) bool
    """
    throughput_violation = throughput_bps < r_min_bps
    delay_violation = delay_s > delay_max_s
    qos_violation = throughput_violation | delay_violation
    return qos_violation.astype(bool)


def estimate_traffic_queue_state(
    serving_ru: np.ndarray,
    serving_sinr_db: np.ndarray,
    queue_bits: np.ndarray,
    eta: np.ndarray,
    lambda_arrival_bps: np.ndarray,
    r_min_bps: np.ndarray,
    delay_max_s: np.ndarray,
    total_bandwidth_hz: float,
    n_ru: int,
    dt: float,
) -> dict:
    """
    Input:
        serving_ru: (N,) int
        serving_sinr_db: (N,) float
        queue_bits: (N,) float
        eta: (N,) float
        lambda_arrival_bps: (N,) float
        r_min_bps: (N,) float
        delay_max_s: (N,) float
        total_bandwidth_hz: float
        n_ru: int
        dt: float

    Output:
        traffic_state: dict
            allocated_bandwidth_hz: (N,)
            throughput_bps: (N,)
            arrival_bits: (N,)
            served_bits: (N,)
            queue_bits_next: (N,)
            delay_s_next: (N,)
            qos_violation: (N,)
            ue_count_per_ru: (M,)
    """
    ue_count_per_ru = np.bincount(serving_ru, minlength=n_ru).astype(np.float64)

    allocated_bandwidth_hz = estimate_equal_share_bandwidth_hz(
        serving_ru=serving_ru,
        total_bandwidth_hz=total_bandwidth_hz,
        n_ru=n_ru,
    )

    throughput_bps = estimate_ue_throughput_bps(
        serving_sinr_db=serving_sinr_db,
        allocated_bandwidth_hz=allocated_bandwidth_hz,
        eta=eta,
    )

    arrival_bits = generate_traffic_arrival_bits(
        lambda_arrival_bps=lambda_arrival_bps,
        dt=dt,
    )

    served_bits = throughput_bps * dt

    queue_bits_next = update_queue_backlog_bits(
        queue_bits=queue_bits,
        arrival_bits=arrival_bits,
        served_bits=served_bits,
    )

    delay_s_next = estimate_queue_delay_s(
        queue_bits=queue_bits_next,
        throughput_bps=throughput_bps,
    )

    qos_violation = check_qos_violation(
        throughput_bps=throughput_bps,
        delay_s=delay_s_next,
        r_min_bps=r_min_bps,
        delay_max_s=delay_max_s,
    )

    return {
        "allocated_bandwidth_hz": allocated_bandwidth_hz,
        "throughput_bps": throughput_bps,
        "arrival_bits": arrival_bits,
        "served_bits": served_bits,
        "queue_bits_next": queue_bits_next,
        "delay_s_next": delay_s_next,
        "qos_violation": qos_violation,
        "ue_count_per_ru": ue_count_per_ru,
    }