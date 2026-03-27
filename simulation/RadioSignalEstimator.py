import numpy as np
from typing import Tuple

def compute_distance_matrix(ue_pos: np.ndarray, ru_pos: np.ndarray) -> np.ndarray:
    """
    Input:
        ue_pos: (N, 2) float
        ru_pos: (M, 2) float

    Output:
        distance_m: (N, M) float
    """
    diff = ue_pos[:, None, :] - ru_pos[None, :, :]
    distance_m = np.linalg.norm(diff, axis=2)
    return np.maximum(distance_m, 1.0).astype(np.float64)

def compute_pathloss_db(distance_m: np.ndarray, carrier_freq_ghz: float) -> np.ndarray:
    """
    Free-space-like pathloss surrogate.

    Input:
        distance_m: (N, M) float
        carrier_freq_ghz: float

    Output:
        pathloss_db: (N, M) float
    """
    pathloss_db = (
        32.4
        + 20.0 * np.log10(distance_m)
        + 20.0 * np.log10(carrier_freq_ghz)
    )
    return pathloss_db.astype(np.float64)

def compute_rsrp_dbm(pathloss_db: np.ndarray, ru_tx_power_dbm: float) -> np.ndarray:
    """
    Input:
        pathloss_db: (N, M) float
        ru_tx_power_dbm: float

    Output:
        rsrp_dbm: (N, M) float
    """
    rsrp_dbm = ru_tx_power_dbm - pathloss_db
    return rsrp_dbm.astype(np.float64)

def compute_noise_power_dbm(bandwidth_mhz: float, noise_figure_db: float) -> float:
    """
    Thermal noise approximation:
        noise_dbm = -174 + 10*log10(B_hz) + NF

    Input:
        bandwidth_mhz: float
        noise_figure_db: float

    Output:
        noise_power_dbm: float
    """
    bandwidth_hz = bandwidth_mhz * 1e6
    noise_power_dbm = -174.0 + 10.0 * np.log10(bandwidth_hz) + noise_figure_db
    return float(noise_power_dbm)


def dbm_to_mw(power_dbm: np.ndarray) -> np.ndarray:
    """
    Input:
        power_dbm: ndarray

    Output:
        power_mw: ndarray
    """
    return np.power(10.0, power_dbm / 10.0)


def mw_to_db(power_mw: np.ndarray) -> np.ndarray:
    """
    Input:
        power_mw: ndarray

    Output:
        power_db: ndarray
    """
    return 10.0 * np.log10(np.maximum(power_mw, 1e-12))


def compute_sinr_db(rsrp_dbm: np.ndarray, noise_power_dbm: float) -> np.ndarray:
    """
    Approximate full-frequency reuse interference:
    for each UE and RU:
        SINR_i,j = signal_j / (sum(other signals) + noise)

    Input:
        rsrp_dbm: (N, M) float
        noise_power_dbm: float

    Output:
        sinr_db: (N, M) float
    """
    signal_mw = dbm_to_mw(rsrp_dbm)              # (N, M)
    total_power_mw = np.sum(signal_mw, axis=1, keepdims=True)
    noise_mw = dbm_to_mw(np.array(noise_power_dbm, dtype=np.float64))

    interference_mw = total_power_mw - signal_mw
    sinr_linear = signal_mw / np.maximum(interference_mw + noise_mw, 1e-12)
    sinr_db = 10.0 * np.log10(np.maximum(sinr_linear, 1e-12))

    return sinr_db.astype(np.float64)


def select_serving_ru( rsrp_dbm: np.ndarray ) -> np.ndarray:
    """
    Input:
        rsrp_dbm: (N, M) float

    Output:
        serving_ru: (N,) int
    """
    return np.argmax(rsrp_dbm, axis=1).astype(np.int32)


def extract_serving_metrics( rsrp_dbm: np.ndarray, sinr_db: np.ndarray, serving_ru: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        rsrp_dbm: (N, M) float
        sinr_db: (N, M) float
        serving_ru: (N,) int

    Output:
        serving_rsrp_dbm: (N,) float
        serving_sinr_db: (N,) float
    """
    ue_idx = np.arange(serving_ru.shape[0])
    serving_rsrp_dbm = rsrp_dbm[ue_idx, serving_ru]
    serving_sinr_db = sinr_db[ue_idx, serving_ru]
    return serving_rsrp_dbm.astype(np.float64), serving_sinr_db.astype(np.float64)


def extract_best_neighbor_ru( rsrp_dbm: np.ndarray, serving_ru: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        rsrp_dbm: (N, M) float
        serving_ru: (N,) int

    Output:
        best_neighbor_ru: (N,) int
        best_neighbor_rsrp_dbm: (N,) float
    """
    rsrp_copy = rsrp_dbm.copy()
    ue_idx = np.arange(serving_ru.shape[0])
    rsrp_copy[ue_idx, serving_ru] = -1e12

    best_neighbor_ru = np.argmax(rsrp_copy, axis=1).astype(np.int32)
    best_neighbor_rsrp_dbm = rsrp_copy[ue_idx, best_neighbor_ru].astype(np.float64)

    return best_neighbor_ru, best_neighbor_rsrp_dbm


def estimate_radio_state( ue_pos: np.ndarray, ru_pos: np.ndarray, carrier_freq_ghz: float, bandwidth_mhz: float, noise_figure_db: float, ru_tx_power_dbm: float
) -> dict:
    """
    Input:
        ue_pos: (N, 2) float
        ru_pos: (M, 2) float
        carrier_freq_ghz: float
        bandwidth_mhz: float
        noise_figure_db: float
        ru_tx_power_dbm: float

    Output:
        radio_state: dict
            distance_m: (N, M)
            pathloss_db: (N, M)
            rsrp_dbm: (N, M)
            sinr_db: (N, M)
            serving_ru: (N,)
            serving_rsrp_dbm: (N,)
            serving_sinr_db: (N,)
            best_neighbor_ru: (N,)
            best_neighbor_rsrp_dbm: (N,)
            noise_power_dbm: float
    """
    distance_m = compute_distance_matrix(ue_pos=ue_pos, ru_pos=ru_pos)
    pathloss_db = compute_pathloss_db(
        distance_m=distance_m,
        carrier_freq_ghz=carrier_freq_ghz,
    )
    rsrp_dbm = compute_rsrp_dbm(
        pathloss_db=pathloss_db,
        ru_tx_power_dbm=ru_tx_power_dbm,
    )
    noise_power_dbm = compute_noise_power_dbm(
        bandwidth_mhz=bandwidth_mhz,
        noise_figure_db=noise_figure_db,
    )
    sinr_db = compute_sinr_db(
        rsrp_dbm=rsrp_dbm,
        noise_power_dbm=noise_power_dbm,
    )
    serving_ru = select_serving_ru(rsrp_dbm=rsrp_dbm)
    serving_rsrp_dbm, serving_sinr_db = extract_serving_metrics(
        rsrp_dbm=rsrp_dbm,
        sinr_db=sinr_db,
        serving_ru=serving_ru,
    )
    best_neighbor_ru, best_neighbor_rsrp_dbm = extract_best_neighbor_ru(
        rsrp_dbm=rsrp_dbm,
        serving_ru=serving_ru,
    )

    return {
        "distance_m": distance_m,
        "pathloss_db": pathloss_db,
        "rsrp_dbm": rsrp_dbm,
        "sinr_db": sinr_db,
        "serving_ru": serving_ru,
        "serving_rsrp_dbm": serving_rsrp_dbm,
        "serving_sinr_db": serving_sinr_db,
        "best_neighbor_ru": best_neighbor_ru,
        "best_neighbor_rsrp_dbm": best_neighbor_rsrp_dbm,
        "noise_power_dbm": noise_power_dbm,
    }