from dataclasses import dataclass
import numpy as np
from typing import Tuple

# ================================
# SLICE CONFIG
# ================================

@dataclass
class SliceConfig:
    name: str 
    r_min_bps: float        # minimum required throughput
    sinr_min_db: float      # minimum SINR
    delay_max_s: float      # max delay
    eta: float              # slice efficiency factor
    lambda_arrival_bps: float   # traffic arrival rate (bits/s)


# ================================
# MAIN SIMULATION CONFIG
# ================================

@dataclass
class SimulationConfig:
    # ---------- topology ----------
    n_ru: int
    n_du: int
    n_cu: int
    n_ue: int

    # ---------- radio ----------
    carrier_freq_ghz: float
    bandwidth_mhz: float
    rb_bandwidth_hz: float
    noise_figure_db: float
    ru_tx_power_dbm: float
    prb_total: int 

    # ---------- resource ----------
    ru_prb_cap: int
    du_cpu_cap: float
    cu_cpu_cap: float

    # ---------- cpu model ----------
    k_du: float
    k_cu: float

    # ---------- mobility ----------
    ue_speed_mean: float
    ue_speed_std: float

    # ---------- simulation ----------
    time_step_s: float
    n_steps: int

    # ---------- slice ----------
    embb_config: SliceConfig
    urllc_config: SliceConfig

    # ---------- random ----------
    seed: int = 42


# ================================
# DEFAULT CONFIG BUILDER
# ================================

def create_default_config() -> SimulationConfig:
    """
    Returns:
        cfg: SimulationConfig
    """
# Check lại 
    embb = SliceConfig(
        name = "EMBB",
        r_min_bps=2e6,
        sinr_min_db=0.0,
        delay_max_s=0.05,
        eta=0.1,
        lambda_arrival_bps=4e6,
    )
    urllc = SliceConfig(
        name = "URLLC",
        r_min_bps=1e6,
        sinr_min_db=5.0,
        delay_max_s=0.005,
        eta=0.2,
        lambda_arrival_bps=1e6,
    )
    

    cfg = SimulationConfig(
        # topology
        n_ru=5,
        n_du=3,
        n_cu=3,
        n_ue=50,
        # Thay đổi UE theo thời gian +-3 
        
        # radio
        carrier_freq_ghz=3.5,
        bandwidth_mhz=20.0,
        rb_bandwidth_hz=180e3,
        noise_figure_db=9.0,
        ru_tx_power_dbm=43.0,
        prb_total=100,

        # resource
        ru_prb_cap=40,
        du_cpu_cap=100.0,
        cu_cpu_cap=100.0,

        # cpu model
        k_du=1e-9,
        k_cu=1e-9,

        # mobility
        ue_speed_mean=1.5,
        ue_speed_std=0.5,

        # simulation
        time_step_s=0.1,
        n_steps=100,

        # slice
        embb_config = embb,
        urllc_config = urllc,

        # random
        seed=42,
    )

    return cfg


# ================================
# UTILITY FUNCTIONS
# ================================

def set_random_seed(cfg: SimulationConfig):
    """
    Set global random seed

    Input:
        cfg: SimulationConfig
    """
    np.random.seed(cfg.seed)


def get_slice_params(cfg: SimulationConfig, ue_slice: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract per-UE slice parameters

    Input:
        ue_slice: (N,) int
            0->EMBB
            1->URLLC

    Returns:
        r_min: (N,) float
        sinr_min: (N,) float
        delay_max: (N,) float
        eta: (N,) float
        lambda_arrival: (N,) float
    """
    N = ue_slice.shape[0]

    r_min_bps = np.zeros(N, dtype=np.float64)
    sinr_min_db = np.zeros(N, dtype=np.float64)
    delay_max_s = np.zeros(N, dtype=np.float64)
    eta = np.zeros(N, dtype=np.float64)
    lambda_arrival_bps = np.zeros(N, dtype=np.float64)

    embb_mask = ue_slice == 0
    urllc_mask = ue_slice == 1
    
    r_min_bps[embb_mask] = cfg.embb_config.r_min_bps
    sinr_min_db[embb_mask] = cfg.embb_config.sinr_min_db
    delay_max_s[embb_mask] = cfg.embb_config.delay_max_s
    eta[embb_mask] = cfg.embb_config.eta
    lambda_arrival_bps[embb_mask] = cfg.embb_config.lambda_arrival_bps

    r_min_bps[urllc_mask] = cfg.urllc_config.r_min_bps
    sinr_min_db[urllc_mask] = cfg.urllc_config.sinr_min_db
    delay_max_s[urllc_mask] = cfg.urllc_config.delay_max_s
    eta[urllc_mask] = cfg.urllc_config.eta
    lambda_arrival_bps[urllc_mask] = cfg.urllc_config.lambda_arrival_bps

    return r_min_bps, sinr_min_db, delay_max_s, eta, lambda_arrival_bps

