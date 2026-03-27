import numpy as np
from typing import Tuple

def generate_ue_positions(n_ue: int, area_size: float = 500.0) -> np.ndarray:
    """
    Input:
        n_ue: number of UE
        area_size: square area size in meters

    Output:
        ue_pos: (N, 2) float
            UE positions [x, y]
    """
    return np.random.uniform(0.0, area_size, size=(n_ue, 2)).astype(np.float64)

#  Tìm mô hình di chuyển của UE (có thư viện càng tốt)
def generate_ue_velocities(n_ue: int, speed_mean: float, speed_std: float
                            ) -> np.ndarray:
    """
    Input:
        n_ue: number of UE
        speed_mean: mean UE speed in m/s
        speed_std: std UE speed in m/s

    Output:
        ue_vel: (N, 2) float
            UE velocity vectors [vx, vy]
    """
    speeds = np.random.normal(speed_mean, speed_std, size=n_ue)
    speeds = np.clip(speeds, 0.0, None)
    
    angles = np.random.uniform(0.0, 2.0*np.pi, size=n_ue)
    
    vx = speeds*np.cos(angles)
    vy = speeds*np.sin(angles)
    return np.stack([vx, vy], axis=1).astype(np.float64)

def generate_ue_slices(n_ue: int, embb_ratio: float = 0.7)->np.ndarray: 
    rand_val = np.random.uniform(0.0, 1.0, size=n_ue)
    ue_slice = np.where(rand_val < embb_ratio, 0, 1)
    
    return ue_slice.astype(np.int32)

def init_ue_state(n_ue:int, speed_mean: float, speed_std: float, area_size: float = 500, embb_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input:
        n_ue: number of UE
        speed_mean: mean UE speed in m/s
        speed_std: std UE speed in m/s
        area_size: square area size in meters
        embb_ratio: probability of assigning eMBB

    Output:
        ue_pos: (N, 2) float
        ue_vel: (N, 2) float
        ue_slice: (N,) int
    """
    ue_pos = generate_ue_positions(n_ue, area_size)
    ue_vel = generate_ue_velocities(n_ue, speed_mean, speed_std)
    ue_slice = generate_ue_slices(n_ue, embb_ratio)
    
    return ue_pos, ue_vel, ue_slice

def update_ue_positions(
    ue_pos: np.ndarray,
    ue_vel: np.ndarray,
    dt: float,
    area_size: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        ue_pos: (N, 2) float
        ue_vel: (N, 2) float
        dt: timestep in seconds
        area_size: square area size in meters

    Output:
        new_ue_pos: (N, 2) float
        new_ue_vel: (N, 2) float
    """
    new_ue_pos = ue_pos + ue_vel * dt
    new_ue_vel = ue_vel.copy()

    # reflect on x boundaries
    x_low_mask = new_ue_pos[:, 0] < 0.0
    x_high_mask = new_ue_pos[:, 0] > area_size
    new_ue_vel[x_low_mask | x_high_mask, 0] *= -1.0
    new_ue_pos[:, 0] = np.clip(new_ue_pos[:, 0], 0.0, area_size)

    # reflect on y boundaries
    y_low_mask = new_ue_pos[:, 1] < 0.0
    y_high_mask = new_ue_pos[:, 1] > area_size
    new_ue_vel[y_low_mask | y_high_mask, 1] *= -1.0
    new_ue_pos[:, 1] = np.clip(new_ue_pos[:, 1], 0.0, area_size)

    return new_ue_pos, new_ue_vel