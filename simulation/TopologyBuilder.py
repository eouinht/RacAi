import numpy as np
from typing import Tuple

def generate_ru_positions( 
                          n_ru:int, 
                          area_size: float,
                          offset: float
                         ) -> np.ndarray:
    """
    Returns:
        ru_pos: (5, 2)
    """

    cx = area_size / 2.0
    cy = area_size / 2.0

    ru_pos = np.array([
        [cx, cy],                    # RU0 center
        [cx - offset, cy + offset], # RU1 top-left
        [cx + offset, cy + offset], # RU2 top-right
        [cx - offset, cy - offset], # RU3 bottom-left
        [cx + offset, cy - offset], # RU4 bottom-right
    ], dtype=np.float64)
    
    return ru_pos


# Map link động (RU-DU-CU) cho phép số lượng thực thể thay đổi 
def map_ru_to_du( n_ru: int, n_du: int) -> np.ndarray:
    """
    Output:
        ru_to_du: (M,) int
    """
    return np.random.randint(0, n_du, size=n_ru)

def map_du_to_cu(n_du: int, n_cu: int) -> np.ndarray:

    return np.random.randint(0, n_cu, size=n_du)

def build_link_matrix(src_to_dst: np.ndarray, n_src: int, n_dst: int
                      ) -> np.ndarray:
    """
    Input:
        src_to_dst: (n_src,)
    Output:
        link_matrix: (n_src, n_dst) binary
    """
    matrix = np.zeros((n_src, n_dst), dtype=np.int32)
    matrix[np.arange(n_src), src_to_dst] = 1
    return matrix

def build_capacity_vector(n: int, cap_value: float)-> np.ndarray:
    return np.full(n, cap_value, dtype=np.float64)

def build_topology( n_ru: int,
                   n_du: int, 
                   n_cu: int, 
                   ru_prb_cap: int, 
                   du_cpu_cap: float, 
                   cu_cpu_cap: float,
                   layout_type: str, 
                   area_size: float,
                   offset: float
) -> dict:
    """
    Output:
        topology: dict
            ru_pos: (M, 2)
            ru_to_du: (M,)
            du_to_cu: (D,)
            l_ru_du: (M, D)
            l_du_cu: (D, C)
            ru_prb_cap: (M,)
            du_cpu_cap: (D,)
            cu_cpu_cap: (C,)
    """

    # 1. positions
    ru_pos = generate_ru_positions(n_ru)

    # 2. mapping
    ru_to_du = map_ru_to_du(n_ru, n_du)
    du_to_cu = map_du_to_cu(n_du, n_cu)

    # 3. link matrices
    l_ru_du = build_link_matrix(ru_to_du, n_ru, n_du)
    l_du_cu = build_link_matrix(du_to_cu, n_du, n_cu)

    # 4. capacities
    ru_cap = build_capacity_vector(n_ru, ru_prb_cap)
    du_cap = build_capacity_vector(n_du, du_cpu_cap)
    cu_cap = build_capacity_vector(n_cu, cu_cpu_cap)

    return {
        "ru_pos": ru_pos,
        "ru_to_du": ru_to_du,
        "du_to_cu": du_to_cu,
        "l_ru_du": l_ru_du,
        "l_du_cu": l_du_cu,
        "ru_prb_cap": ru_cap,
        "du_cpu_cap": du_cap,
        "cu_cpu_cap": cu_cap,
    }
