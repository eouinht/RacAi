import numpy as np
from numpy.linalg import norm

def channel_gain(distances_RU_UE, num_RUs, num_UEs, bandwidth_per_RB):
    """
    distances_RU_UE: ma trận [num_RUs x num_UEs] khoảng cách RU-UE (m)
    bandwidth_per_RB: băng thông 1 RB (Hz)
    """
    # ------------------- Antenna config -------------------
    num_antennas = 32  # anten mỗi RU
    
    # ------------------- Noise power ----------------------
    k_B = 1.38064852e-23   # Boltzmann constant (J/K)
    T_K = 290              # Nhiệt độ (K)
    N0_W_per_Hz = k_B * T_K
    noise_figure_dB = 5
    noise_figure_linear = 10 ** (noise_figure_dB / 10)
    noise_power_RB = N0_W_per_Hz * bandwidth_per_RB * noise_figure_linear
    
    # ------------------- Carrier frequency ----------------
    f_c_GHz = 6
    
    # ------------------- Pathloss model (3GPP UMa) --------
    distances_RU_UE = np.maximum(distances_RU_UE, 1.0)  # tránh log(0)
    
    # scenarios (GHz) UMa (TR 38.901)
    # Môi trường: Thành phố, nhà cao tâng
    # Bán kính cell 500m - 1km
    path_loss_db = 28 + 20 * np.log10(f_c_GHz) + 22 * np.log10(distances_RU_UE/1.0)
    
    # ------------------- Pathloss linear ------------------
    path_loss_linear = 10 ** (-path_loss_db / 10)
    
    # ------------------- Rayleigh fading ------------------
    channel_matrix = np.zeros((num_RUs, num_UEs))
    for i in range(num_RUs):
        for k in range(num_UEs):
            # kênh MIMO Rayleigh (num_antennas anten)
            h_real = np.random.randn(num_antennas)
            h_imag = np.random.randn(num_antennas)
            h = np.sqrt(path_loss_linear[i, k]) * (h_real + 1j*h_imag) / np.sqrt(2)
            
            # power gain (chuẩn hóa theo norm-2)
            channel_matrix[i, k] = norm(h, 2) ** 2
    
    # ------------------- Channel gain (normalized by noise) ----
    gain = channel_matrix / noise_power_RB
    
    return gain
