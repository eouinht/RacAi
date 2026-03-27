import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

radius_in  = 10        # m
radius_out = 1000      # m
slice_ratio = [0.7, 0.3]   # 70% eMBB, 30% uRLLC
def gen_coordinates_RU(num_RUs, radius_out = radius_out):
    circle_RU_out = radius_out * 0.65
    angles = np.linspace(0, 2 * np.pi, num_RUs - 1, endpoint=False) 
    x = np.concatenate(([0], circle_RU_out * np.cos(angles)))  
    y = np.concatenate(([0], circle_RU_out * np.sin(angles)))  
    coordinates_RU = list(zip(x, y)) 
    return coordinates_RU

def gen_coordinates_UE(num_UEs, radius_in = radius_in, radius_out = radius_out):

    angles = np.random.uniform(0, 2 * np.pi, num_UEs)
    r = np.random.uniform(radius_in, radius_out, num_UEs)
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    coordinates_UE = list(zip(x, y))  
    return coordinates_UE



def calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs):
    distances_RU_UE = np.zeros((num_RUs, num_UEs))
    for i in range(num_RUs):
        for j in range(num_UEs):
            x_RU, y_RU = coordinates_RU[i]
            x_UE, y_UE = coordinates_UE[j]
            distances_RU_UE[i, j] = np.sqrt((x_RU - x_UE)**2 + (y_RU - y_UE)**2)
    return distances_RU_UE



def adjust_coordinates_UE(coordinates_UE, delta_coordinate):
    # Khởi tạo seed cho ngẫu nhiên để kết quả có thể tái tạo
    new_coordinates_UE = []
    
    for x, y in coordinates_UE:
        # Tạo độ lệch ngẫu nhiên trong khoảng [-delta_coordinate, delta_coordinate] cho cả x và y
        delta_x = np.random.uniform(-delta_coordinate, delta_coordinate)
        delta_y = np.random.uniform(-delta_coordinate, delta_coordinate)
        
        # Tọa độ mới sau khi thêm độ lệch
        new_x = x + delta_x
        new_y = y + delta_y
        
        # Thêm tọa độ mới vào danh sách
        new_coordinates_UE.append((new_x, new_y))
    
    return new_coordinates_UE


def gen_UE_requirements(num_UEs, SLICE_PRESET, slice_ratio = slice_ratio):
    """
    Gán loại slice (eMBB hoặc uRLLC) cho từng UE dựa theo tỷ lệ slice_ratio.
    Trả về: danh sách tên slice cho num_UEs UEs.
    """
    slice_names = list(SLICE_PRESET.keys())  # ['eMBB', 'uRLLC']

    # Chuẩn hóa xác suất (đảm bảo tổng = 1)
    probs = np.array(slice_ratio, dtype=float)
    probs = probs / probs.sum()

    # Chọn ngẫu nhiên loại slice cho từng UE
    UE_slice_name = np.random.choice(slice_names, size=num_UEs, p=probs)

    return list(UE_slice_name)



def plot_save_network(coordinates_RU, coordinates_UE, radius_in, radius_out, output_folder_time):
    # Vẽ các vòng tròn cho Inner và Outer Radius
    circle_in = plt.Circle((0, 0), radius_in, color='gray', fill=False, linestyle='--')
    circle_out = plt.Circle((0, 0), radius_out, color='black', fill=False, linestyle='--')
    
    plt.gca().add_artist(circle_in)
    plt.gca().add_artist(circle_out)
    
    # Vẽ các điểm RU
    for (x, y) in coordinates_RU:
        plt.scatter(x, y, color='green', marker='^', s=100, label='RU' if (x, y) == (0, 0) else "")
    
    # Vẽ các điểm UE
    for index, (x, y) in enumerate(coordinates_UE):
        plt.scatter(x, y, color='blue', marker='o')
        if index == 0:  # Chỉ chú thích cho UE đầu tiên
            plt.scatter(x, y, color='blue', marker='o', label='UE')

    # Cài đặt giới hạn trục và các đặc tính đồ họa
    plt.xlim(-radius_out * 1.2, radius_out * 1.2)
    plt.ylim(-radius_out * 1.2, radius_out * 1.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()

    # Thêm tiêu đề và nhãn cho các trục
    plt.title("Network with RU and UE")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right')
    
    # Tạo thư mục kết quả nếu chưa tồn tại
    os.makedirs(output_folder_time, exist_ok=True)
    
    # Đặt tên file lưu với thời gian hiện tại
    fig_name = os.path.join(output_folder_time, f"network_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
    
    # Lưu hình vẽ dưới định dạng PDF
    plt.savefig(fig_name, format="PDF")
    plt.close()  # Đóng để tránh hiển thị ảnh thêm nữa
    print(f"Network saved in {fig_name}")

