import numpy as np
import os
import matplotlib.pyplot as plt
import cvxpy as cp
def extract_optimization_results(array, threshold=0.5):
    """
    Chuyển giá trị từ biến hoặc mảng biến CVXPY sang numpy.ndarray 0/1.
    Xử lý đúng cả trường hợp cp.Variable dạng ma trận (boolean matrix).
    """
    # Trường hợp là 1 biến CVXPY duy nhất (ma trận)
    if isinstance(array, cp.Expression):
        val = array.value
        if val is None:
            val = np.zeros(array.shape)
        result = np.rint(np.nan_to_num(val, nan=0.0)).astype(int)
        return result

    # Trường hợp là mảng numpy chứa các biến con (object)
    shape = array.shape
    flat_vals = []
    for v in array.flatten():
        val = getattr(v, "value", v)
        if val is None:
            val = 0.0
        elif isinstance(val, (list, np.ndarray)):
            val = float(np.mean(val))
        flat_vals.append(val)

    arr = np.array(flat_vals, dtype=float).reshape(shape)
    arr = np.nan_to_num(arr, nan=0.0)

    # Ép nhị phân theo ngưỡng
    arr = (arr >= threshold).astype(int)
    return arr

def extract_optimization_results_2(long_z_ib_sk):
    def extract_values(array):
        shape = array.shape
        flat_array = np.array([np.rint(x.value) for x in array.flatten()], dtype=int)  # Làm tròn về 0 hoặc 1
        return flat_array.reshape(shape)

    arr_long_z_ib_sk = extract_values(long_z_ib_sk)


    return arr_long_z_ib_sk


def generate_new_num_UEs(num_UEs, delta_num_UE):
    # Tính sai số ngẫu nhiên trong khoảng [-delta_num_UE, delta_num_UE]
    delta = np.random.randint(-delta_num_UE, delta_num_UE)

    # Tính số lượng UE mới
    new_num_UEs = num_UEs + delta
    # Đảm bảo số lượng UE không âm
    return max(new_num_UEs, 0)

def mapping_nearest_RU_UE(distances_RU_UE, slice_mapping, num_RUs, num_UEs, num_slices):
    # Khởi tạo biến nhị phân nearest_phi_i_sk với tất cả giá trị là 0
    nearest_phi_i_sk = np.zeros((num_RUs, num_slices, num_UEs), dtype=int)
    
    for k in range(num_UEs):
        # Tìm RU gần nhất cho UE k (tức là tìm i có khoảng cách nhỏ nhất với UE k)
        i_nearest = np.argmin(distances_RU_UE[:, k])
        
        # Xác định slice mà UE k đang chọn
        for s in range(num_slices):
            if slice_mapping[s, k] == 1:  # UE k thuộc slice s
                nearest_phi_i_sk[i_nearest, s, k] = 1  # Đánh dấu RU gần nhất phục vụ UE này

    return nearest_phi_i_sk

def mapping_random_RU_UE(num_RUs, num_UEs, num_slices, slice_mapping):

    # Khởi tạo ma trận ánh xạ ngẫu nhiên với tất cả giá trị là 0
    random_phi_i_sk = np.zeros((num_RUs, num_slices, num_UEs), dtype=int)

    for k in range(num_UEs):
        # Chọn RU ngẫu nhiên từ danh sách các RU có sẵn
        chosen_RU = np.random.randint(0, num_RUs)

        # Xác định slice mà UE k thuộc về
        for s in range(num_slices):
            if slice_mapping[s, k] == 1:  # UE k thuộc slice s
                random_phi_i_sk[chosen_RU, s, k] = 1  # Đánh dấu ánh xạ ngẫu nhiên

    return random_phi_i_sk


def save_simulation_parameters(output_folder_time, **parameters):
    # Tạo đường dẫn đến file .txt để lưu
    output_file = os.path.join(output_folder_time, "simulation_parameters.txt")
    
    # Mở file ở chế độ ghi (write) và lưu các tham số dưới dạng văn bản
    with open(output_file, 'w') as txt_file:
        # Lặp qua từng tham số trong dictionary `parameters` và ghi vào file
        for key, value in parameters.items():
            txt_file.write(f"{key}: {value}\n")
    
    print(f"Simulation parameters saved to {output_file}")

"""def save_results(result_prefix, execution_time, total_pi_sk, total_R_sk, total_z_ib_sk, total_p_ib_sk, objective, output_folder):
    # Tạo thư mục chứa kết quả cho thuật toán
    result_folder = os.path.join(output_folder, result_prefix)
    os.makedirs(result_folder, exist_ok=True)

    # Lưu kết quả vào các file
    time_file = os.path.join(result_folder, "times.txt")
    pi_sk_file = os.path.join(result_folder, "pi_sk.txt")
    R_sk_file = os.path.join(result_folder, "R_sk.txt")
    z_ib_sk_file = os.path.join(result_folder, "z_ib_sk.txt")
    p_ib_sk_file = os.path.join(result_folder, "p_ib_sk.txt")
    objective_file = os.path.join(result_folder, "objective.txt")

    # Append kết quả vào các file
    with open(time_file, "a") as f:
        f.write(f"{execution_time:.4f}\n")
    with open(pi_sk_file, "a") as f:
        f.write(f"{total_pi_sk}\n")
    with open(R_sk_file, "a") as f:
        f.write(f"{total_R_sk}\n")
    with open(z_ib_sk_file, "a") as f:
        f.write(f"{total_z_ib_sk}\n")
    with open(p_ib_sk_file, "a") as f:
        f.write(f"{total_p_ib_sk}\n")
    with open(objective_file, "a") as f:
        f.write(f"{objective}\n")
"""
"""def save_results_2(total_pi_sk, total_R_sk, total_z_ib_sk, total_p_ib_sk, objective, output_folder):
    # Tạo thư mục chứa kết quả cho thuật toán
    result_folder = output_folder
    os.makedirs(result_folder, exist_ok=True)

    # Lưu kết quả vào các file
    pi_sk_file = os.path.join(result_folder, "pi_sk.txt")
    R_sk_file = os.path.join(result_folder, "R_sk.txt")
    z_ib_sk_file = os.path.join(result_folder, "z_ib_sk.txt")
    p_ib_sk_file = os.path.join(result_folder, "p_ib_sk.txt")
    objective_file = os.path.join(result_folder, "objective.txt")

    # Append kết quả vào các file
    with open(pi_sk_file, "a") as f:
        f.write(f"{total_pi_sk}\n")
    with open(R_sk_file, "a") as f:
        f.write(f"{total_R_sk}\n")
    with open(z_ib_sk_file, "a") as f:
        f.write(f"{total_z_ib_sk}\n")
    with open(p_ib_sk_file, "a") as f:
        f.write(f"{total_p_ib_sk}\n")
    with open(objective_file, "a") as f:
        f.write(f"{objective}\n")
"""

def convert_to_array(num_RUs, num_slices, num_UEs, long_phi_i_sk):
    # Tạo mảng NumPy 3 chiều rỗng
    array_long_phi_i_sk = np.zeros((num_RUs, num_slices, num_UEs), dtype=float)

    # Ánh xạ giá trị từ long_phi_i_sk sang mảng NumPy
    for i, ru in enumerate(long_phi_i_sk):
        for s, slice_ in enumerate(ru):
            for k, variable in enumerate(slice_):
                array_long_phi_i_sk[i, s, k] = 1.0 if variable.value else 0.0

    return array_long_phi_i_sk



def save_results(result_prefix, execution_time, total_pi_sk, total_R_sk, total_z_ib_sk, total_p_ib_sk, objective, output_folder):
    # Tạo thư mục chứa kết quả cho thuật toán
    result_folder = os.path.join(output_folder, result_prefix)
    os.makedirs(result_folder, exist_ok=True)

    # Lưu kết quả vào các file
    time_file = os.path.join(result_folder, "times.txt")
    pi_sk_file = os.path.join(result_folder, "pi_sk.txt")
    R_sk_file = os.path.join(result_folder, "R_sk.txt")
    z_ib_sk_file = os.path.join(result_folder, "z_ib_sk.txt")
    p_ib_sk_file = os.path.join(result_folder, "p_ib_sk.txt")
    objective_file = os.path.join(result_folder, "objective.txt")

    # Append kết quả vào các file
    with open(time_file, "a") as f:
        f.write(f"{execution_time:.4f}\n")
    with open(pi_sk_file, "a") as f:
        f.write(f"{total_pi_sk}\n")
    with open(R_sk_file, "a") as f:
        f.write(f"{total_R_sk}\n")
    with open(z_ib_sk_file, "a") as f:
        f.write(f"{total_z_ib_sk}\n")
    with open(p_ib_sk_file, "a") as f:
        f.write(f"{total_p_ib_sk}\n")
    with open(objective_file, "a") as f:
        f.write(f"{objective}\n")


def save_results_SCA(result_prefix, execution_time, total_pi_sk, total_R_sk, total_z_ib_sk, total_p_ib_sk, objective_value, penalty_value, objective_without_penalty, objective_history, output_folder):
    # Tạo thư mục chứa kết quả cho thuật toán
    result_folder = os.path.join(output_folder, result_prefix)
    os.makedirs(result_folder, exist_ok=True)

    # Lưu kết quả vào các file
    time_file = os.path.join(result_folder, "times.txt")
    pi_sk_file = os.path.join(result_folder, "pi_sk.txt")
    R_sk_file = os.path.join(result_folder, "R_sk.txt")
    z_ib_sk_file = os.path.join(result_folder, "z_ib_sk.txt")
    p_ib_sk_file = os.path.join(result_folder, "p_ib_sk.txt")
    objective_file = os.path.join(result_folder, "objective.txt")
    penalty_file = os.path.join(result_folder, "penalty.txt")
    objective_without_penalty_file = os.path.join(result_folder, "objective_without_penalty.txt")
    objective_history_file = os.path.join(result_folder, "objective_history.txt")

    # Append kết quả vào các file
    with open(time_file, "a") as f:
        f.write(f"{execution_time:.4f}\n")
    with open(pi_sk_file, "a") as f:
        f.write(f"{total_pi_sk}\n")
    with open(R_sk_file, "a") as f:
        f.write(f"{total_R_sk}\n")
    with open(z_ib_sk_file, "a") as f:
        f.write(f"{total_z_ib_sk}\n")
    with open(p_ib_sk_file, "a") as f:
        f.write(f"{total_p_ib_sk}\n")
    with open(objective_file, "a") as f:
        f.write(f"{objective_value}\n")
    with open(penalty_file, "a") as f:
        f.write(f"{penalty_value}\n")
    with open(objective_without_penalty_file, "a") as f:
        f.write(f"{objective_without_penalty}\n")
    with open(objective_history_file, "a") as f:
        f.write(f"{objective_history}\n")

def save_results_SCA_long(result_prefix, execution_time, objective_value, total_R_sk, total_z_ib_sk, total_p_ib_sk, penalty_value, objective_without_penalty, objective_history, output_folder):
    # Tạo thư mục chứa kết quả cho thuật toán
    result_folder = os.path.join(output_folder, result_prefix)
    os.makedirs(result_folder, exist_ok=True)

    # Lưu kết quả vào các file
    time_file = os.path.join(result_folder, "times.txt")
    total_z_ib_sk_file = os.path.join(result_folder, "z_ib_sk.txt")
    total_p_ib_sk_file = os.path.join(result_folder, "p_ib_sk.txt")
    total_R_sk_file = os.path.join(result_folder, "R_sk.txt")
    objective_file = os.path.join(result_folder, "objective.txt")
    penalty_file = os.path.join(result_folder, "penalty.txt")
    objective_without_penalty_file = os.path.join(result_folder, "objective_without_penalty.txt")
    objective_history_file = os.path.join(result_folder, "objective_history.txt")

    # Append kết quả vào các file
    with open(time_file, "a") as f:
        f.write(f"{execution_time:.4f}\n")
    with open(total_z_ib_sk_file, "a") as f:
        f.write(f"{total_z_ib_sk}\n")
    with open(total_p_ib_sk_file, "a") as f:
        f.write(f"{total_p_ib_sk}\n")
    with open(total_R_sk_file, "a") as f:
        f.write(f"{total_R_sk}\n")
    with open(objective_file, "a") as f:
        f.write(f"{objective_value}\n")
    with open(penalty_file, "a") as f:
        f.write(f"{penalty_value}\n")
    with open(objective_without_penalty_file, "a") as f:
        f.write(f"{objective_without_penalty}\n")
    with open(objective_history_file, "a") as f:
        f.write(f"{objective_history}\n")


def save_results_SCA_short(result_prefix, execution_time, total_pi_sk, objective_value, output_folder):
    # Tạo thư mục chứa kết quả cho thuật toán
    result_folder = os.path.join(output_folder, result_prefix)
    os.makedirs(result_folder, exist_ok=True)

    # Lưu kết quả vào các file
    time_file = os.path.join(result_folder, "times.txt")
    pi_sk_file = os.path.join(result_folder, "pi_sk.txt")
    objective_file = os.path.join(result_folder, "objective.txt")

    # Append kết quả vào các file
    with open(time_file, "a") as f:
        f.write(f"{execution_time:.4f}\n")
    with open(pi_sk_file, "a") as f:
        f.write(f"{total_pi_sk}\n")
    with open(objective_file, "a") as f:
        f.write(f"{objective_value}\n")
    




def round_all_binary_variables(pi_sk, z_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk):
    pi_bin = np.zeros(pi_sk.shape, dtype=int)
    for s in range(pi_sk.shape[0]):
        for k in range(pi_sk.shape[1]):
            val = pi_sk[s, k].value
            if val is None:
                raise ValueError(f"pi_sk[{s},{k}] chưa được gán giá trị.")
            pi_bin[s, k] = int(round(val))

    z_bin = np.zeros(z_ib_sk.shape, dtype=int)
    for i in range(z_ib_sk.shape[0]):
        for b in range(z_ib_sk.shape[1]):
            for s in range(z_ib_sk.shape[2]):
                for k in range(z_ib_sk.shape[3]):
                    val = z_ib_sk[i, b, s, k].value
                    if val is None:
                        raise ValueError(f"z_ib_sk[{i},{b},{s},{k}] chưa được gán giá trị.")
                    z_bin[i, b, s, k] = int(round(val))

    phi_i_bin = np.zeros(phi_i_sk.shape, dtype=int)
    for i in range(phi_i_sk.shape[0]):
        for s in range(phi_i_sk.shape[1]):
            for k in range(phi_i_sk.shape[2]):
                val = phi_i_sk[i, s, k].value
                if val is None:
                    raise ValueError(f"phi_i_sk[{i},{s},{k}] chưa được gán giá trị.")
                phi_i_bin[i, s, k] = int(round(val))

    phi_j_bin = np.zeros(phi_j_sk.shape, dtype=int)
    for j in range(phi_j_sk.shape[0]):
        for s in range(phi_j_sk.shape[1]):
            for k in range(phi_j_sk.shape[2]):
                val = phi_j_sk[j, s, k].value
                if val is None:
                    raise ValueError(f"phi_j_sk[{j},{s},{k}] chưa được gán giá trị.")
                phi_j_bin[j, s, k] = int(round(val))

    phi_m_bin = np.zeros(phi_m_sk.shape, dtype=int)
    for m in range(phi_m_sk.shape[0]):
        for s in range(phi_m_sk.shape[1]):
            for k in range(phi_m_sk.shape[2]):
                val = phi_m_sk[m, s, k].value
                if val is None:
                    raise ValueError(f"phi_m_sk[{m},{s},{k}] chưa được gán giá trị.")
                phi_m_bin[m, s, k] = int(round(val))

    return pi_bin, z_bin, phi_i_bin, phi_j_bin, phi_m_bin



def plot_convergence(obj_values):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(obj_values)+1), obj_values, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Convergence of SCA Algorithm')
    plt.grid(True)
    plt.show()




def convert_z_to_phi(z_ib_sk, epsilon=1e-6):
    # z_ib_sk: numpy array shape (num_RUs, num_RBs, num_slices, num_UEs)
    num_RUs, num_RBs, num_slices, num_UEs = z_ib_sk.shape
    
    phi_i_sk = np.zeros((num_RUs, num_slices, num_UEs), dtype=int)
    
    for i in range(num_RUs):
        for s in range(num_slices):
            for k in range(num_UEs):
                avg_z = np.sum(z_ib_sk[i, :, s, k]) / num_RBs
                # Điều kiện phi_i_sk >= avg_z, phi_i_sk nhị phân nên:
                phi_i_sk[i, s, k] = 1 if avg_z > epsilon else 0
    
    return phi_i_sk






def check_feasible(long_pi_sk, long_z_ib_sk, long_phi_i_sk, long_phi_j_sk, long_phi_m_sk):
    """
    Kiểm tra tính khả thi cơ bản trước khi giải Short_Doraemon.
    - long_pi_sk: (num_slices, num_UEs)
    - long_z_ib_sk: (num_RUs, num_RBs, num_slices, num_UEs)
    - long_phi_i_sk: (num_RUs, num_slices, num_UEs)
    - long_phi_j_sk: (num_DUs, num_slices, num_UEs)
    - long_phi_m_sk: (num_CUs, num_slices, num_UEs)
    """
    num_slices, num_UEs = long_pi_sk.shape
    num_RUs, num_RBs, _, _ = long_z_ib_sk.shape

    feasible = True
    infeas_cases = []

    for s in range(num_slices):
        for k in range(num_UEs):
            if long_pi_sk[s, k] == 1:
                # --- Kiểm tra có RB cấp hay không ---
                total_z = np.sum(long_z_ib_sk[:, :, s, k])
                if total_z == 0:
                    infeas_cases.append((s, k, "accept nhưng không có RB nào cấp"))
                    feasible = False

                # --- Kiểm tra ánh xạ RU/DU/CU ---
                phi_i = np.sum(long_phi_i_sk[:, s, k])
                phi_j = np.sum(long_phi_j_sk[:, s, k])
                phi_m = np.sum(long_phi_m_sk[:, s, k])

                if phi_i == 0:
                    infeas_cases.append((s, k, "thiếu RU ánh xạ"))
                    feasible = False
                if phi_j == 0:
                    infeas_cases.append((s, k, "thiếu DU ánh xạ"))
                    feasible = False
                if phi_m == 0:
                    infeas_cases.append((s, k, "thiếu CU ánh xạ"))
                    feasible = False

    # --------- In kết quả kiểm tra ----------
    print("========== CHECK FEASIBILITY ==========")
    print(f"Tổng số slice: {num_slices}, UE: {num_UEs}")
    print("--------------------------------------")

    if feasible:
        print("✅ Mọi UE được accept đều có RB, RU, DU, CU hợp lệ.")
    else:
        print("❌ Phát hiện các trường hợp infeasible:")
        for (s, k, reason) in infeas_cases:
            print(f"   - Slice {s}, UE {k}: {reason}")

    print("=======================================")
    return feasible, infeas_cases










