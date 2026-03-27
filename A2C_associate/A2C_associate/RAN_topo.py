import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

def create_topo(num_RUs, num_DUs, num_CUs, P_i_random_list, A_j_random_list, A_m_random_list, bw_ru_du_random_list, bw_du_cu_random_list):
    G = nx.Graph()

    # Tạo danh sách các nút RU, DU và CU
    RUs = [f'RU{i+1}' for i in range(num_RUs)]
    DUs = [f'DU{i+1}' for i in range(num_DUs)]
    CUs = [f'CU{i+1}' for i in range(num_CUs)]

    # Thêm các nút RU, DU và CU vào đồ thị
    for ru in RUs:
        G.add_node(ru, type='RU', power = np.random.choice(P_i_random_list))
    for du in DUs:
        G.add_node(du, type='DU', capacity = np.random.choice(A_j_random_list))
    for cu in CUs:
        G.add_node(cu, type='CU', capacity = np.random.choice(A_m_random_list))

    # Kết nối RUs với DUs (Mỗi DU có thể kết nối với tất cả các RU)
    for du in DUs:
        for ru in RUs:
            G.add_edge(ru, du, link_type="RU-DU", bandwidth=np.random.choice(bw_ru_du_random_list))

    # Kết nối DUs với CUs (Mỗi DU kết nối với tất cả các CU)
    for du in DUs:
        for cu in CUs:
            G.add_edge(du, cu, link_type="DU-CU", bandwidth=np.random.choice(bw_du_cu_random_list))
    return G


def get_links(G):
    # Lấy danh sách các node theo loại
    RUs = [n for n, d in G.nodes(data=True) if d['type'] == 'RU']
    DUs = [n for n, d in G.nodes(data=True) if d['type'] == 'DU']
    CUs = [n for n, d in G.nodes(data=True) if d['type'] == 'CU']

    # Khởi tạo ma trận băng thông RU–DU và DU–CU (đơn vị bps)
    l_ru_du = np.zeros((len(RUs), len(DUs)))
    l_du_cu = np.zeros((len(DUs), len(CUs)))

    # Duyệt qua các cạnh trong đồ thị
    for u, v, data in G.edges(data=True):
        bw = data.get('bandwidth', 0.0)

        if G.nodes[u]['type'] == 'RU' and G.nodes[v]['type'] == 'DU':
            l_ru_du[RUs.index(u), DUs.index(v)] = bw
        elif G.nodes[u]['type'] == 'DU' and G.nodes[v]['type'] == 'RU':
            l_ru_du[RUs.index(v), DUs.index(u)] = bw
        elif G.nodes[u]['type'] == 'DU' and G.nodes[v]['type'] == 'CU':
            l_du_cu[DUs.index(u), CUs.index(v)] = bw
        elif G.nodes[u]['type'] == 'CU' and G.nodes[v]['type'] == 'DU':
            l_du_cu[DUs.index(v), CUs.index(u)] = bw

    return l_ru_du, l_du_cu

def get_node_cap(G):
    ru_weights = []  # Mảng chứa trọng số của các nút RU
    du_weights = []  # Mảng chứa trọng số của các nút DU
    cu_weights = []  # Mảng chứa trọng số của các nút CU

    # Duyệt qua tất cả các nút trong đồ thị
    for node, data in G.nodes(data=True):
        if data['type'] == 'RU':  # Nếu nút là RU
            ru_weights.append(data['power'])
        if data['type'] == 'DU':  # Nếu nút là DU
            du_weights.append(data['capacity'])
        elif data['type'] == 'CU':  # Nếu nút là CU
            cu_weights.append(data['capacity'])

    return ru_weights, du_weights, cu_weights


def get_links_2(G):
    """Tạo ma trận liên kết toàn mạng (bps) giữa các node trong G."""
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    link_bw_topo = np.zeros((N, N))

    for u, v, data in G.edges(data=True):
        bw = data.get('bandwidth', 0.0)
        i, j = idx[u], idx[v]
        link_bw_topo[i, j] = link_bw_topo[j, i] = bw 

    return link_bw_topo

def get_node_cap_2(G):
    """Trả về vector trọng số node toàn mạng và danh sách node."""
    nodes = list(G.nodes())
    node_cap_topo = np.array([
        G.nodes[n].get('power', G.nodes[n].get('capacity', 0.0))
        for n in nodes
    ], dtype=float)
    return node_cap_topo


# Hàm vẽ đồ thị
def draw_topo(G, output_folder_time):
    # Lọc các nút RU, DU và CU từ đồ thị dựa trên thuộc tính 'type'
    RUs = [node for node, data in G.nodes(data=True) if data['type'] == 'RU']
    DUs = [node for node, data in G.nodes(data=True) if data['type'] == 'DU']
    CUs = [node for node, data in G.nodes(data=True) if data['type'] == 'CU']
    
    # Vị trí của các nút: RU, DU, CU xếp thành cột
    pos = {ru: (0, 3 - i) for i, ru in enumerate(RUs)}
    pos.update({du: (1, 2.5 - i * 2) for i, du in enumerate(DUs)})
    pos.update({cu: (2, 2 - i) for i, cu in enumerate(CUs)})

    # Vẽ đồ thị với các tùy chỉnh
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')

    # Vẽ các nút
    node_colors = ['lightblue' if 'RU' in node else 'lightgreen' if 'DU' in node else 'lightcoral' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, linewidths=2)

    # Hiển thị dung lượng chỉ cho các nút DU và CU
    node_labels = {node: f"{node}\nCap: {data['capacity']}" if 'capacity' in data else f"{node}" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold', font_color='black')

    plt.title(f"Network Model: {len(RUs)} RU, {len(DUs)} DU, {len(CUs)} CU (Column Layout)", fontsize=15)
    plt.axis('off')  # Tắt trục
    plt.tight_layout()  # Điều chỉnh bố cục

    # Tạo thư mục kết quả nếu chưa tồn tại
    os.makedirs(output_folder_time, exist_ok=True)

    # Đặt tên file lưu với thời gian hiện tại
    fig_name = os.path.join(output_folder_time, f"network_topology_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")

    # Lưu hình ảnh dưới định dạng PDF
    plt.savefig(fig_name, format="PDF")
    plt.close()  # Đóng để tránh hiển thị ảnh thêm nữa
    print(f"Topo RAN saved in {fig_name}")
