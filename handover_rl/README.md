- Đọc dữ liệu trace từ ns-3
- Dựng lại topo mạng và trạng thái UE theo từng timestep
- Cho agent chọn action
- Môi trường cập nhật trạng thái mạng theo action
- Tính reward để đánh giá action
- Chưa có thuật toán  
* Các file chính:
    * main.py: entry point
    * env.py: Định nghĩa môi trường RL
        * reset môi trường
        * Nhận action
        * Cập nhật state
        * Tính reward 
        * Trả về state mới
    * reward_engine.py: Định nghĩa hàm phần thưởng bao gồm
        * throughput
        * queue
        * delay
        * handover cost 
    * state_builder.py: Định nghĩa những dữ liệu trong state
    * parser.py: Đọc json thành data được dùng trong mô phỏng  
    * models.py: Định nghĩa các dataclass và cấu trúc dữ liệu cơ bản
    * enums.py: Phân loại Ho, slice type, traffic class theo kiểu enum

    * ppo folder:
        * obs_adaptor.py: chuyển state dict thành tensor cố định cho neural network
        * encoder.py: mã hóa obs
        * actor_critic.py: sinh policy và value
        * train.py dùng PPO actor-critic để train policy
    

* models.py - Định nghĩa dữ liệu
    Định nghĩa các class RU, DU, CU:
    * RU:
        * ru_id
        * du_id
        * cu_id 
        * (x,y)
        * total_prb 
        * total_ptx 
        * cell_type
    * DU:
        * du_id
        * cu_id 
        * capacity 
    * CU:
        * cu_id 
        * capacity

    * UEMetrics: Mô tả trạng thái UE tại một thời điểm
        * ue_id: ID của UE
        * serving_ru: RU mà UE đang kết nối
        * du_id: DU tương ứng với serving RU
        * cu_id: CU tương ứng với serving RU
        * (x,y): vị trí của UE
        * sinr_db: SINR hiện tại tới serving cell
        * rsrp_dbm: RSRP hiện tại tới serving cell
        * path_loss_db: path loss
        * tput_mbps: throughput hiện tại
        * bsr_bytes: queue / buffer status theo byte
        * latencty_ms: độ trễ hiện tại
        * mcs: modulation and coding scheme
        * cqi: channel quality indicator
        * ho_src: nguồn handover nếu trace có ghi lại
        * ho_dst: đích handover nếu trace có ghi lại
        * slice_type: loại slice, ví dụ eMBB hay URLLC
        * traffic_class: loại traffic, ví dụ payload hay control
        * payload_arrival_bytes: lưu lượng payload mới đến trong bước đó
        * control_demmand: nhu cầu control traffic
        * candidate_cells: danh sách các RU mà UE có thể xem xét
        * air_metrics: dictionary chứa CellAirMetric cho từng candidate cell

    * Topology:
        * rus: danh sách các RU
        * dus: danh sách các DU 
        * cus: danh sách các CU

    * get_du(ru_id): Trả về du_id của RU tương ứng 
    * get_cu(ru_id): tar về cu_id của RU tương ứng

    * Phân loại HO:
        * NO_HO: Không HO
        * INTRA_DU_INTRA_CU: HO Khác RU cùng DU, CU
        * INTER_DU_INTRA_CU: HO khác RU, khác DU, cùng CU
        * INTER_CU: còn lại 

    * CellAirMetric: Thông tin chất lượng tín hiệu từ một UE đến một cell cụ thể
        * ru_id
        * rsrp_dbm 
        * sinr_db

    * UEAction: 
        * target_ru: RU đích mà agent muốn UE chuyển tới
        * prb_alloc: tài nguyên PRB cấp cho UE
        * ptx_alloc: công suất phát cấp cho UE
        * du_alloc: phần tài nguyên DU
        * cu_alloc: phần tài nguyên CU

    * TimeStep:
        * t
        * ue_metrics 
        * raw: record gốc từ trace

    * TraceBundle: là object cấp cao hơn, đại diện cho toàn bộ trace đã parse xong
        * config 
        * topology 
        * steps 
        * summary

    * RewardWieghts: Class này chứa trọng số cho từng thành phần reward.
        * throughput
        * delay
        * queue
        * handover

    * Handover cost: Class này chứa cost của từng loại HO
        * intra_du_intra_cu: nhẹ
        * inter_du_intra_cu: vừa
        * inter_cu: khá
        * còn lại: cao

    * EnvConfig: Class cấu hình cho env
        * delay_threshold_ms: Ngưỡng delay chấp nhận được
        * default_slice_type: slice mặc định (khi data chưa phân loại đc slice )
        * default_slice_type: traffic mặc định (khi data chưa phân loại được traffic)

* parser.py: đọc trace ns3 và biến đổi thành object python để dùng
    * parser config 
    * parser topology 
    * paser timestep 
    * tạo TraceBundle

* state_builder.py: Dữ liệu từ object python được đưa vào thành các dataclass sau đó được dùng để tạo state có dạng Dict (full state)
    * state["ues"]: trong đó mỗi UE chứa các UE metrics
    * state["rus"]: mỗi RU lại chứa các RU metrics
    * state["t"]

* ppo/obs_adapter.py: Chuyển state dict thành tensor obs
    * lấy state ban đầu
    * Biến đổi mỗi  UE thành vector feature
    * Biến đổi mỗi Cell thành vector feature
    * Pading đối với các dữ liệu thay đổi: N_max, M_max
    * tạo ue_mask, cell_mask cho các dữ liệu thay đổi trên 
* ppo/normalizer.py: Chuẩn hóa feature, reward 
* ppo/encoder.py: Dùng MLPBlock để encode dữ liệu cell feature, ue features
* ppo/actor-critic.py: 
    * actor: Mỗi UE được chấm điểm từng cell ứng viên bằng một actor_pair 
    * critic: nhận global_latent và dự đoán hàm giá trị 
* ppo/buffer: Nói dễ hiểu là dùng một bộ đệm để lưu các giá trị đã học, để tránh học cái mới nhưng quên cái cũ, nhưng cũng k lưu quá nhiều, và được uppdate sau một thời gian nhất định (Dùng GAE để update)



