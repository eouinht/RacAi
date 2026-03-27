from SimulationConfig import create_default_config, set_random_seed, get_slice_params
from TopologyBuilder import build_topology
from UEPositionGenerator import init_ue_state
from RadioSignalEstimator import estimate_radio_state
from TrafficQueueManager import estimate_traffic_queue_state
from ResourceStateManager import init_resource_state, update_ru_usage, release_unused_prb,request_prb_for_ru, redistribute_prb_within_ru
import numpy as np 

cfg = create_default_config()
set_random_seed(cfg)

topo = build_topology(
    n_ru=cfg.n_ru,
    n_du=cfg.n_du,
    n_cu=cfg.n_cu,
    ru_prb_cap=cfg.ru_prb_cap,
    du_cpu_cap=cfg.du_cpu_cap,
    cu_cpu_cap=cfg.cu_cpu_cap,
)

# UE state
ue_pos, ue_vel, ue_slice = init_ue_state(
    n_ue=cfg.n_ue,
    speed_mean=cfg.ue_speed_mean,
    speed_std=cfg.ue_speed_std,
    area_size=500.0,
    embb_ratio=0.7,
)

# Radio State 
radio_state = estimate_radio_state(
    ue_pos=ue_pos,
    ru_pos=topo["ru_pos"],
    carrier_freq_ghz=cfg.carrier_freq_ghz,
    bandwidth_mhz=cfg.bandwidth_mhz,
    noise_figure_db=cfg.noise_figure_db,
    ru_tx_power_dbm=cfg.ru_tx_power_dbm,
)

serving_ru=radio_state["serving_ru"]

resource_state = init_resource_state(
        serving_ru=serving_ru,
        prb_total=cfg.prb_total,
        ru_prb_cap=cfg.ru_prb_cap,
        n_ru=cfg.n_ru,
    )


print("=== INIT RESOURCE STATE ===")
print("UE count per RU:", resource_state["ue_count_per_ru"])
print("RU PRB allocated:", resource_state["ru_prb_allocated"])
print("RU used PRB:", resource_state["ru_used_prb"])
print("RU free PRB:", resource_state["ru_available_prb"])
print("UE allocated PRB first 10:", resource_state["ue_allocated_prb"][:10])
print("PRB pool free:", resource_state["prb_pool_free"])
print("Sum allocated PRB:", np.sum(resource_state["ru_prb_allocated"]))
print("Sum used PRB:", np.sum(resource_state["ru_used_prb"]))
print()
ru_used_prb_recomputed = update_ru_usage(
        serving_ru=serving_ru,
        ue_allocated_prb=resource_state["ue_allocated_prb"],
        n_ru=cfg.n_ru,
    )

print("=== RECOMPUTED RU USAGE ===")
print("RU used PRB recomputed:", ru_used_prb_recomputed)
print(
    "Usage matches init:",
    np.allclose(ru_used_prb_recomputed, resource_state["ru_used_prb"])
)
print()

# 7. release unused PRB
release_out = release_unused_prb(
    ru_prb_allocated=resource_state["ru_prb_allocated"],
    ru_used_prb=ru_used_prb_recomputed,
)

print("=== RELEASE UNUSED PRB ===")
print("New RU allocated PRB:", release_out["ru_prb_allocated"])
print("Released PRB:", release_out["released_prb"])
print()

# 8. request extra PRB for one RU (example RU 0)
request_out = request_prb_for_ru(
    ru_id=0,
    required_prb=5.0,
    ru_prb_allocated=release_out["ru_prb_allocated"].copy(),
    ru_prb_cap=cfg.ru_prb_cap,
    prb_pool_free=resource_state["prb_pool_free"] + release_out["released_prb"],
)

print("=== REQUEST EXTRA PRB FOR RU 0 ===")
print("Success:", request_out["success"])
print("Updated RU allocated PRB:", request_out["ru_prb_allocated"])
print("Updated PRB pool free:", request_out["prb_pool_free"])
print()

# 9. redistribute equal-share inside RU after updated allocation
ue_allocated_prb_new = redistribute_prb_within_ru(
    serving_ru=serving_ru,
    ru_prb_allocated=request_out["ru_prb_allocated"],
    n_ru=cfg.n_ru,
)

print("=== REDISTRIBUTE PRB WITHIN RU ===")
print("UE allocated PRB new first 10:", ue_allocated_prb_new[:10])

ru_used_after_redistribute = update_ru_usage(
    serving_ru=serving_ru,
    ue_allocated_prb=ue_allocated_prb_new,
    n_ru=cfg.n_ru,
)

print("RU used after redistribute:", ru_used_after_redistribute)
print(
    "Used <= allocated for all RU:",
    np.all(ru_used_after_redistribute <= request_out["ru_prb_allocated"] + 1e-9)
)

r_min_bps, sinr_min_db, delay_max_s, eta, lambda_arrival_bps = get_slice_params(
    cfg=cfg,
    ue_slice=ue_slice,
)

queue_bits = np.zeros(cfg.n_ue, dtype=np.float64)

traffic_state = estimate_traffic_queue_state(
    serving_ru=radio_state["serving_ru"],
    serving_sinr_db=radio_state["serving_sinr_db"],
    queue_bits=queue_bits,
    eta=eta,
    lambda_arrival_bps=lambda_arrival_bps,
    r_min_bps=r_min_bps,
    delay_max_s=delay_max_s,
    total_bandwidth_hz=cfg.bandwidth_mhz * 1e6,
    n_ru=cfg.n_ru,
    dt=cfg.time_step_s,
)

print("Distance matrix:", radio_state["distance_m"].shape)
print("RSRP matrix:", radio_state["rsrp_dbm"].shape)
print("SINR matrix:", radio_state["sinr_db"].shape)
print("Serving RU:", radio_state["serving_ru"])
# serving_ru = radio_state["serving_ru"]

# # for ue_id, ru_id in enumerate(serving_ru):
#     print(f"UE {ue_id:02d} -> RU {ru_id}")
print("Serving SINR:", radio_state["serving_sinr_db"][:5])
print("Best neighbor RU:", radio_state["best_neighbor_ru"][:5])

print("Throughput shape:", traffic_state["throughput_bps"].shape)
print("Queue next shape:", traffic_state["queue_bits_next"].shape)
print("Delay next shape:", traffic_state["delay_s_next"].shape)
print("QoS violations:", traffic_state["qos_violation"][:10])
print("UE count per RU:", traffic_state["ue_count_per_ru"])
print("Throughput first 5:", traffic_state["throughput_bps"][:5])

print("Serving SINR first 10:", radio_state["serving_sinr_db"][:10])
print("Slice first 10:", ue_slice[:10])