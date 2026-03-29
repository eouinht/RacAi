import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation.SimulationConfig import create_default_config, set_random_seed, get_slice_params
from simulation.TopologyBuilder import build_topology
from simulation.UEPositionGenerator import init_ue_state, update_ue_positions
from simulation.RadioSignalEstimator import estimate_radio_state
from simulation.ResourceStateManager import init_resource_state, estimate_cpu_requirements, estimate_required_prb, compact_stable_ue_allocation, compute_ru_free_prb, release_unused_prb
from simulation.TrafficQueueManager import estimate_traffic_state, check_qos_violation
from simulation.LatencyModel import estimate_latency_state
from simulation.HandoverCandidateFilter import classify_stable_and_candidate_ue
from simulation.CandidateActionAllocator import process_candidate_ues


class HandoverEnv(gym.Env):
    """
    Gymnasium environment for handover optimization.

    Design choices:
    - One environment step = one simulator timestep.
    - The environment first classifies UEs into stable/candidate.
    - The agent acts only on candidate UEs.
    - Action format: one target RU per candidate slot.
    - Candidate slots are padded to max_candidate_ue.
    - Stable UEs are handled by the simulator/resource logic.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg=None,
        area_size: float = 500.0,
        embb_ratio: float = 0.7,
        max_candidate_ue: int = 5,
        max_steps: int | None = None,
    ):
        super().__init__()

        self.cfg = create_default_config() if cfg is None else cfg
        self.area_size = area_size
        self.embb_ratio = embb_ratio
        self.max_candidate_ue = max_candidate_ue
        self.max_steps = self.cfg.n_steps if max_steps is None else max_steps

        # Observation layout per candidate slot:
        # [serving_ru, best_neighbor_ru, serving_gain, best_neighbor_gain,
        #  current_prb, required_prb, throughput, latency, r_min, delay_max,
        #  slice_id, qos_violation]
        self.candidate_feature_dim = 12

        # Global features:
        # [prb_pool_free,
        #  ru_used_prb(M), ru_allocated_prb(M), ru_free_prb(M), ue_count_per_ru(M)]
        self.global_feature_dim = 1 + 4 * self.cfg.n_ru

        obs_dim = self.max_candidate_ue * self.candidate_feature_dim + self.global_feature_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Mỗi ứng viên UE chọn 1 RU 
        # One discrete RU target / candidate slot.
        # If a slot is inactive, the action at that slot is ignored.
        self.action_space = spaces.MultiDiscrete([self.cfg.n_ru] * self.max_candidate_ue)

        self.topology = None
        self.ue_pos = None
        self.ue_vel = None
        self.ue_slice = None
        self.queue_bits = None
        self.resource_state = None

        self.current_step = 0
        self.last_state = None
        self.last_candidate_indices = None
        self.last_candidate_mask = None
        self.last_obs = None

    # ------------------------------------------------------------------
    # Reset / initialization
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.cfg = type(self.cfg)(**{**self.cfg.__dict__, "seed": int(seed)})
        set_random_seed(self.cfg)

        self.topology = build_topology(
            n_ru=self.cfg.n_ru,
            n_du=self.cfg.n_du,
            n_cu=self.cfg.n_cu,
            ru_prb_cap=self.cfg.ru_prb_cap,
            du_cpu_cap=self.cfg.du_cpu_cap,
            cu_cpu_cap=self.cfg.cu_cpu_cap,
        )

        self.ue_pos, self.ue_vel, self.ue_slice = init_ue_state(
            n_ue=self.cfg.n_ue,
            speed_mean=self.cfg.ue_speed_mean,
            speed_std=self.cfg.ue_speed_std,
            area_size=self.area_size,
            embb_ratio=self.embb_ratio,
        )

        radio_state = estimate_radio_state(
            ue_pos=self.ue_pos,
            ru_pos=self.topology["ru_pos"],
            carrier_freq_ghz=self.cfg.carrier_freq_ghz,
            rb_bandwidth_hz=self.cfg.rb_bandwidth_hz,
            noise_figure_db=self.cfg.noise_figure_db,
            ru_tx_power_dbm=self.cfg.ru_tx_power_dbm,
        )

        total_tx_power_w = 10.0 ** ((self.cfg.ru_tx_power_dbm - 30.0) / 10.0)
        
        self.resource_state = init_resource_state(
            serving_ru=radio_state["serving_ru"],
            prb_total=self.cfg.prb_total,
            ru_prb_cap=self.cfg.ru_prb_cap,
            n_ru=self.cfg.n_ru,
            total_tx_power_w=total_tx_power_w,
        )

        self.queue_bits = np.zeros(self.cfg.n_ue, dtype=np.float64)
        self.current_step = 0

        #  Buil state đầu tiên
        self.last_state = self._build_simulation_state(use_action=False, action_targets=None)
        #  Chuyển thành obs
        obs = self._build_observation(self.last_state)
        self.last_obs = obs

        info = self._build_info(self.last_state)
        
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        #  Nhận action 
        action = np.asarray(action, dtype=np.int32)
        if action.shape != (self.max_candidate_ue,):
            raise ValueError(f"Expected action shape {(self.max_candidate_ue,)}, got {action.shape}")

        self.current_step += 1

        # Buil state mới dựa trên action vừa nhận 
        state = self._build_simulation_state(use_action=True, action_targets=action)
        self.last_state = state

        # Commit next state
        self.ue_pos = state["ue_pos"]
        self.ue_vel = state["ue_vel"]
        self.queue_bits = state["queue_bits"]
        self.resource_state = state["resource_state"]

        # Build obs mới 
        obs = self._build_observation(state)
        reward = self._compute_reward(state)
        terminated = False
        truncated = self.current_step >= self.max_steps
        info = self._build_info(state)

        self.last_obs = obs
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Core simulator integration
    # ------------------------------------------------------------------
    def _build_simulation_state(
        self,
        use_action: bool,
        action_targets: np.ndarray | None):
        
        # 1) Mobility update
        ue_pos_next, ue_vel_next = update_ue_positions(
            ue_pos=self.ue_pos,
            ue_vel=self.ue_vel,
            dt=self.cfg.time_step_s,
            area_size=self.area_size,
        )

        # 2) Radio state
        radio_state = estimate_radio_state(
            ue_pos=ue_pos_next,
            ru_pos=self.topology["ru_pos"],
            carrier_freq_ghz=self.cfg.carrier_freq_ghz,
            rb_bandwidth_hz=self.cfg.rb_bandwidth_hz,
            noise_figure_db=self.cfg.noise_figure_db,
            ru_tx_power_dbm=self.cfg.ru_tx_power_dbm,
        )

        serving_ru = radio_state["serving_ru"].copy()
        ue_idx = np.arange(self.cfg.n_ue)

        # 3) Slice params
        r_min_bps, _, delay_max_s, eta, lambda_arrival_bps = get_slice_params(
            cfg=self.cfg,
            ue_slice=self.ue_slice,
        )
        packet_size_bits = np.where(self.ue_slice == 0, 12000.0, 4000.0).astype(np.float64)

        # 4) Traffic state using current allocation
        traffic_state = estimate_traffic_state(
            serving_gain=radio_state["serving_gain"],
            ue_power_alloc_w=self.resource_state["ue_power_alloc_w"],
            ue_allocated_prb=self.resource_state["ue_allocated_prb"],
            rb_bandwidth_hz=self.cfg.rb_bandwidth_hz,
            queue_bits=self.queue_bits,
            lambda_arrival_bps=lambda_arrival_bps,
            dt=self.cfg.time_step_s,
        )
        throughput_bps = traffic_state["throughput_bps"]

        # 5) CPU requirements and latency
        cpu_state = estimate_cpu_requirements(
            r_min_bps=r_min_bps,
            eta=eta,
            k_du=self.cfg.k_du,
            k_cu=self.cfg.k_cu,
        )

        serving_du = self.topology["ru_to_du"][serving_ru]
        serving_cu = self.topology["du_to_cu"][serving_du]

        du_cpu_capacity = self.topology["du_cpu_cap"][serving_du]
        cu_cpu_capacity = self.topology["cu_cpu_cap"][serving_cu]

        # Aggregate used CPU from current serving path
        du_cpu_used = np.zeros(self.cfg.n_du, dtype=np.float64)
        cu_cpu_used = np.zeros(self.cfg.n_cu, dtype=np.float64)
        for i in range(self.cfg.n_ue):
            du_cpu_used[serving_du[i]] += cpu_state["du_cpu_required"][i]
            cu_cpu_used[serving_cu[i]] += cpu_state["cu_cpu_required"][i]

        arrival_rate_packets_per_s = lambda_arrival_bps / np.maximum(packet_size_bits, 1e-9)

        # Load-aware service rates
        du_service_rate_packets_per_s = np.maximum(
            1e-6,
            (du_cpu_capacity - du_cpu_used[serving_du]) / np.maximum(cpu_state["du_cpu_required"], 1e-9),
        )
        cu_service_rate_packets_per_s = np.maximum(
            1e-6,
            (cu_cpu_capacity - cu_cpu_used[serving_cu]) / np.maximum(cpu_state["cu_cpu_required"], 1e-9),
        )

        serving_distance_m = radio_state["distance_m"][ue_idx, serving_ru]
        latency_state = estimate_latency_state(
            serving_distance_m=serving_distance_m,
            packet_size_bits=packet_size_bits,
            throughput_bps=throughput_bps,
            arrival_rate_packets_per_s=arrival_rate_packets_per_s,
            du_cpu_required=cpu_state["du_cpu_required"],
            du_cpu_capacity=du_cpu_capacity,
            cu_cpu_required=cpu_state["cu_cpu_required"],
            cu_cpu_capacity=cu_cpu_capacity,
            du_service_rate_packets_per_s=du_service_rate_packets_per_s,
            cu_service_rate_packets_per_s=cu_service_rate_packets_per_s,
        )
        total_latency_s = latency_state["total_latency_s"]

        # 6) Candidate filtering
        

        ue_required_prb = estimate_required_prb(
            r_min_bps=r_min_bps,
            serving_gain=radio_state["serving_gain"],
            ue_power_alloc_w=self.resource_state["ue_power_alloc_w"],
            rb_bandwidth_hz=self.cfg.rb_bandwidth_hz,
        )

        filter_state = classify_stable_and_candidate_ue(
            serving_gain=radio_state["serving_gain"],
            best_neighbor_gain=radio_state["best_neighbor_gain"],
            throughput_bps=throughput_bps,
            total_latency_s=total_latency_s,
            r_min_bps=r_min_bps,
            delay_max_s=delay_max_s,
            ue_required_prb=ue_required_prb,
            current_allocated_prb=self.resource_state["ue_allocated_prb"],
        )
        candidate_mask = filter_state["candidate_mask"]

        # Stable UE compaction first


        stable_margin_ratio = np.where(self.ue_slice == 0, 0.05, 0.15).astype(np.float64)
        compact_state = compact_stable_ue_allocation(
            serving_ru=serving_ru,
            stable_mask=filter_state["stable_mask"],
            ue_allocated_prb=self.resource_state["ue_allocated_prb"],
            ue_required_prb=ue_required_prb,
            stable_margin_ratio=stable_margin_ratio,
            n_ru=self.cfg.n_ru,
        )

        resource_state = {
            **self.resource_state,
            "ue_allocated_prb": compact_state["ue_allocated_prb"],
            "ru_used_prb": compact_state["ru_used_prb"],
        }
        resource_state["ru_free_prb"] = compute_ru_free_prb(
            ru_prb_allocated=resource_state["ru_prb_allocated"],
            ru_used_prb=resource_state["ru_used_prb"],
        )
        release_state = release_unused_prb(
            ru_prb_allocated=resource_state["ru_prb_allocated"],
            ru_used_prb=resource_state["ru_used_prb"],
            prb_pool_free=resource_state["prb_pool_free"],
        )
        resource_state["ru_prb_allocated"] = release_state["ru_prb_allocated"]
        resource_state["prb_pool_free"] = release_state["prb_pool_free"]
        resource_state["ru_free_prb"] = resource_state["ru_prb_allocated"] - resource_state["ru_used_prb"]

        # 7) Candidate action application
        self.last_candidate_indices = np.where(candidate_mask)[0][: self.max_candidate_ue]
        self.last_candidate_mask = candidate_mask.copy()

        if use_action and self.last_candidate_indices.size > 0:
            best_neighbor_ru = radio_state["best_neighbor_ru"].copy()
            for slot_idx, ue_id in enumerate(self.last_candidate_indices):
                best_neighbor_ru[ue_id] = int(action_targets[slot_idx])
            radio_state_for_action = {**radio_state, "best_neighbor_ru": best_neighbor_ru}
        else:
            radio_state_for_action = radio_state

        action_out = process_candidate_ues(
            candidate_mask=candidate_mask,
            radio_state=radio_state_for_action,
            topology=self.topology,
            resource_state=resource_state,
            r_min_bps=r_min_bps,
            delay_max_s=delay_max_s,
            packet_size_bits=packet_size_bits,
            lambda_arrival_bps=lambda_arrival_bps,
            du_cpu_required=cpu_state["du_cpu_required"],
            cu_cpu_required=cpu_state["cu_cpu_required"],
            du_cpu_used=du_cpu_used,
            cu_cpu_used=cu_cpu_used,
            rb_bandwidth_hz=self.cfg.rb_bandwidth_hz,
            ru_prb_cap=self.cfg.ru_prb_cap,
        )

        serving_ru_after = action_out["serving_ru"]
        resource_state_after = action_out["resource_state"]
        du_cpu_used_after = action_out["du_cpu_used"]
        cu_cpu_used_after = action_out["cu_cpu_used"]

        # 8) Recompute final traffic/latency after HO
        serving_gain_after = radio_state["gain"][ue_idx, serving_ru_after]
        serving_distance_after = radio_state["distance_m"][ue_idx, serving_ru_after]

        traffic_state_after = estimate_traffic_state(
            serving_gain=serving_gain_after,
            ue_power_alloc_w=resource_state_after["ue_power_alloc_w"],
            ue_allocated_prb=resource_state_after["ue_allocated_prb"],
            rb_bandwidth_hz=self.cfg.rb_bandwidth_hz,
            queue_bits=self.queue_bits,
            lambda_arrival_bps=lambda_arrival_bps,
            dt=self.cfg.time_step_s,
        )
        queue_bits_next = traffic_state_after["queue_bits_next"]
        throughput_after = traffic_state_after["throughput_bps"]

        serving_du_after = self.topology["ru_to_du"][serving_ru_after]
        serving_cu_after = self.topology["du_to_cu"][serving_du_after]
        du_cpu_capacity_after = self.topology["du_cpu_cap"][serving_du_after]
        cu_cpu_capacity_after = self.topology["cu_cpu_cap"][serving_cu_after]

        du_service_rate_after = np.maximum(
            1e-6,
            (du_cpu_capacity_after - du_cpu_used_after[serving_du_after]) / np.maximum(cpu_state["du_cpu_required"], 1e-9),
        )
        cu_service_rate_after = np.maximum(
            1e-6,
            (cu_cpu_capacity_after - cu_cpu_used_after[serving_cu_after]) / np.maximum(cpu_state["cu_cpu_required"], 1e-9),
        )

        latency_state_after = estimate_latency_state(
            serving_distance_m=serving_distance_after,
            packet_size_bits=packet_size_bits,
            throughput_bps=throughput_after,
            arrival_rate_packets_per_s=arrival_rate_packets_per_s,
            du_cpu_required=cpu_state["du_cpu_required"],
            du_cpu_capacity=du_cpu_capacity_after,
            cu_cpu_required=cpu_state["cu_cpu_required"],
            cu_cpu_capacity=cu_cpu_capacity_after,
            du_service_rate_packets_per_s=du_service_rate_after,
            cu_service_rate_packets_per_s=cu_service_rate_after,
        )

        qos_violation = check_qos_violation(
            throughput_bps=throughput_after,
            total_latency_s=latency_state_after["total_latency_s"],
            r_min_bps=r_min_bps,
            delay_max_s=delay_max_s,
        )

        return {
            "ue_pos": ue_pos_next,
            "ue_vel": ue_vel_next,
            "ue_slice": self.ue_slice,
            "queue_bits": queue_bits_next,
            "radio_state": {**radio_state, "serving_ru": serving_ru_after},
            "traffic_state": traffic_state_after,
            "latency_state": latency_state_after,
            "filter_state": filter_state,
            "resource_state": resource_state_after,
            "serving_ru": serving_ru_after,
            "candidate_indices": self.last_candidate_indices,
            "candidate_mask": candidate_mask,
            "qos_violation": qos_violation,
            "ue_required_prb": ue_required_prb,
            "r_min_bps": r_min_bps,
            "delay_max_s": delay_max_s,
            "du_cpu_used": du_cpu_used_after,
            "cu_cpu_used": cu_cpu_used_after,
            "arrival_rate_packets_per_s": arrival_rate_packets_per_s,
            "packet_size_bits": packet_size_bits,
        }

    # ------------------------------------------------------------------
    # Observation / reward / info
    # ------------------------------------------------------------------
    def _build_observation(self, state: dict) -> np.ndarray:
        features = np.zeros((self.max_candidate_ue, self.candidate_feature_dim), dtype=np.float32)
        candidate_indices = state["candidate_indices"]

        radio_state = state["radio_state"]
        traffic_state = state["traffic_state"]
        latency_state = state["latency_state"]
        ue_required_prb = state["ue_required_prb"]
        r_min_bps = state["r_min_bps"]
        delay_max_s = state["delay_max_s"]
        qos_violation = state["qos_violation"]
        ue_slice = state["ue_slice"]
        resource_state = state["resource_state"]

        for slot, ue_id in enumerate(candidate_indices):
            serving_ru = state["serving_ru"][ue_id]
            best_neighbor_ru = radio_state["best_neighbor_ru"][ue_id]
            features[slot] = np.array([
                float(serving_ru),
                float(best_neighbor_ru),
                float(radio_state["gain"][ue_id, serving_ru]),
                float(radio_state["best_neighbor_gain"][ue_id]),
                float(resource_state["ue_allocated_prb"][ue_id]),
                float(ue_required_prb[ue_id]),
                float(traffic_state["throughput_bps"][ue_id]),
                float(latency_state["total_latency_s"][ue_id]),
                float(r_min_bps[ue_id]),
                float(delay_max_s[ue_id]),
                float(ue_slice[ue_id]),
                float(qos_violation[ue_id]),
            ], dtype=np.float32)

        global_features = np.concatenate([
            np.array([resource_state["prb_pool_free"]], dtype=np.float32),
            resource_state["ru_used_prb"].astype(np.float32),
            resource_state["ru_prb_allocated"].astype(np.float32),
            resource_state["ru_free_prb"].astype(np.float32),
            resource_state["ue_count_per_ru"].astype(np.float32),
        ])

        obs = np.concatenate([features.reshape(-1), global_features]).astype(np.float32)
        return obs

    def _compute_reward(self, state: dict) -> float:
        throughput = state["traffic_state"]["throughput_bps"]
        latency = state["latency_state"]["total_latency_s"]
        qos_violation = state["qos_violation"]
        candidate_count = int(state["candidate_mask"].sum())

        reward = 0.0
        reward += float(np.mean(throughput) / 1e6)
        reward -= 10.0 * float(np.mean(latency))
        reward -= 2.0 * float(np.mean(qos_violation.astype(np.float64)))
        reward -= 0.05 * candidate_count
        return float(reward)

    def _build_info(self, state: dict) -> dict:
        return {
            "step": self.current_step,
            "candidate_count": int(state["candidate_mask"].sum()),
            "qos_violation_count": int(state["qos_violation"].sum()),
            "mean_throughput_bps": float(np.mean(state["traffic_state"]["throughput_bps"])),
            "mean_latency_s": float(np.mean(state["latency_state"]["total_latency_s"])),
            "prb_pool_free": float(state["resource_state"]["prb_pool_free"]),
        }

    # ------------------------------------------------------------------
    # Optional render
    # ------------------------------------------------------------------
    def render(self):
        return None

    def close(self):
        return None