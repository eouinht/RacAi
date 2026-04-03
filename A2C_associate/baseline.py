# baseline.py
import os
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# ======================================================
# ===================== Helpers ========================
# ======================================================
def _active_ues(state):
    """Danh sách UE còn active lấy từ state['UE_requests'] (dict có khóa 'active' và 'id')."""
    return [ue for ue in state["UE_requests"] if int(ue.get("active", 0)) == 1]


def _categorize_ues(state):
    """Nhóm UE thành stable/ho/new theo trạng thái đã có trong state.K"""
    stable, ho, new = [], [], []
    for ue in state["UE_requests"]:
        if int(ue.get("active", 0)) != 1:
            continue
        if int(ue.get("is_ho_candidate", 0)) == 1:
            ho.append(ue)
        elif int(ue.get("is_new", 0)) == 1:
            new.append(ue)
        else:
            stable.append(ue)
    return stable, ho, new


# ======================================================
# =================== Baseline Agents ==================
# ======================================================
import numpy as np

class RandomRUAgent:
    """
    Pure Random baseline:
      - UE random
      - accept/reject random
      - RU, DU, CU random toàn bộ không gian
      - RB random [1, max_RBs_per_UE]
      - Power random trong codebook
    Mọi kiểm tra hợp lệ để môi trường xử lý.
    """

    def __init__(self, num_RUs, num_DUs, num_CUs, max_RBs_per_UE, power_levels, seed=42):
        self.num_RUs = num_RUs
        self.num_DUs = num_DUs
        self.num_CUs = num_CUs
        self.max_RBs_per_UE = int(max_RBs_per_UE)
        self.power_levels = np.array(power_levels, dtype=float)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, ue_pool=None):
        if ue_pool is None:
            ue_pool = _active_ues(state)
        if not ue_pool:
            return None, None, None

        # -------- UE random (pool) --------
        ue = self.rng.choice(ue_pool)
        ue_id = int(ue["id"])

        # -------- Accept / Reject random --------
        accept_flag = int(self.rng.integers(0, 2))  # 0 or 1

        # -------- RU / DU / CU random toàn bộ --------
        ru = int(self.rng.integers(0, self.num_RUs))
        du = int(self.rng.integers(0, self.num_DUs))
        cu = int(self.rng.integers(0, self.num_CUs))

        # -------- RB random --------
        n_rb = int(self.rng.integers(1, self.max_RBs_per_UE + 1))

        # -------- Power random --------
        power = float(self.rng.choice(self.power_levels))

        return (ue_id, accept_flag, ru, du, cu, n_rb, power), 0, 0




class NearestRUAgent:
    """
    Nearest-RU baseline:
      - UE chọn RU gần nhất còn hợp lệ
      - DU random trong các DU nối được RU
      - CU random trong các CU nối được DU
      - RB random
      - Power random trong codebook hợp lệ
    """

    def __init__(self, num_RUs, num_DUs, num_CUs, max_RBs_per_UE, power_levels, distances_RU_UE, seed=42):
        self.num_RUs = num_RUs
        self.num_DUs = num_DUs
        self.num_CUs = num_CUs
        self.max_RBs_per_UE = int(max_RBs_per_UE)
        self.power_levels = np.array(power_levels, dtype=float)
        self.distances_RU_UE = np.array(distances_RU_UE, dtype=float)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, ue_pool=None):
        if ue_pool is None:
            ue_pool = _active_ues(state)
        if not ue_pool:
            return None, None, None

        # UE select from pool
        ue = self.rng.choice(ue_pool)
        ue_id = int(ue["id"])

        if int(state["RB_remaining"]) <= 0:
            return (ue_id, 0, 0, 0, 0, 1, 0.0), 0, 0

        # -------- RU gần nhất còn hợp lệ --------
        dists = self.distances_RU_UE[:, ue_id]
        ru_order = np.argsort(dists)

        ru = None
        for r in ru_order:
            if state["RU_power_remaining"][r] > 0 and np.any(state["l_ru_du"][r] == 1):
                ru = int(r)
                break

        if ru is None:
            return (ue_id, 0, 0, 0, 0, 1, 0.0), 0, 0

        # -------- DU random --------
        valid_dus = [
            d for d in range(self.num_DUs)
            if state["l_ru_du"][ru, d] == 1 and state["DU_remaining"][d] > 0
        ]
        if not valid_dus:
            return (ue_id, 0, ru, 0, 0, 1, 0.0), 0, 0
        du = int(self.rng.choice(valid_dus))

        # -------- CU random --------
        valid_cus = [
            c for c in range(self.num_CUs)
            if state["l_du_cu"][du, c] == 1 and state["CU_remaining"][c] > 0
        ]
        if not valid_cus:
            return (ue_id, 0, ru, du, 0, 1, 0.0), 0, 0
        cu = int(self.rng.choice(valid_cus))

        # -------- RB random --------
        max_rb = min(int(state["RB_remaining"]), self.max_RBs_per_UE)
        n_rb = int(self.rng.integers(1, max_rb + 1))

        # -------- Power random --------
        ru_budget = float(state["RU_power_remaining"][ru])
        valid_powers = self.power_levels[self.power_levels <= ru_budget]
        if len(valid_powers) == 0:
            return (ue_id, 0, ru, du, cu, n_rb, 0.0), 0, 0

        power = float(self.rng.choice(valid_powers))

        return (ue_id, 1, ru, du, cu, n_rb, power), 0, 0




class RoundRobinAgent:
    """
    Round-Robin baseline:
      - UE được phục vụ theo vòng tròn
      - Với mỗi UE, RU được chọn theo vòng tròn riêng (per-UE pointer)
      - DU: DU hợp lệ đầu tiên
      - CU: CU hợp lệ đầu tiên
      - RB: dùng tối đa có thể
      - Power: mức nhỏ nhất hợp lệ trong codebook
    """

    def __init__(self, num_RUs, num_DUs, num_CUs, max_RBs_per_UE, power_levels):
        self.num_RUs = num_RUs
        self.num_DUs = num_DUs
        self.num_CUs = num_CUs
        self.max_RBs_per_UE = int(max_RBs_per_UE)
        self.power_levels = np.sort(np.array(power_levels, dtype=float))

        self._ue_ptr = 0
        self._ru_ptr = defaultdict(int)  # ue_id -> next RU index

    def select_action(self, state, ue_pool=None):
        if ue_pool is None:
            ue_pool = _active_ues(state)
        if not ue_pool:
            return None, None, None

        # -------- UE Round Robin --------
        ue = ue_pool[self._ue_ptr % len(ue_pool)]
        ue_id = int(ue["id"])
        self._ue_ptr = (self._ue_ptr + 1) % max(1, len(ue_pool))

        if int(state["RB_remaining"]) <= 0:
            return (ue_id, 0, 0, 0, 0, 1, 0.0), 0, 0

        # -------- RU Round Robin per UE --------
        start = self._ru_ptr[ue_id]
        ru = None

        for i in range(self.num_RUs):
            cand = (start + i) % self.num_RUs
            if state["RU_power_remaining"][cand] > 0 and np.any(state["l_ru_du"][cand] == 1):
                ru = int(cand)
                self._ru_ptr[ue_id] = (cand + 1) % self.num_RUs
                break

        if ru is None:
            return (ue_id, 0, 0, 0, 0, 1, 0.0), 0, 0

        # -------- DU: first feasible --------
        du = None
        for d in range(self.num_DUs):
            if state["l_ru_du"][ru, d] == 1 and state["DU_remaining"][d] > 0:
                du = int(d)
                break
        if du is None:
            return (ue_id, 0, ru, 0, 0, 1, 0.0), 0, 0

        # -------- CU: first feasible --------
        cu = None
        for c in range(self.num_CUs):
            if state["l_du_cu"][du, c] == 1 and state["CU_remaining"][c] > 0:
                cu = int(c)
                break
        if cu is None:
            return (ue_id, 0, ru, du, 0, 1, 0.0), 0, 0

        # -------- RB: max possible --------
        n_rb = min(int(state["RB_remaining"]), self.max_RBs_per_UE)

        # -------- Power: smallest feasible --------
        ru_budget = float(state["RU_power_remaining"][ru])
        valid_power = self.power_levels[self.power_levels <= ru_budget]
        if len(valid_power) == 0:
            return (ue_id, 0, ru, du, cu, n_rb, 0.0), 0, 0

        power = float(valid_power[0])

        return (ue_id, 1, ru, du, cu, n_rb, power), 0, 0



# ======================================================
# =============== Evaluation & Runner ==================
# ======================================================
def evaluate_baseline(agent, env, episodes=20, render=False):
    """
    Trả về:
      mean_reward, mean_accept, mean_throughput_avg, mean_latency_avg,
      và các list theo episode để lưu file.
    Throughput/Latency ở đây là **trung bình trên các UE được nhận** trong mỗi episode.
    """
    reward_hist, accept_hist = [], []
    thr_hist, lat_hist, time_hist = [], [], []

    for ep in range(episodes):
        state = env.reset_env()
        done = False
        total_reward = 0.0
        total_accept = 0
        total_throughput = 0.0
        total_latency = 0.0

        t0 = time.time()
        while not done:
            stable, ho, new_ues = _categorize_ues(state)
            # Ưu tiên HO -> new -> stable
            if ho:
                pool = ho
            elif new_ues:
                pool = new_ues
            else:
                pool = stable

            action, _, _ = agent.select_action(state, ue_pool=pool)
            if action is None:
                break
            next_state, reward, done, info = env.step(action)
            if info.get("success", False):
                total_reward += float(reward)
                total_accept += 1
                total_throughput += float(info.get("throughput_bps", 0.0))
                total_latency += float(info.get("delay_s", 0.0))
            state = next_state
        elapsed = time.time() - t0

        # Trung bình theo UE được nhận
        avg_thr = (total_throughput / total_accept) if total_accept > 0 else 0.0
        avg_lat = (total_latency / total_accept) if total_accept > 0 else 0.0

        reward_hist.append(total_reward)
        accept_hist.append(total_accept)
        thr_hist.append(avg_thr)
        lat_hist.append(avg_lat)
        time_hist.append(elapsed)

        if render:
            print(f"[{agent.__class__.__name__}] Ep {ep+1}/{episodes}: "
                  f"accept={total_accept}, thr_avg={avg_thr:.2e}, lat_avg={avg_lat:.2e}, time={elapsed:.3f}s")

    mean_reward = float(np.mean(reward_hist)) if reward_hist else 0.0
    mean_accept = float(np.mean(accept_hist)) if accept_hist else 0.0
    mean_thr    = float(np.mean(thr_hist))    if thr_hist    else 0.0
    mean_lat    = float(np.mean(lat_hist))    if lat_hist    else 0.0

    return (mean_reward, mean_accept, mean_thr, mean_lat,
            reward_hist, accept_hist, thr_hist, lat_hist, time_hist)

def _save_baseline_results(base_dir, name,
                           reward_hist, accept_hist, thr_hist, lat_hist, time_hist,
                           mean_reward, mean_accept, mean_thr, mean_lat):
    out_dir = Path(base_dir) / "baseline_results" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _w(fname, arr):
        with open(out_dir / fname, "w") as f:
            for x in arr:
                f.write(f"{x}\n")

    _w(f"reward_hist_{name}.txt",     reward_hist)
    _w(f"accept_hist_{name}.txt",     accept_hist)
    _w(f"throughput_hist_{name}.txt", thr_hist)   # trung bình / UE nhận
    _w(f"latency_hist_{name}.txt",    lat_hist)   # trung bình / UE nhận (s)
    _w(f"time_hist_{name}.txt",       time_hist)

    with open(out_dir / f"summary_{name}.txt", "w") as fs:
        fs.write(
            "==== Baseline summary ====\n"
            f"mean_reward={mean_reward:.6f}\n"
            f"mean_accept={mean_accept:.6f}\n"
            f"mean_throughput_avg={mean_thr:.6f}\n"
            f"mean_latency_avg={mean_lat:.6e}\n"
        )

def run_all_baselines(env, results_dir, episodes=20, render=False):
    """
    Tạo và chạy 3 baseline: RoundRobin, NearestRU, RandomRU.
    Lưu kết quả vào: {results_dir}/baseline_results/<BaselineName>/
    """
    num_RUs, num_DUs, num_CUs = env.num_RUs, env.num_DUs, env.num_CUs
    max_RBs_per_UE = env.max_RBs_per_UE
    power_levels   = env.P_ib_sk_val
    distances      = getattr(env, "distances_RU_UE", None)
    if distances is None:
        # fallback nếu env không có ma trận khoảng cách
        distances = np.zeros((num_RUs, getattr(env, "num_UEs", 1)), dtype=float)

    agents = {
        "RoundRobin": RoundRobinAgent(num_RUs, num_DUs, num_CUs, max_RBs_per_UE, power_levels),
        "NearestRU":  NearestRUAgent(num_RUs, num_DUs, num_CUs, max_RBs_per_UE, power_levels, distances),
        "RandomRU":   RandomRUAgent(num_RUs, num_DUs, num_CUs, max_RBs_per_UE, power_levels),
    }

    print("🚦 Running baseline evaluations...")
    for name, agent in agents.items():
        print(f"  ▶ {name}")
        (mean_reward, mean_accept, mean_thr, mean_lat,
         r_hist, a_hist, tput_hist, lat_hist, time_hist) = evaluate_baseline(agent, env, episodes, render)

        _save_baseline_results(results_dir, name,
                               r_hist, a_hist, tput_hist, lat_hist, time_hist,
                               mean_reward, mean_accept, mean_thr, mean_lat)

        print(f"    ✓ {name}: "
              f"mean_reward={mean_reward:.3f}, "
              f"mean_accept={mean_accept:.2f}, "
              f"thr_avg={mean_thr:.2e} bps, "
              f"lat_avg={mean_lat:.2e} s")

    print(f"📂 Results saved under: {Path(results_dir) / 'baseline_results'}")
