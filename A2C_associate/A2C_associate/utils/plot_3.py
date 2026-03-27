import os
import numpy as np
import matplotlib.pyplot as plt

# =======================================================
# === HỖ TRỢ ĐỌC DỮ LIỆU & FONT ========================
# =======================================================
def is_number(s):
    try:
        float(s); return True
    except ValueError:
        return False

def read_series(filepath):
    """Đọc 1-d series float (bỏ dòng trống / comment)."""
    if not os.path.exists(filepath):
        print(f"⚠️ Không tìm thấy file: {filepath}")
        return np.array([], dtype=float)
    vals = []
    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals.append(float(s))
            except:
                pass
    return np.asarray(vals, dtype=np.float64)

def read_mean_from_file(filepath):
    arr = read_series(filepath)
    return float(arr.mean()) if arr.size else 0.0

def enable_latex_font():
    """Cấu hình font LaTeX (Computer Modern)."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}",
        "axes.labelsize": 28,
        "axes.titlesize": 28,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 26,
    })

# =======================================================
# === CẤU HÌNH CHUNG VẼ CỘT ============================
# =======================================================
ALGOS  = ["RandomRU", "RoundRobin", "NearestRU", "PPO-GraphSAGE",  "MILP"]
LABELS = ["Random-RU", "RoundRobin", "Nearest-RU", "PPO-GSA", "Optimal"]

# === IEEE-style pastel palette with strong highlights for PPO & MILP ===
COLORS = ['#C2C2C2', '#A6D854', '#80B1D3', '#D95F02', '#1B9E77']
HATCH  = ['/', '.', 'o', '//', '']

BAR_W, BAR_GAP = 40, 10

def _new_fig_ax(figsize=(10,6), dpi=300):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax

def _save(fig, out_dir, fname_base):
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{fname_base}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{fname_base}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

# =======================================================
# === 1) ACCEPTANCE RATE ================================
# =======================================================
def plot_accept_bar(run_path, ue_label="SingleRun", divide_by=None):
    enable_latex_font()
    out_dir = os.path.join(run_path, "compare_plots")

    paths = {
        "RandomRU":      os.path.join(run_path, "baseline_results", "RandomRU",   "accept_hist_RandomRU.txt"),
        "RoundRobin":    os.path.join(run_path, "baseline_results", "RoundRobin", "accept_hist_RoundRobin.txt"),
        "NearestRU":     os.path.join(run_path, "baseline_results", "NearestRU",  "accept_hist_NearestRU.txt"),
        "PPO-GraphSAGE": os.path.join(run_path, "evaluation_agent_PPO", "total_accept_PPO.txt"),
        "MILP":          os.path.join(run_path, "MILP_results", "Doraemon", "total_accept_MILP.txt"),
    }

    vals = []
    for a in ALGOS:
        v = read_mean_from_file(paths[a])
        if divide_by and divide_by > 0:
            v = v / divide_by
        vals.append(v)

    fig, ax = _new_fig_ax(figsize=(10,6))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Acceptance rate", fontsize=36)

    x, xt = 0, []
    for i, a in enumerate(ALGOS):
        ax.bar(x, vals[i], width=BAR_W, color='white',
               edgecolor=COLORS[i], hatch=HATCH[i], label=LABELS[i], zorder=3, linewidth=2.2)
        xt.append(x); x += BAR_W + BAR_GAP

    ax.set_xticks(xt); ax.set_xticklabels([])
    ax.grid(color='lightgrey', linestyle='--', zorder=0)
    ax.legend(
        loc="upper left",
        framealpha=0,
        fontsize=30,          # tăng cỡ chữ legend
        handlelength=1.6,     # chiều dài ký hiệu
        handletextpad=0.6,    # khoảng cách giữa ký hiệu và text
        labelspacing=0.3,     # khoảng cách giữa các dòng legend
        borderpad=0.3,        # padding bên trong khung legend
        ncol=1,         # số cột legend
        columnspacing=1.0,  # khoảng cách giữa các cột
        bbox_to_anchor=(0.0, 1.05),  # ↑ dịch legend lên cao hơn một chút
    )
    _save(fig, out_dir, f"accept_rate_{ue_label}")

# =======================================================
# === 2) EVALUATION TIME (log-scale, major grid) ========
# =======================================================
def plot_time_bar(run_path, ue_label="SingleRun"):
    enable_latex_font()
    out_dir = os.path.join(run_path, "compare_plots")

    paths = {
        "RandomRU":      os.path.join(run_path, "baseline_results", "RandomRU",   "time_hist_RandomRU.txt"),
        "RoundRobin":    os.path.join(run_path, "baseline_results", "RoundRobin", "time_hist_RoundRobin.txt"),
        "NearestRU":     os.path.join(run_path, "baseline_results", "NearestRU",  "time_hist_NearestRU.txt"),
        "PPO-GraphSAGE": os.path.join(run_path, "evaluation_agent_PPO", "evaluation_time_PPO.txt"),
        "MILP":          os.path.join(run_path, "MILP_results", "Doraemon", "evaluation_time_MILP.txt"),
    }

    vals = [max(read_mean_from_file(paths[a]), 1e-6) for a in ALGOS]

    fig, ax = _new_fig_ax(figsize=(10,9))
    ax.set_ylabel("Computing time (s)", fontsize=40)
    ax.set_yscale("log")

    ymin = min(vals) / 3.0
    ymax = max(vals) * 1.8
    ax.set_ylim(ymin, ymax)

    x, xt = 0, []
    for i, a in enumerate(ALGOS):
        ax.bar(x, vals[i], width=BAR_W, color='white',
               edgecolor=COLORS[i], hatch=HATCH[i], label=LABELS[i], zorder=3, linewidth=2.2)
        xt.append(x); x += BAR_W + BAR_GAP

    ax.set_xticks(xt); ax.set_xticklabels([])
    ax.grid(which='major', color='lightgrey', linestyle='--', alpha=0.85, zorder=0)
    ax.grid(which='minor', visible=False)
    ax.legend(
        loc="upper left",
        framealpha=0,
        fontsize=30,          # tăng cỡ chữ legend
        handlelength=1.6,     # chiều dài ký hiệu
        handletextpad=0.6,    # khoảng cách giữa ký hiệu và text
        labelspacing=0.3,     # khoảng cách giữa các dòng legend
        borderpad=0.3,        # padding bên trong khung legend
        ncol=1,         # số cột legend
        columnspacing=1.0,  # khoảng cách giữa các cột
    )
    _save(fig, out_dir, f"time_{ue_label}")

# =======================================================
# === 3) LATENCY (ms) ==================================
# =======================================================
def plot_latency_bar(run_path, ue_label="SingleRun"):
    enable_latex_font()
    out_dir = os.path.join(run_path, "compare_plots")

    paths = {
        "RandomRU":      os.path.join(run_path, "baseline_results", "RandomRU",   "latency_hist_RandomRU.txt"),
        "RoundRobin":    os.path.join(run_path, "baseline_results", "RoundRobin", "latency_hist_RoundRobin.txt"),
        "NearestRU":     os.path.join(run_path, "baseline_results", "NearestRU",  "latency_hist_NearestRU.txt"),
        "PPO-GraphSAGE": os.path.join(run_path, "evaluation_agent_PPO", "total_latency_PPO.txt"),
        "MILP":          os.path.join(run_path, "MILP_results", "Doraemon", "total_latency_MILP.txt"),
    }

    vals_ms = [read_mean_from_file(paths[a]) * 1e3 for a in ALGOS]

    fig, ax = _new_fig_ax(figsize=(10,6))
    ax.set_ylabel("Average latency (ms)", fontsize=36)
    ymax = max(vals_ms) if vals_ms else 1.0
    ax.set_ylim(0, ymax * 1.4)

    x, xt = 0, []
    for i, a in enumerate(ALGOS):
        ax.bar(x, vals_ms[i], width=BAR_W, color='white',
               edgecolor=COLORS[i], hatch=HATCH[i], label=LABELS[i], zorder=3, linewidth=2.2)
        xt.append(x); x += BAR_W + BAR_GAP

    ax.set_xticks(xt); ax.set_xticklabels([])
    ax.grid(color='lightgrey', linestyle='--', zorder=0)
    ax.legend(
        loc="upper left",
        framealpha=0,
        fontsize=30,          # tăng cỡ chữ legend
        handlelength=1.6,     # chiều dài ký hiệu
        handletextpad=0.6,    # khoảng cách giữa ký hiệu và text
        labelspacing=0.3,     # khoảng cách giữa các dòng legend
        borderpad=0.3,        # padding bên trong khung legend
        ncol=2,         # số cột legend
        columnspacing=1.0,  # khoảng cách giữa các cột
        bbox_to_anchor=(0.0, 1.05),  # ↑ dịch legend lên cao hơn một chút
    )
    _save(fig, out_dir, f"latency_{ue_label}")

# =======================================================
# === 4) REWARD =========================================
# =======================================================
def plot_reward_bar(run_path, ue_label="SingleRun"):
    enable_latex_font()
    out_dir = os.path.join(run_path, "compare_plots")

    paths = {
        "RandomRU":      os.path.join(run_path, "baseline_results", "RandomRU",   "reward_hist_RandomRU.txt"),
        "RoundRobin":    os.path.join(run_path, "baseline_results", "RoundRobin", "reward_hist_RoundRobin.txt"),
        "NearestRU":     os.path.join(run_path, "baseline_results", "NearestRU",  "reward_hist_NearestRU.txt"),
        "PPO-GraphSAGE": os.path.join(run_path, "evaluation_agent_PPO", "total_reward_PPO.txt"),
        "MILP":          os.path.join(run_path, "MILP_results", "Doraemon", "total_objective_MILP.txt"),
    }

    vals = [read_mean_from_file(paths[a]) for a in ALGOS]

    fig, ax = _new_fig_ax(figsize=(10,9))
    ax.set_ylabel("Average reward", fontsize=40)

    x, xt = 0, []
    for i, a in enumerate(ALGOS):
        ax.bar(x, vals[i], width=BAR_W, color='white',
               edgecolor=COLORS[i], hatch=HATCH[i], label=LABELS[i], zorder=3, linewidth=2.2)
        xt.append(x); x += BAR_W + BAR_GAP

    ax.set_xticks(xt); ax.set_xticklabels([])
    ax.grid(color='lightgrey', linestyle='--', zorder=0)
    ax.legend(
        loc="upper left",
        framealpha=0,
        fontsize=30,          # tăng cỡ chữ legend
        handlelength=1.6,     # chiều dài ký hiệu
        handletextpad=0.6,    # khoảng cách giữa ký hiệu và text
        labelspacing=0.3,     # khoảng cách giữa các dòng legend
        borderpad=0.3,        # padding bên trong khung legend
        ncol=1,         # số cột legend
        columnspacing=1.0,  # khoảng cách giữa các cột
    )
    _save(fig, out_dir, f"reward_{ue_label}")

# =======================================================
# === 5) THROUGHPUT (Mbps) ==============================
# =======================================================
def plot_throughput_bar(run_path, ue_label="SingleRun"):
    enable_latex_font()
    out_dir = os.path.join(run_path, "compare_plots")

    paths = {
        "RandomRU":      os.path.join(run_path, "baseline_results", "RandomRU",   "throughput_hist_RandomRU.txt"),
        "RoundRobin":    os.path.join(run_path, "baseline_results", "RoundRobin", "throughput_hist_RoundRobin.txt"),
        "NearestRU":     os.path.join(run_path, "baseline_results", "NearestRU",  "throughput_hist_NearestRU.txt"),
        "PPO-GraphSAGE": os.path.join(run_path, "evaluation_agent_PPO", "total_throughput_PPO.txt"),
        "MILP":          os.path.join(run_path, "MILP_results", "Doraemon", "total_throughput_MILP.txt"),
    }

    vals_mbps = [read_mean_from_file(paths[a]) / 1e6 for a in ALGOS]

    fig, ax = _new_fig_ax(figsize=(10,6))
    ax.set_ylabel("Average throughput (Mbps)", fontsize=36)
    ymax = max(vals_mbps) if vals_mbps else 1.0
    ax.set_ylim(0, ymax * 1.3)

    x, xt = 0, []
    for i, a in enumerate(ALGOS):
        ax.bar(x, vals_mbps[i], width=BAR_W, color='white',
               edgecolor=COLORS[i], hatch=HATCH[i], label=LABELS[i], zorder=3, linewidth=2.2)
        xt.append(x); x += BAR_W + BAR_GAP

    ax.set_xticks(xt); ax.set_xticklabels([])
    ax.grid(color='lightgrey', linestyle='--', zorder=0)
    ax.legend(
        loc="upper left",
        framealpha=0,
        fontsize=30,          # tăng cỡ chữ legend
        handlelength=1.6,     # chiều dài ký hiệu
        handletextpad=0.6,    # khoảng cách giữa ký hiệu và text
        labelspacing=0.3,     # khoảng cách giữa các dòng legend
        borderpad=0.3,        # padding bên trong khung legend
        ncol=2,         # số cột legend
        columnspacing=1.0,  # khoảng cách giữa các cột
        bbox_to_anchor=(0.0, 1.05),  # ↑ dịch legend lên cao hơn một chút
    )
    _save(fig, out_dir, f"throughput_{ue_label}")

# =======================================================
# === 6) PPO: Reward & Avg-Reward Curve (PPO/) ==========
# =======================================================
def plot_ppo_reward_curve(run_path, ue_label="SingleRun"):
    """Vẽ reward & average reward trong thư mục PPO/."""
    enable_latex_font()
    ppo_dir = os.path.join(run_path, "PPO")
    reward_file = os.path.join(ppo_dir, "reward_hist_PPO.txt")
    avg_file    = os.path.join(ppo_dir, "avg_reward_hist_PPO.txt")

    reward = read_series(reward_file)
    avg    = read_series(avg_file)

    if reward.size == 0:
        print(f"❌ Không có dữ liệu reward trong {ppo_dir}")
        return

    episodes = np.arange(1, reward.size + 1)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    ax.plot(episodes, reward, color="gray", alpha=0.5, lw=1, label="Reward per Episode")
    if avg.size > 0:
        ax.plot(episodes[:avg.size], avg, color="tab:orange", lw=4, label="Average Reward (window=20)")
    else:
        w = 20
        ma = np.convolve(reward, np.ones(w)/w, mode='valid')
        ax.plot(np.arange(w, reward.size+1), ma, color="tab:orange", lw=2.6,
                label=f"Moving Avg (window={w})")

    ax.set_xlabel("Episode", fontsize=36)
    ax.set_ylabel("Reward", fontsize=36)
    ax.legend(loc="lower right", framealpha=0, fontsize=30,)
    ax.grid(True, linestyle="--", color="lightgrey", alpha=0.6)

    out_dir = os.path.join(ppo_dir, "plots")
    _save(fig, out_dir, f"reward_curve_{ue_label}")

# =======================================================
# === MAIN-1: gọi 1 lần xuất tất cả ====================
# =======================================================
def main_1(run_path, ue_label, divide_by):
    # Bars
    plot_accept_bar(run_path, ue_label=ue_label, divide_by=divide_by)
    plot_time_bar(run_path, ue_label=ue_label)
    plot_latency_bar(run_path, ue_label=ue_label)
    plot_reward_bar(run_path, ue_label=ue_label)
    plot_throughput_bar(run_path, ue_label=ue_label)
    # PPO curve
    plot_ppo_reward_curve(run_path, ue_label=ue_label)
    print(f"✅ All figures saved in:\n   • {os.path.join(run_path, 'compare_plots')}\n   • {os.path.join(run_path, 'PPO', 'plots')}")

# =======================================================
# === CHẠY THỬ ==========================================
# =======================================================
if __name__ == "__main__":
    main_1("./results_final_3/run_20260122_131319", ue_label="single", divide_by=20)
