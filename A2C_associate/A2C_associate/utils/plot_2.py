import os
import numpy as np
import matplotlib.pyplot as plt

# =======================================================
# === HELPERS ===========================================
# =======================================================
def read_series(path):
    """Đọc file float theo dòng -> np.array."""
    if not os.path.exists(path):
        print(f"⚠️ Missing: {path}")
        return np.array([])
    vals = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals.append(float(s))
            except:
                pass
    return np.asarray(vals, dtype=np.float64)

def read_mean(path):
    a = read_series(path)
    return float(a.mean()) if a.size else 0.0

def moving_avg(x, window):
    if x.size == 0:
        return x
    window = max(1, int(window))
    if window >= x.size:
        return np.full_like(x, x.mean())
    kernel = np.ones(window, dtype=float) / window
    y = np.convolve(x, kernel, mode="valid")
    pad_left = window - 1
    left = np.full(pad_left // 2, y[0])
    right = np.full(pad_left - pad_left // 2, y[-1])
    return np.concatenate([left, y, right])

def enable_latex_font():
    """Cấu hình LaTeX (Computer Modern) cho toàn bộ hình."""
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

# alias cho phần training curve
def read_reward_file(path):
    return read_series(path)

# =======================================================
# === 0) PPO training reward (across runs, averaged samples)
# =======================================================
def plot_training_curve_avg(root_dir, run_dirs, ue_labels):
    """
    Vẽ 1 hình đường: trung bình reward theo "điểm mẫu đều" cho 30/40/50UE.
    """
    enable_latex_font()

    colors  = ['#E69F00', "#009E35", '#0072B2']  # orange, green, blue (paper colors)
    markers = ['o', 's', 'D']

    #plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for i, run in enumerate(run_dirs):
        ppo_dir  = os.path.join(root_dir, run, "PPO")
        avg_file = os.path.join(ppo_dir, "avg_reward_hist_PPO.txt")
        data     = read_reward_file(avg_file)
        if len(data) == 0:
            print(f"⚠️ No avg_reward_hist_PPO.txt in {ppo_dir}")
            continue

        # Lấy ~10 điểm trung bình đều nhau trên trục episode
        num_points = 10
        step = max(1, len(data) // num_points)
        sampled = [np.mean(data[j:j+step]) for j in range(0, len(data), step)]
        episodes = np.linspace(1, len(data), len(sampled))

        ax.plot(
            episodes, sampled,
            lw=3.2, color=colors[i],
            marker=markers[i], markersize=9, markevery=1,
            label=ue_labels[i], alpha=0.9
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.set_xlim(0, None)

    # Phóng nhẹ trục Y để tách đường
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin * 0.95, ymax * 1.03)

    ax.grid(True, linestyle="--", color="lightgrey", alpha=0.7, linewidth=1.0)
    ax.tick_params(axis='both', which='major', length=6, width=1.3)

    # ======= Viền đen cho toàn hình =======
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    legend = ax.legend(
        loc="lower right", frameon=True, framealpha=0.9,
        fancybox=True, edgecolor="lightgray", fontsize=22
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=1.4)

    out_dir = os.path.join(root_dir, "compare_plots")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "reward_training_avg.pdf")
    png_path = os.path.join(out_dir, "reward_training_avg.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=350)
    plt.close(fig)
    print(f"✅ Saved enhanced PPO training reward curve:\n  • {pdf_path}\n  • {png_path}")


# =======================================================
# === 1) ACCEPT (per-run, no MILP)
# =======================================================
def plot_accept_bar_single(run_path, ue_label="Run", divide_by=None):
    enable_latex_font()
    algos  = ["RandomRU", "RoundRobin", "NearestRU", "PPO-GraphSAGE", "A2C-MLP"]
    labels = ["Random-RU", "RoundRobin", "Nearest-RU", "PPO-GSA", "A2C-MLP"]
    colors = ['#C2C2C2', '#A6D854', '#80B1D3', '#D95F02', "#1F0F03"]
    hatches  = ['/', '.', 'o', '//',  '///']
    bar_w, gap = 40, 10

    paths = {
        "RandomRU":     os.path.join(run_path, "baseline_results", "RandomRU",   "accept_hist_RandomRU.txt"),
        "RoundRobin":   os.path.join(run_path, "baseline_results", "RoundRobin", "accept_hist_RoundRobin.txt"),
        "NearestRU":    os.path.join(run_path, "baseline_results", "NearestRU",  "accept_hist_NearestRU.txt"),
        "PPO-GraphSAGE":os.path.join(run_path, "evaluation_agent_PPO", "total_accept_PPO.txt"),
        "A2C-MLP":      os.path.join(run_path, "evaluation_agent_A2C", "total_accept_A2C.txt"),
    }
    vals = [read_mean(paths[a]) for a in algos]
    if divide_by and divide_by > 0:
        vals = [v / divide_by for v in vals]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Acceptance rate", fontsize=36)

    x, xt = 0, []
    for i, a in enumerate(algos):
        ax.bar(x, vals[i], width=bar_w, color='white', edgecolor=colors[i],
               hatch=hatches[i], label=labels[i], zorder=3, linewidth=2.2)
        xt.append(x); x += bar_w + gap

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

    out = os.path.join(run_path, "compare_plots"); os.makedirs(out, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"accept_{ue_label}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(out, f"accept_{ue_label}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

# =======================================================
# === 2) REWARD (per-run, no MILP)
# =======================================================
def plot_reward_bar_single(run_path, ue_label="Run"):
    enable_latex_font()
    algos  = ["RandomRU", "RoundRobin", "NearestRU", "PPO-GraphSAGE", "A2C-MLP"]
    labels = ["Random-RU", "RoundRobin", "Nearest-RU", "PPO-GSA", "A2C-MLP"]
    colors = ['#C2C2C2', '#A6D854', '#80B1D3', '#D95F02', "#1F0F03"]
    hatches  = ['/', '.', 'o', '//',  '///']
    bar_w, gap = 40, 10

    paths = {
        "RandomRU":     os.path.join(run_path, "baseline_results", "RandomRU",   "reward_hist_RandomRU.txt"),
        "RoundRobin":   os.path.join(run_path, "baseline_results", "RoundRobin", "reward_hist_RoundRobin.txt"),
        "NearestRU":    os.path.join(run_path, "baseline_results", "NearestRU",  "reward_hist_NearestRU.txt"),
        "PPO-GraphSAGE":os.path.join(run_path, "evaluation_agent_PPO", "total_reward_PPO.txt"),
        "A2C-MLP":      os.path.join(run_path, "evaluation_agent_A2C", "total_reward_A2C.txt"),
    }
    vals = [read_mean(paths[a]) for a in algos]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_ylabel("Average reward", fontsize=40)

    x, xt = 0, []
    for i, a in enumerate(algos):
        ax.bar(x, vals[i], width=bar_w, color='white', edgecolor=colors[i],
               hatch=hatches[i], label=labels[i], zorder=3, linewidth=2.2)
        xt.append(x); x += bar_w + gap

    ax.set_xticks(xt); ax.set_xticklabels([])
    ax.grid(color='lightgrey', linestyle='--', zorder=0)

    # ===== Viền đen cho toàn hình =====
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    ax.legend(
        loc="upper left",
        framealpha=0,
        fontsize=30,          # tăng cỡ chữ legend
        handlelength=1.6,     # chiều dài ký hiệu
        handletextpad=0.6,    # khoảng cách giữa ký hiệu và text
        labelspacing=0.3,     # khoảng cách giữa các dòng legend
        borderpad=0.3,        # padding bên trong khung legend
        ncol=1,               # số cột legend
        columnspacing=1.0,    # khoảng cách giữa các cột
    )

    out = os.path.join(run_path, "compare_plots")
    os.makedirs(out, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"reward_{ue_label}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(out, f"reward_{ue_label}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# =======================================================
# === 3) THROUGHPUT (Mbps, per-run, no MILP)
# =======================================================
def plot_throughput_bar_single(run_path, ue_label="Run"):
    enable_latex_font()
    algos  = ["RandomRU", "RoundRobin", "NearestRU", "PPO-GraphSAGE", "A2C-MLP"]
    labels = ["Random-RU", "RoundRobin", "Nearest-RU", "PPO-GSA", "A2C-MLP"]
    colors = ['#C2C2C2', '#A6D854', '#80B1D3', '#D95F02', "#1F0F03"]
    hatches  = ['/', '.', 'o', '//',  '///']
    bar_w, gap = 40, 10

    paths = {
        "RandomRU":     os.path.join(run_path, "baseline_results", "RandomRU",   "throughput_hist_RandomRU.txt"),
        "RoundRobin":   os.path.join(run_path, "baseline_results", "RoundRobin", "throughput_hist_RoundRobin.txt"),
        "NearestRU":    os.path.join(run_path, "baseline_results", "NearestRU",  "throughput_hist_NearestRU.txt"),
        "PPO-GraphSAGE":os.path.join(run_path, "evaluation_agent_PPO", "total_throughput_PPO.txt"),
        "A2C-MLP":      os.path.join(run_path, "evaluation_agent_A2C", "total_throughput_A2C.txt"),
    }
    vals_mbps = [read_mean(paths[a]) / 1e6 for a in algos]  # bps -> Mbps

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_ylabel("Average throughput (Mbps)", fontsize=36)

    ymax = max(vals_mbps) if vals_mbps else 1.0
    ax.set_ylim(0, ymax * 1.25)

    x, xt = 0, []
    for i, a in enumerate(algos):
        ax.bar(x, vals_mbps[i], width=bar_w, color='white', edgecolor=colors[i],
               hatch=hatches[i], label=labels[i], zorder=3, linewidth=2.2)
        xt.append(x); x += bar_w + gap

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

    out = os.path.join(run_path, "compare_plots"); os.makedirs(out, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"throughput_{ue_label}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(out, f"throughput_{ue_label}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

# =======================================================
# === 4) LATENCY (ms, per-run, no MILP)
# =======================================================
def plot_latency_bar_single(run_path, ue_label="Run"):
    enable_latex_font()
    algos  = ["RandomRU", "RoundRobin", "NearestRU", "PPO-GraphSAGE", "A2C-MLP"]
    labels = ["Random-RU", "RoundRobin", "Nearest-RU", "PPO-GSA", "A2C-MLP"]
    colors = ['#C2C2C2', '#A6D854', '#80B1D3', '#D95F02', "#1F0F03"]
    hatches  = ['/', '.', 'o', '//',  '///']
    bar_w, gap = 40, 10

    paths = {
        "RandomRU":     os.path.join(run_path, "baseline_results", "RandomRU",   "latency_hist_RandomRU.txt"),
        "RoundRobin":   os.path.join(run_path, "baseline_results", "RoundRobin", "latency_hist_RoundRobin.txt"),
        "NearestRU":    os.path.join(run_path, "baseline_results", "NearestRU",  "latency_hist_NearestRU.txt"),
        "PPO-GraphSAGE":os.path.join(run_path, "evaluation_agent_PPO", "total_latency_PPO.txt"),
        "A2C-MLP":      os.path.join(run_path, "evaluation_agent_A2C", "total_latency_A2C.txt"),
    }
    vals_ms = [read_mean(paths[a]) * 1e3 for a in algos]  # s -> ms

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_ylabel("Average latency (ms)", fontsize=36)

    ymax = max(vals_ms) if vals_ms else 1.0
    ax.set_ylim(0, ymax * 1.35)

    x, xt = 0, []
    for i, a in enumerate(algos):
        ax.bar(x, vals_ms[i], width=bar_w, color='white', edgecolor=colors[i],
               hatch=hatches[i], label=labels[i], zorder=3, linewidth=2.2)
        xt.append(x); x += bar_w + gap

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

    out = os.path.join(run_path, "compare_plots"); os.makedirs(out, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"latency_{ue_label}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(out, f"latency_{ue_label}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

# =======================================================
# === 5) TIME (log scale, per-run, no MILP)
# =======================================================
def plot_time_bar_single(run_path, ue_label="Run"):
    enable_latex_font()
    algos  = ["RandomRU", "RoundRobin", "NearestRU", "PPO-GraphSAGE", "A2C-MLP"]
    labels = ["Random-RU", "RoundRobin", "Nearest-RU", "PPO-GSA", "A2C-MLP"]
    colors = ['#C2C2C2', '#A6D854', '#80B1D3', '#D95F02', "#1F0F03"]
    hatches  = ['/', '.', 'o', '//',  '///']
    bar_w, gap = 40, 10

    paths = {
        "RandomRU":     os.path.join(run_path, "baseline_results", "RandomRU",   "time_hist_RandomRU.txt"),
        "RoundRobin":   os.path.join(run_path, "baseline_results", "RoundRobin", "time_hist_RoundRobin.txt"),
        "NearestRU":    os.path.join(run_path, "baseline_results", "NearestRU",  "time_hist_NearestRU.txt"),
        "PPO-GraphSAGE":os.path.join(run_path, "evaluation_agent_PPO", "evaluation_time_PPO.txt"),
        "A2C-MLP":      os.path.join(run_path, "evaluation_agent_A2C", "evaluation_time_A2C.txt"),
    }
    vals = [max(read_mean(paths[a]), 1e-6) for a in algos]  # tránh 0 cho log

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_ylabel("Computing time (s)", fontsize=40)
    ax.set_yscale("log")

    ymin = min(vals) / 3.0
    ymax = max(vals) * 1.8
    ax.set_ylim(ymin, ymax)

    x, xt = 0, []
    for i, a in enumerate(algos):
        ax.bar(x, vals[i], width=bar_w, color='white', edgecolor=colors[i],
               hatch=hatches[i], label=labels[i], zorder=3, linewidth=2.4)
        xt.append(x); x += bar_w + gap

    ax.set_xticks(xt); ax.set_xticklabels([])
    ax.grid(which='major', color='lightgrey', linestyle='--', alpha=0.85, zorder=0)
    ax.grid(which='minor', visible=False)

    # ===== Viền đen cho toàn hình =====
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    ax.legend(
        loc="upper left",
        framealpha=0,
        fontsize=30,          # tăng cỡ chữ legend
        handlelength=1.6,     # chiều dài ký hiệu
        handletextpad=0.6,    # khoảng cách giữa ký hiệu và text
        labelspacing=0.3,     # khoảng cách giữa các dòng legend
        borderpad=0.3,        # padding bên trong khung legend
        ncol=1,               # số cột legend
        columnspacing=1.0,    # khoảng cách giữa các cột
    )

    out = os.path.join(run_path, "compare_plots")
    os.makedirs(out, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"time_{ue_label}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(out, f"time_{ue_label}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)



# =======================================================
# === MAIN-2: per-run bars + global PPO training curve + correlation
# =======================================================
def main_2(root_dir="./results_5",
           runs=("run_30UE_20251107_230034", "run_40UE_20251108_034613", "run_50UE_20251108_080043"),
           ue_labels=("30 UEs", "40 UEs", "50 UEs"),
           divide_by=20,
           ma_window=500,       # reserved if you later want moving avg
           sample_step=2000):   # reserved if you later want sampling

    # 0) Vẽ đường training reward tổng hợp cho 30/40/50UE
    plot_training_curve_avg(root_dir, runs, ue_labels)

    # 1) Vẽ các biểu đồ thanh + 2) biểu đồ tương quan reward-time cho từng run
    for run, lbl in zip(runs, ue_labels):
        run_path = os.path.join(root_dir, run)
        plot_accept_bar_single(run_path, ue_label=lbl, divide_by=divide_by)
        plot_reward_bar_single(run_path, ue_label=lbl)
        plot_throughput_bar_single(run_path, ue_label=lbl)
        plot_latency_bar_single(run_path, ue_label=lbl)
        plot_time_bar_single(run_path, ue_label=lbl)

# =======================================================
# === RUN ===============================================
# =======================================================
if __name__ == "__main__":
    main_2()
