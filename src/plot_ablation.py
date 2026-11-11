import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


# 讀取每組實驗的 PR-AUC 平均值與標準差
def get_result(tag):
    path = RUNS / tag / "metrics.json"

    with open(path, "r") as f:
        data = json.load(f)

    result = data["summary"]["test_pr_auc"]

    return result["mean"], result["std"]


# 畫出含 Error Bars 的折線圖
def plot_ablation(x, tags, xlabel, filename):

    means = []
    stds = []

    for tag in tags:
        mean, std = get_result(tag)
        means.append(mean)
        stds.append(std)

    fig, ax = plt.subplots(figsize=(5, 3.2))

    ax.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        linewidth=1.5
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test PR-AUC")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=300)
    plt.close(fig)

    print(f"Saved: {filename}")


# Figure 1：Feature Selection
plot_ablation(
    [5, 9, 15, 30],
    [
        "feat5_local",
        "feat9_local",
        "t2_ae__feat-top15__cont-0__score-ce__tau-f1",
        "baseline_local"
    ],
    "Number of Features",
    "fig_ablation_features.png"
)


# Figure 2：Training Contamination
plot_ablation(
    [0, 1, 5, 10],
    [
        "baseline_local",
        "cont1_fixed",
        "cont5_fixed",
        "cont10_fixed"
    ],
    "Training Contamination (%)",
    "fig_ablation_contamination_fixed.png"
)