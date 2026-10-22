import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Data produced by download_occupancy_wandb_data.py
wandb_dir = os.environ.get("WANDB_DIR", "./")
data_path = os.path.join(wandb_dir, "data", "occupancy_fusion_T_sweep", "merged_occupancy_final.parquet")

print(f"Loading data from {data_path}...")
df = pd.read_parquet(data_path)

FUSIONS = ["naive", "ci", "firsthand"]
FUSION_LABELS = {"naive": "Naive", "ci": "Covariance intersection", "firsthand": "First-hand only"}
FUSION_COLORS = {"naive": "#2a78d6", "ci": "#eb6834", "firsthand": "#1baf7a"}  # fixed categorical order
CONDITION_TITLES = {0: "Start near good target", 1: "Start near bad target"}


def wilson_ci(k, n, z=1.96):
    """95% Wilson interval for a binomial proportion."""
    p = k / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


# Fraction of groups per final class, for each run
counts = (df.groupby(["close_target_index", "social_fusion", "eval_interval"], observed=True)["final_class"]
          .value_counts().unstack(fill_value=0).reset_index())
counts["n"] = counts[["good", "bad", "lost"]].sum(axis=1)
counts["trapped"] = counts["bad"] + counts["lost"]
counts["ci_low"], counts["ci_high"] = wilson_ci(counts["trapped"], counts["n"])
print(counts)

eval_intervals = sorted(counts["eval_interval"].unique())
conditions = sorted(counts["close_target_index"].unique())

os.makedirs('figures', exist_ok=True)

fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4), sharey=True)
axes = np.atleast_1d(axes)
width = 0.26
x = np.arange(len(eval_intervals))

for ax, cond in zip(axes, conditions):
    for j, sf in enumerate(FUSIONS):
        d = counts[(counts["close_target_index"] == cond) & (counts["social_fusion"] == sf)]
        d = d.set_index("eval_interval").reindex(eval_intervals)
        xs = x + (j - 1) * width
        bad = 100 * d["bad"] / d["n"]
        lost = 100 * d["lost"] / d["n"]
        total = bad + lost
        color = FUSION_COLORS[sf]

        # Trapped on the bad target (solid) + trapped on neither target (hatched)
        ax.bar(xs, bad, width * 0.92, color=color, edgecolor="white", linewidth=0.8)
        ax.bar(xs, lost, width * 0.92, bottom=bad, color=color, alpha=0.35, hatch="///",
               edgecolor=color, linewidth=0)
        err = np.vstack([total - 100 * d["ci_low"], 100 * d["ci_high"] - total])
        ax.errorbar(xs, total, yerr=err, fmt="none", ecolor="#555555", elinewidth=1, capsize=2)
        for xi, t, hi in zip(xs, total, 100 * d["ci_high"]):
            ax.text(xi, hi + 1, f"{t:.1f}", ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_title(CONDITION_TITLES.get(cond, f"close_target_index={cond}"))
    ax.set_xticks(x)
    ax.set_xticklabels([str(T) for T in eval_intervals])
    ax.set_xlabel("Generation length T (ticks)")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

axes[0].set_ylabel("Trapped groups (%)")
axes[0].set_ylim(0, None)

handles = [Patch(facecolor=FUSION_COLORS[sf], label=FUSION_LABELS[sf]) for sf in FUSIONS]
handles += [Patch(facecolor="#888888", label="On bad target"),
            Patch(facecolor="#888888", alpha=0.35, hatch="///", edgecolor="#888888", label="On neither target")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1))

out_file = os.path.join('figures', 'trapped_fraction_fusion_T.png')
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.savefig(out_file.replace(".png", ".pdf"), bbox_inches='tight')
print(f"Saved {out_file}")
