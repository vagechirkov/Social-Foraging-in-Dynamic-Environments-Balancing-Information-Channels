import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from dotenv import load_dotenv


def blend_with_white(color, alpha: float):
    """Blend color with white in RGB so PDF hatch patterns have distinct fill and stripe RGBs."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(alpha * rgb + (1.0 - alpha) * 1.0)

# Load environment variables from .env if present
load_dotenv()

# Data produced by download_occupancy_wandb_data.py
wandb_dir = os.environ.get("WANDB_DIR", "./")
data_dir = os.path.join(wandb_dir, "data", "occupancy_fusion_T_sweep")
blocks_path = os.path.join(data_dir, "merged_occupancy_blocks.parquet")
runs_path = os.path.join(data_dir, "runs.parquet")

print(f"Loading data from {blocks_path}...")
blocks = pd.read_parquet(blocks_path)
runs = pd.read_parquet(runs_path)

TRAP_BLOCK = int(runs["trap_block"].iloc[0]) if "trap_block" in runs else 250
TRAP_WINDOW = int(runs["trap_window"].iloc[0]) if "trap_window" in runs else 10

GENERATIONS = [25, 50, 100, 200]

FUSIONS = [
    "naive",
    "flat_factor_merging_2",
    "flat_factor_merging_8",
    "ci_optimal_weight",
    "firsthand",
]
FUSION_LABELS = {
    "naive": "Naive",
    "flat_factor_merging_2": r"Flat factor ($\kappa=2$)",
    "flat_factor_merging_8": r"Flat factor ($\kappa=8$)",
    "ci_optimal_weight": "Covariance intersection",
    "firsthand": "First-hand only",
}
FUSION_COLORS = {
    "naive": "#2a78d6",
    "flat_factor_merging_2": "#7b61c2",
    "flat_factor_merging_8": "#c2549c",
    "ci_optimal_weight": "#eb6834",
    "firsthand": "#1baf7a",
}
CONDITION_TITLES = {0: "Start near good target", 1: "Start near bad target"}

run_meta = blocks[["run_id", "close_target_index", "social_fusion", "eval_interval"]].drop_duplicates()


def compute_counts_at_generation(gen: int) -> pd.DataFrame:
    """Classify each group over the trailing TRAP_WINDOW blocks ending at generation `gen`."""
    end_block = (gen * blocks["eval_interval"]) // TRAP_BLOCK
    start_block = (end_block - TRAP_WINDOW).clip(lower=0)
    win = blocks[(blocks["block"] >= start_block) & (blocks["block"] < end_block)]

    g = win.groupby(["run_id", "group_id", "status"], observed=False).size().unstack("status", fill_value=0)
    n_blk = g[["good", "bad", "lost"]].sum(axis=1)
    g = g[n_blk > 0].copy()
    n_blk = n_blk[n_blk > 0]

    cls = np.where(g["bad"] >= g["lost"], "bad", "lost")
    cls = np.where(g["good"] / n_blk > 0.5, "good", cls)
    g["trap_class"] = pd.Categorical(cls, categories=["good", "bad", "lost"])

    merged = g.reset_index()[["run_id", "group_id", "trap_class"]].merge(run_meta, on="run_id")
    counts = (
        merged.groupby(["close_target_index", "social_fusion", "eval_interval"], observed=True)["trap_class"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["good", "bad", "lost"]:
        if col not in counts:
            counts[col] = 0
    counts["n"] = counts[["good", "bad", "lost"]].sum(axis=1)
    counts["trapped"] = counts["bad"] + counts["lost"]
    counts["generation"] = gen
    return counts


counts_by_gen = {gen: compute_counts_at_generation(gen) for gen in GENERATIONS}
all_counts = pd.concat(counts_by_gen.values(), ignore_index=True)
print(all_counts.to_string())

eval_intervals = sorted(all_counts["eval_interval"].unique())
conditions = sorted(all_counts["close_target_index"].unique())

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dirs = list(dict.fromkeys([os.path.join(script_dir, "figures"), "figures"]))
for d in out_dirs:
    os.makedirs(d, exist_ok=True)

n_fusions = len(FUSIONS)
width = 0.84 / n_fusions
x = np.arange(len(eval_intervals))


def plot_row(axes_row, counts_df, gen, show_title=True, show_xlabel=True):
    row_max = 0.0
    for ax, cond in zip(axes_row, conditions):
        for j, sf in enumerate(FUSIONS):
            d = counts_df[(counts_df["close_target_index"] == cond) & (counts_df["social_fusion"] == sf)]
            d = d.set_index("eval_interval").reindex(eval_intervals)
            xs = x + (j - (n_fusions - 1) / 2) * width
            bad = 100 * d["bad"] / d["n"]
            lost = 100 * d["lost"] / d["n"]
            total = bad + lost
            color = FUSION_COLORS[sf]

            # Trapped on the bad target (solid) + trapped on neither target (hatched)
            ax.bar(xs, bad, width * 0.92, color=color, edgecolor="white", linewidth=0.6)
            ax.bar(
                xs,
                lost,
                width * 0.92,
                bottom=bad,
                color=blend_with_white(color, 0.30),
                hatch="///",
                edgecolor=blend_with_white(color, 0.70),
                linewidth=0,
            )
            for xi, t in zip(xs, total):
                if np.isfinite(t):
                    row_max = max(row_max, float(t))
                    ax.text(xi, t + 0.8, f"{t:.1f}", ha="center", va="bottom", fontsize=6.5, color="#333333")

        if show_title:
            ax.set_title(CONDITION_TITLES.get(cond, f"close_target_index={cond}"))
        ax.set_xticks(x)
        ax.set_xticklabels([str(T) for T in eval_intervals])
        if show_xlabel:
            ax.set_xlabel("Generation length T (ticks)")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

    axes_row[0].set_ylabel(f"Gen {gen}\nTrapped groups (%)")
    axes_row[0].set_ylim(0, max(row_max * 1.14, 10.0))


handles = [Patch(facecolor=FUSION_COLORS[sf], label=FUSION_LABELS[sf]) for sf in FUSIONS]
handles += [
    Patch(facecolor="#888888", label="On bad target"),
    Patch(
        facecolor=blend_with_white("#888888", 0.30),
        hatch="///",
        edgecolor=blend_with_white("#888888", 0.70),
        linewidth=0,
        label="On neither target",
    ),
]

# 1. Combined 4x2 grid figure (rows = generations 25, 50, 100, 200)
fig, axes = plt.subplots(
    len(GENERATIONS),
    len(conditions),
    figsize=(6.2 * len(conditions), 3.1 * len(GENERATIONS)),
    sharey="row",
)
axes = np.atleast_2d(axes)

for row_idx, gen in enumerate(GENERATIONS):
    plot_row(
        axes[row_idx],
        counts_by_gen[gen],
        gen,
        show_title=(row_idx == 0),
        show_xlabel=(row_idx == len(GENERATIONS) - 1),
    )

fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.04, 1, 1))

for d in out_dirs:
    out_file = os.path.join(d, "trapped_fraction_fusion_T.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    fig.savefig(out_file.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {out_file}")
plt.close(fig)

# 2. Individual 1x2 figure per generation
for gen in GENERATIONS:
    fig_g, axes_g = plt.subplots(1, len(conditions), figsize=(6.2 * len(conditions), 4.0), sharey=True)
    axes_g = np.atleast_1d(axes_g)
    plot_row(axes_g, counts_by_gen[gen], gen, show_title=True, show_xlabel=True)
    axes_g[0].set_ylabel("Trapped groups (%)")
    fig_g.suptitle(f"Generation {gen}", fontsize=12, y=1.01)
    fig_g.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig_g.tight_layout(rect=(0, 0.06, 1, 1))

    for d in out_dirs:
        out_file_g = os.path.join(d, f"trapped_fraction_fusion_T_gen{gen}.png")
        fig_g.savefig(out_file_g, dpi=300, bbox_inches="tight")
        fig_g.savefig(out_file_g.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"Saved {out_file_g}")
    plt.close(fig_g)

