import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter1d
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

wandb_dir = os.environ.get("WANDB_DIR", "./")
data_path = os.path.join(wandb_dir, "data", "trap_split_T1000_long", "merged_group_stats.parquet")

print(f"Loading data from {data_path}...")
df = pd.read_parquet(data_path)
df["social_fusion"] = df["social_fusion"].astype(str)
df["frac_lost"] = 1.0 - df["frac_on_good"] - df["frac_on_bad"]

FUSIONS = [
    "naive",
    "flat_factor_merging_2",
    "flat_factor_merging_8",
    "ci",
    "ci_optimal_weight",
    "firsthand",
]
FUSION_LABELS = {
    "naive": "Naive",
    "flat_factor_merging_2": r"Flat factor ($\kappa=2$)",
    "flat_factor_merging_8": r"Flat factor ($\kappa=8$)",
    "ci": "CI (equal)",
    "ci_optimal_weight": "CI (optimal)",
    "firsthand": "First-hand only",
}
FUSION_COLORS = {
    "naive": "#2a78d6",
    "flat_factor_merging_2": "#7b61c2",
    "flat_factor_merging_8": "#c2549c",
    "ci": "#eb6834",
    "ci_optimal_weight": "#e6a520",
    "firsthand": "#1baf7a",
}
CONDITION_TITLES = {0: "Start near good target", 1: "Start near bad target"}

SMOOTH_SIGMA = 12        # Gaussian smoothing sigma for Row 1 (generations)
BIN_GEN = 10             # 10-generation bins for Row 2
RATE_SMOOTH_SIGMA = 2.5  # Gaussian smoothing sigma in units of 10-gen bins (25 generations)


def gaussian_smooth_valid(y: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized Gaussian smoothing without boundary flat-padding artifacts."""
    mask = np.isfinite(y)
    y_filled = np.where(mask, y, 0.0)
    num = gaussian_filter1d(y_filled, sigma=sigma, mode="constant", cval=0.0)
    den = gaussian_filter1d(mask.astype(float), sigma=sigma, mode="constant", cval=0.0)
    return num / np.maximum(den, 1e-12)


conditions = sorted(df["close_target_index"].unique())
n_gen = int(df["generation"].max()) + 1
n_bins = n_gen // BIN_GEN
bin_gens = np.arange(1, n_bins + 1) * BIN_GEN

fig, axes = plt.subplots(2, len(conditions), figsize=(6.2 * len(conditions), 7.2), sharex=True, sharey="row")
axes = np.atleast_2d(axes)

for col_idx, cond in enumerate(conditions):
    ax_lost = axes[0, col_idx]
    ax_rate = axes[1, col_idx]
    sub = df[df["close_target_index"] == cond]

    for sf in FUSIONS:
        d_sf = sub[sub["social_fusion"] == sf]
        if len(d_sf) == 0:
            continue

        # Number of lost groups at each generation (1..1000)
        lost_series = d_sf.groupby("generation")["frac_lost"].sum().reindex(range(n_gen), fill_value=0.0)
        gens = lost_series.index.values + 1
        raw_vals = lost_series.values.astype(float)
        color = FUSION_COLORS[sf]
        label = FUSION_LABELS[sf]

        # Smooth running average for Row 1
        lost_smooth = gaussian_smooth_valid(raw_vals, sigma=SMOOTH_SIGMA)

        # Row 1: Lost groups over generations (smoothed trend)
        ax_lost.plot(gens, lost_smooth, color=color, linewidth=2.0, label=label)

        # Row 2: Newly trapped (lost) groups over 10-generation bins (smoothed trend)
        bin_means = raw_vals[: n_bins * BIN_GEN].reshape(n_bins, BIN_GEN).mean(axis=1)
        newly_trapped_10_raw = np.diff(bin_means, prepend=raw_vals[0])
        newly_trapped_10_smooth = gaussian_smooth_valid(newly_trapped_10_raw, sigma=RATE_SMOOTH_SIGMA)
        ax_rate.plot(bin_gens, newly_trapped_10_smooth, color=color, linewidth=2.0, label=label)

    ax_lost.set_title(CONDITION_TITLES.get(cond, f"close_target_index={cond}"), fontsize=12)
    ax_lost.set_xlim(1, n_gen)
    ax_lost.set_ylim(0, 650)
    ax_lost.grid(True, color="#e5e5e5", linewidth=0.6)

    ax_rate.axhline(0, color="#666666", linestyle="--", linewidth=0.8, zorder=1)
    ax_rate.set_xlim(1, n_gen)
    ax_rate.set_yscale("symlog", linthresh=6.0, linscale=1.6)
    ax_rate.set_ylim(-15, 70)
    ax_rate.yaxis.set_major_locator(mticker.FixedLocator([-10, -6, -4, -2, 0, 2, 4, 6, 20, 60]))
    ax_rate.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_rate.set_xlabel("Generation")
    ax_rate.grid(True, color="#e5e5e5", linewidth=0.6)

    for ax in (ax_lost, ax_rate):
        ax.set_xticks([1, 250, 500, 750, 1000])
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

axes[0, 0].set_ylabel("Lost groups (count)")
axes[1, 0].set_ylabel(f"Newly trapped groups\n(per {BIN_GEN} generations)")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(FUSIONS), frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.04, 1, 1))

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dirs = list(dict.fromkeys([os.path.join(script_dir, "figures"), "figures"]))
for d in out_dirs:
    os.makedirs(d, exist_ok=True)
    out_file = os.path.join(d, "lost_and_newly_trapped_1000gen.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    fig.savefig(out_file.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {out_file}")
plt.close(fig)
