import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

wandb_dir = os.environ.get("WANDB_DIR", "./")
data_path = os.path.join(wandb_dir, "data", "trap_split_T1000_long", "merged_group_stats.parquet")

print(f"Loading data from {data_path}...")
df = pd.read_parquet(data_path)
df["run_id"] = df["run_id"].astype(str)
df["social_fusion"] = df["social_fusion"].astype(str)
df["unique_group"] = df["run_id"] + "_" + df["group_id"].astype(str)
df = df.sort_values(["unique_group", "generation"])

FUSIONS = [
    "naive",
    "flat_factor_merging_2",
    "flat_factor_merging_8",
    "ci_optimal_weight",
    "firsthand",
]
FUSION_LABELS = {
    "naive": "Naive",
    "flat_factor_merging_2": "Flat factor\n($\\kappa=2$)",
    "flat_factor_merging_8": "Flat factor\n($\\kappa=8$)",
    "ci_optimal_weight": "Covariance\nintersection",
    "firsthand": "First-hand\nonly",
}
FUSION_COLORS = {
    "naive": "#2a78d6",
    "flat_factor_merging_2": "#7b61c2",
    "flat_factor_merging_8": "#c2549c",
    "ci_optimal_weight": "#eb6834",
    "firsthand": "#1baf7a",
}

df = df[df["social_fusion"].isin(FUSIONS)].copy()
df["trap_class"] = df["trap_class"].astype(str)
df["final_class"] = df["final_class"].astype(str)

# Per-generation change in group-mean private sensing probability:
# d_prob_priv = w_priv(g) - w_priv(g - 1) (probability change per generation)
df["d_prob_priv"] = df.groupby("unique_group")["prob_priv"].diff()
df["prev_trap_class"] = df.groupby("unique_group")["trap_class"].shift(1)

# Exclude the initial 25-generation transit phase where bad-start groups travel across the arena
df = df[df["generation"] >= 25].dropna(subset=["d_prob_priv", "prev_trap_class"]).copy()

# Split into regimes using position-based trap_class in current and previous generation.
# For Trapped, restrict to groups that actually end up trapped (final_class != "good") so escaping
# groups recovering via private sensing are not mixed into the trapped regime.
regimes = [
    ("On good target", df[(df["trap_class"] == "good") & (df["prev_trap_class"] == "good")]),
    ("Trapped", df[(df["final_class"] != "good") & (df["trap_class"] != "good") & (df["prev_trap_class"] != "good")]),
]


def summarize_regime(sub_df: pd.DataFrame, min_gens: int = 20) -> pd.DataFrame:
    """Compute per-group mean change in w_priv per generation (x 10^-3), then mean & 95% CI across groups."""
    grp = (
        sub_df.groupby(["social_fusion", "unique_group"], observed=True)
        .agg(
            d_prob_priv_mean=("d_prob_priv", "mean"),
            n_gens=("d_prob_priv", "count"),
        )
        .reset_index()
    )
    grp = grp[grp["n_gens"] >= min_gens].copy()
    # Units: change in group-mean private channel probability per generation (x 10^-3)
    grp["grad_priv"] = grp["d_prob_priv_mean"] * 1e3

    summary = (
        grp.groupby("social_fusion", observed=True)
        .agg(
            grad_priv_mean=("grad_priv", "mean"),
            grad_priv_ci=("grad_priv", lambda s: 1.96 * s.sem()),
            n_groups=("unique_group", "count"),
        )
        .reset_index()
    )
    # Reorder rules by selection gradient strength (ascending)
    return summary.sort_values("grad_priv_mean", ascending=True).reset_index(drop=True)


summaries = []
max_abs = 0.0
for title, sub_df in regimes:
    s = summarize_regime(sub_df)
    print(f"=== {title} ===")
    print(s.to_string(index=False))
    summaries.append((title, s))
    bound = (s["grad_priv_mean"].abs() + s["grad_priv_ci"]).max()
    max_abs = max(max_abs, float(bound))

# Use the canonical social fusion order across all figures
ordered_fusions = FUSIONS
ylim_abs = max(max_abs * 1.15, 0.28)

fig, axes = plt.subplots(1, len(summaries), figsize=(9.8, 5.4), sharey=True)
axes = np.atleast_1d(axes)

for col_idx, (ax, (title, summary)) in enumerate(zip(axes, summaries)):
    summary = summary.set_index("social_fusion").reindex(ordered_fusions).reset_index()
    x = np.arange(len(ordered_fusions))
    means = summary["grad_priv_mean"].values
    cis = summary["grad_priv_ci"].values

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9, zorder=1)

    for j, (sf, m, ci) in enumerate(zip(ordered_fusions, means, cis)):
        color = FUSION_COLORS[sf]
        ax.errorbar(
            j,
            m,
            yerr=ci,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=4.5,
            capthick=1.6,
            markersize=8.0,
            zorder=3,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([FUSION_LABELS[sf] for sf in ordered_fusions], fontsize=8.8)
    ax.set_xlabel("Social information fusion rule", labelpad=8)
    ax.set_xlim(-0.5, len(ordered_fusions) - 0.5)
    ax.set_ylim(-ylim_abs, ylim_abs)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

axes[0].set_ylabel(r"Selection gradient on $\omega_{\mathrm{priv}}$" + "\n" + r"($\Delta \bar{\omega}_{\mathrm{priv}}$ per generation, $\times 10^{-3}$)")

fig.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dirs = list(dict.fromkeys([os.path.join(script_dir, "figures"), "figures"]))
for d in out_dirs:
    os.makedirs(d, exist_ok=True)
    out_file = os.path.join(d, "selection_gradient_priv_on_target.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    fig.savefig(out_file.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {out_file}")
plt.close(fig)
