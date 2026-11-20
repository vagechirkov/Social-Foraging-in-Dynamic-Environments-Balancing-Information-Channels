"""Paper figures 2, 3, and supplementary figures (1000 generations):
- Main text:
  - Figure 2 (`naive`): reward heatmap, group location, and channel use of Best 100 and Worst trapped (max 100, shown only if >= 10 trapped).
  - Figure 3 (`naive`): Good start (Panel A) and Bad start (Panel B) ternary flow fields before/after divergence.
- Supplement:
  - Combined Figure 2 (`fig2_supp_1000gen.pdf`) for flat_factor_merging_2, flat_factor_merging_8, ci_optimal_weight, firsthand.
  - Combined Figure 3 (`fig3_supp_1000gen.pdf`) for flat_factor_merging_2, flat_factor_merging_8, ci_optimal_weight, firsthand.
  - Simplex occupancy snapshot heatmap (`fig_simplex_snapshots_1000gen.pdf`) for all 5 fusion rules x both initializations at generations 0, 25, 50, 100, 200, 400, 800.
"""
import os
import argparse
import warnings
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

MAIN_FUSION = "naive"
SUPP_FUSIONS = ["flat_factor_merging_2", "flat_factor_merging_8", "ci_optimal_weight", "firsthand"]
ALL_FUSIONS = [MAIN_FUSION] + SUPP_FUSIONS

FUSION_TITLES = {
    "naive": "Naive merging",
    "flat_factor_merging_2": r"Flat-factor merging ($\kappa = 2$)",
    "flat_factor_merging_8": r"Flat-factor merging ($\kappa = 8$)",
    "ci_optimal_weight": "Covariance intersection",
    "firsthand": "First-hand only",
}
FUSION_SHORT_LABELS = {
    "naive": "Naive",
    "flat_factor_merging_2": "Flat factor\n($\\kappa=2$)",
    "flat_factor_merging_8": "Flat factor\n($\\kappa=8$)",
    "ci_optimal_weight": "Covariance\nintersection",
    "firsthand": "First-hand\nonly",
}

parser = argparse.ArgumentParser(description="Paper figures 2, 3, and supplementary figures (1000 generations).")
parser.add_argument("--fusions", nargs="+", default=ALL_FUSIONS)
parser.add_argument("--generations", nargs="+", type=int, default=[1000])
parser.add_argument("--figures", nargs="+", default=["fig2", "fig3", "supp", "snapshots"])
args = parser.parse_args()

wandb_dir = os.environ.get("WANDB_DIR", "./")
DATASETS = {200: "trap_split_T1000_200gen", 1000: "trap_split_T1000_long"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MANUSCRIPT_DIR = os.path.join(REPO_ROOT, 'figures', 'manuscript')
OUT_DIRS = list(dict.fromkeys([
    os.path.join('figures', 'paper_figures'),
    os.path.join(SCRIPT_DIR, 'figures', 'paper_figures'),
]))
for _d in OUT_DIRS + [MANUSCRIPT_DIR, os.path.join(MANUSCRIPT_DIR, 'supplement')]:
    os.makedirs(_d, exist_ok=True)

MANUSCRIPT_ALIASES = {
    "fig2_2_1000gen_1_naive": "f_2.pdf",
    "fig3_2_1000gen_1_naive": "f_3.pdf",
    "fig2_supp_1_1000gen": os.path.join("supplement", "s1_rewards_1.pdf"),
    "fig2_supp_2_1000gen": os.path.join("supplement", "s1_rewards_2.pdf"),
    "fig3_supp_1_1000gen": os.path.join("supplement", "s2_simplex_1.pdf"),
    "fig3_supp_2_1000gen": os.path.join("supplement", "s2_simplex_2.pdf"),
    "fig_simplex_snapshots_1000gen": os.path.join("supplement", "s3_simplex_snapshots.pdf"),
}

# Reward column used for heatmap colour and ranking the worst/best groups
PERF_COL = 'median_avg_fitness'

# Clusters, ranked by mean median reward over the run: the worst N_CLUSTER trapped groups (if >= MIN_CLUSTER_GROUPS)
# and the best N_CLUSTER groups overall (fig. 2 only)
N_CLUSTER = 100
MIN_CLUSTER_GROUPS = 10
WORST_COLOR, BEST_COLOR = '#a3135f', '#4d9221'
CLUSTER_NAMES = {WORST_COLOR: "Worst trapped", BEST_COLOR: f"Best {N_CLUSTER}"}

# Fig. 3
PANEL_A_END = 100       # generations 1-divergence and divergence-PANEL_A_END
DIVERGENCE = 0.05       # sets diverge when their mean channel use differs by more than this in the simplex
FLOW_MIN_SHARE = 0.02   # draw the flow only where at least this share of the set's groups pass

# Snapshot supplementary figure
SNAPSHOT_GENS = [0, 25, 50, 100, 200, 400, 800]

GEN_ORDER = {200: 1, 1000: 2}
FUSION_ORDER = {
    "naive": 1,
    "flat_factor_merging_2": 2,
    "flat_factor_merging_8": 3,
    "ci_optimal_weight": 4,
    "firsthand": 5,
}

STARTS = [(0, "Start near good target"), (1, "Start near bad target")]
STARTS_MULTILINE = {0: "Start near\ngood target", 1: "Start near\nbad target"}
CHANNELS = [('prob_priv', "Private"), ('prob_soc', "Social"), ('prob_none', "None")]
FREE_COLOR, FREE_CMAP, WORST_CMAP = '#1b7837', 'Greens', 'RdPu'
STATE_COLORS = {"good": '#1b7837', "lost": '#c7c7c7', "bad": '#762a83'}
STATE_LABELS = {"good": "On good target", "lost": "On neither", "bad": "On bad target"}
perf_cmap = mcolors.LinearSegmentedColormap.from_list('custom_perf', [
    '#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7',
    '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419'
])
H = np.sqrt(3) / 2

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def load(n_gen):
    path = os.path.join(wandb_dir, "data", DATASETS[n_gen], "merged_group_stats.parquet")
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    df['run_id'] = df['run_id'].astype(str)
    df['social_fusion'] = df['social_fusion'].astype(str)
    df['trap_class'] = df['trap_class'].astype(str)
    df['unique_group'] = df['run_id'] + '_' + df['group_id'].astype(str)
    df['mx'] = df['prob_none'] + 0.5 * df['prob_soc']
    df['my'] = H * df['prob_soc']
    last_gen = df['generation'].max()
    last_class = df[df['generation'] == last_gen].set_index('unique_group')['trap_class']
    df['outcome'] = np.where(last_class.reindex(df['unique_group']).values != "good", "Trapped", "Not trapped")
    return df


def with_steps(d):
    d = d.sort_values(['unique_group', 'generation']).copy()
    d['dmx'] = d.groupby('unique_group')['mx'].diff()
    d['dmy'] = d.groupby('unique_group')['my'].diff()
    return d


def trapped_clusters(d):
    """Worst N_CLUSTER trapped groups by mean median reward (omitted if fewer than MIN_CLUSTER_GROUPS)."""
    perf = d[d['outcome'] == "Trapped"].groupby('unique_group')[PERF_COL].mean().sort_values()
    n = len(perf)
    if n < MIN_CLUSTER_GROUPS:
        return []
    n_worst = min(N_CLUSTER, n)
    return [(f"Worst {n_worst} trapped", WORST_COLOR, perf.index[:n_worst])]


def best_cluster(d):
    """Best N_CLUSTER non-trapped groups by mean median reward."""
    perf = d[d['outcome'] == "Not trapped"].groupby('unique_group')[PERF_COL].mean().nlargest(N_CLUSTER)
    return d[d['unique_group'].isin(perf.index)]


def divergence_generation(a, b):
    """First generation where the mean channel use of two sets differs by more than DIVERGENCE in the simplex."""
    m = a.groupby('generation')[['mx', 'my']].mean().join(
        b.groupby('generation')[['mx', 'my']].mean(), lsuffix='_a', rsuffix='_b').dropna()
    dist = np.hypot(m['mx_a'] - m['mx_b'], m['my_a'] - m['my_b'])
    return int(m.index[dist > DIVERGENCE].min()) if (dist > DIVERGENCE).any() else 5


# ---------------------------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------------------------
def gen_axis(ax, n_gen, label=True):
    ax.set_xlim(1, n_gen)
    ax.set_xticks([1] + list(np.arange(n_gen // 4, n_gen + 1, n_gen // 4)))
    if label:
        ax.set_xlabel("Generation")
    else:
        ax.set_xticklabels([])


def panel_letter(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=15, weight='bold', va='bottom')


def median_iqr(ax, sub, col, color):
    """Median and interquartile range across groups over generations."""
    g = sub.groupby('generation')[col]
    x = np.array(sorted(sub['generation'].unique())) + 1
    ax.fill_between(x, g.quantile(0.25), g.quantile(0.75), color=color, alpha=0.2, linewidth=0)
    ax.plot(x, g.median(), color=color, linewidth=2)


def ternary_frame(ax, fontsize=9, show_labels=True):
    tri = Polygon([[0, 0], [1, 0], [0.5, H]], fill=False, edgecolor='black', linewidth=1.3, zorder=4)
    ax.add_patch(tri)
    if show_labels:
        ax.text(0.5, H + 0.03, "Social", ha='center', va='bottom', fontsize=fontsize)
        ax.text(0.0, -0.04, "Private", ha='center', va='top', fontsize=fontsize)
        ax.text(1.0, -0.04, "None", ha='center', va='top', fontsize=fontsize)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, H + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    return tri


def simplex_grid(res):
    xb = np.linspace(-0.05, 1.05, res)
    yb = np.linspace(-0.05, H + 0.05, res)
    X, Y = np.meshgrid(xb[:-1] + np.diff(xb) / 2, yb[:-1] + np.diff(yb) / 2)
    inside = (Y >= 0) & (Y <= np.sqrt(3) * X) & (Y <= -np.sqrt(3) * (X - 1))
    return xb, yb, X, Y, inside


def group_visits(phase, xb, yb):
    """Number of distinct groups passing through each bin (each group counted once per bin)."""
    n = len(xb) - 1
    ix = np.clip(np.digitize(phase['mx'], xb) - 1, 0, n - 1)
    iy = np.clip(np.digitize(phase['my'], yb) - 1, 0, n - 1)
    visits = pd.DataFrame({'g': phase['unique_group'].values, 'ix': ix, 'iy': iy}).drop_duplicates()
    G = np.zeros((n, n))
    np.add.at(G, (visits['iy'].values, visits['ix'].values), 1)
    return G


def flow_mask(phase, xb, yb, inside):
    """Bins outside the simplex or passed by fewer than FLOW_MIN_SHARE of the groups (at least 2), counting
    visits smoothed over neighbouring bins so the kept region is not broken up by the grid."""
    G = gaussian_filter(group_visits(phase, xb, yb), 1.0)
    return (G < max(2, FLOW_MIN_SHARE * phase['unique_group'].nunique())) | ~inside


def draw_flow(ax, phase, cmap, res=50):
    """Mean change of channel use per generation (streamlines, where at least FLOW_MIN_SHARE of the groups pass)
    over where groups are at the end of the period (shading)."""
    xb, yb, X, Y, inside = simplex_grid(res)
    tri = ternary_frame(ax)
    phase = phase.dropna(subset=['dmx', 'dmy'])
    if len(phase) == 0:
        ax.text(0.5, H / 3, "no groups", ha='center', va='center', color='gray', fontsize=9)
        return
    U = binned_statistic_2d(phase['mx'], phase['my'], phase['dmx'], 'mean', bins=[xb, yb])[0].T
    V = binned_statistic_2d(phase['mx'], phase['my'], phase['dmy'], 'mean', bins=[xb, yb])[0].T
    mask = flow_mask(phase, xb, yb, inside)
    U = ma.array(gaussian_filter(np.nan_to_num(U), 1.5), mask=mask)
    V = ma.array(gaussian_filter(np.nan_to_num(V), 1.5), mask=mask)
    end = phase.groupby('unique_group').last()
    E = gaussian_filter(binned_statistic_2d(end['mx'], end['my'], end['mx'], 'count', bins=[xb, yb])[0].T, 1.5)
    im = ax.imshow(ma.array(E, mask=E < 1e-4), extent=[xb[0], xb[-1], yb[0], yb[-1]], origin='lower',
                   cmap=cmap, alpha=0.65, interpolation='bilinear', zorder=1)
    im.set_clip_path(tri)
    try:
        s = ax.streamplot(X, Y, U, V, color='black', linewidth=0.9, density=1.1, arrowsize=0.9, zorder=2)
        s.lines.set_clip_path(tri)
        s.arrows.set_clip_path(tri)
    except ValueError:
        pass


def ordered_name(figure, fusion, n_gen):
    order_idx = FUSION_ORDER.get(fusion, 9)
    return f"{figure}_{GEN_ORDER[n_gen]}_{n_gen}gen_{order_idx}_{fusion}"


def save(fig, name):
    for out_dir in OUT_DIRS:
        path = os.path.join(out_dir, name) + '.pdf'
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f"Saved {path}")
    if name in MANUSCRIPT_ALIASES:
        ms_path = os.path.join(MANUSCRIPT_DIR, MANUSCRIPT_ALIASES[name])
        os.makedirs(os.path.dirname(ms_path), exist_ok=True)
        fig.savefig(ms_path, bbox_inches='tight', dpi=300)
        print(f"Saved {ms_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Fig. 2: rows = start; A reward of every group, B where groups are,
# C-E channel use of the worst trapped (max 100, if >= 10) and the best (max 100) groups
# ---------------------------------------------------------------------------------------------
def perf_clusters(d):
    """Groups ordered for the heatmap (trapped first, each block by mean median reward, ascending), and the
    clusters as (label, colour, groups): the worst trapped groups (max 100, if >= 10) and the best 100 groups."""
    info = d.groupby('unique_group').agg(mean_perf=(PERF_COL, 'mean'), outcome=('outcome', 'first'))
    info = info.iloc[np.lexsort((info['mean_perf'].values, info['outcome'].values != "Trapped"))]
    clusters = trapped_clusters(d) + [(f"Best {N_CLUSTER}", BEST_COLOR, info['mean_perf'].nlargest(N_CLUSTER).index)]
    return info.index, clusters


def draw_fig2_block(fig, gs_block, df, fusion, n_gen, perf_range, shown, show_col_titles=True, show_letters=True,
                    show_xlabel=True, show_colorbar=True, block_title=None):
    titles = ["Group reward", "Group location"] + [label for _, label in CHANNELS]
    image = None
    ax_h_last = None
    for r, (cond, start) in enumerate(STARTS):
        d = df[(df['social_fusion'] == fusion) & (df['close_target_index'] == cond)]
        order, clusters = perf_clusters(d)
        last = (r == len(STARTS) - 1)
        axes = [fig.add_subplot(gs_block[r, c]) for c in range(5)]

        ax_h = axes[0]
        ax_h_last = ax_h
        heat = d.pivot(index='unique_group', columns='generation', values=PERF_COL).reindex(order)
        image = ax_h.imshow(heat, aspect='auto', cmap=perf_cmap, vmin=perf_range[0], vmax=perf_range[1],
                            interpolation='none', extent=[1, n_gen, 0, len(heat)], origin='lower', rasterized=True)
        box = dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.2)
        row = pd.Series(np.arange(len(heat)), index=heat.index)
        for label, color, groups in clusters:
            if len(groups) < MIN_CLUSTER_GROUPS:
                continue
            y0, y1 = row[groups].min(), row[groups].max() + 1
            for y in (y0, y1):
                if 0 < y < len(heat):
                    ax_h.axhline(y, color='black', linewidth=0.9)
            ax_h.text(n_gen * 0.98, (y0 + y1) / 2, label, ha='right', va='center', fontsize=8.5, color=color,
                      weight='bold', bbox=box)
        ax_h.set_yticks([])
        ax_h.set_ylabel(f"{STARTS_MULTILINE[cond]}\n\nGroups")
        gen_axis(ax_h, n_gen, label=(last and show_xlabel))

        ax_s = axes[1]
        shares = d.groupby('generation')[['frac_on_good', 'frac_on_bad']].mean()
        x = shares.index.values + 1
        ax_s.stackplot(x, shares['frac_on_good'], 1 - shares.sum(axis=1), shares['frac_on_bad'],
                       colors=[STATE_COLORS[k] for k in ['good', 'lost', 'bad']], linewidth=0)
        ax_s.set_ylim(0, 1)
        ax_s.set_ylabel("Share of groups")
        gen_axis(ax_s, n_gen, label=(last and show_xlabel))

        for ax, (col, _) in zip(axes[2:], CHANNELS):
            for _, color, groups in clusters:
                if len(groups) >= MIN_CLUSTER_GROUPS:
                    median_iqr(ax, d[d['unique_group'].isin(groups)], col, color)
                    shown.add(color)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            gen_axis(ax, n_gen, label=(last and show_xlabel))
        axes[2].set_ylabel("Probability")

        if r == 0:
            if show_col_titles:
                for c_idx, a in enumerate(axes):
                    a.set_title(titles[c_idx], fontsize=12)
            if show_letters:
                for c_idx, a in enumerate(axes):
                    panel_letter(a, 'ABCDE'[c_idx])
            if block_title is not None:
                ax_h.text(0.0, 1.18 if show_col_titles else 1.08, block_title, transform=ax_h.transAxes,
                          fontsize=13, weight='bold', va='bottom', ha='left')

    if show_colorbar and ax_h_last is not None and image is not None:
        cax = ax_h_last.inset_axes([0.0, -0.32, 1.0, 0.045])
        cb = fig.colorbar(image, cax=cax, orientation='horizontal')
        cb.set_label("Median reward", fontsize=9, labelpad=1)
        cb.ax.tick_params(labelsize=8)
    return ax_h_last


def fig2(df, fusion, n_gen, perf_range):
    fig = plt.figure(figsize=(19, 6.2))
    gs = gridspec.GridSpec(2, 5, width_ratios=[1.5, 1, 1, 1, 1], hspace=0.12, wspace=0.25)
    shown = set()
    draw_fig2_block(fig, gs, df, fusion, n_gen, perf_range, shown,
                    show_col_titles=True, show_letters=True, show_xlabel=True, show_colorbar=True)
    handles = [Patch(facecolor=STATE_COLORS[k], label=STATE_LABELS[k]) for k in ['good', 'lost', 'bad']]
    handles += [Line2D([0], [0], color=c, linewidth=2.5, label=CLUSTER_NAMES[c])
                for c in [WORST_COLOR, BEST_COLOR] if c in shown]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, bbox_to_anchor=(0.62, 0.035),
               fontsize=9.5, handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
    save(fig, ordered_name("fig2", fusion, n_gen))


def fig2_supp(df, fusions, n_gen, perf_range, out_name=None):
    """Combined supplementary Figure 2 stacked vertically (2 rules per figure for readability)."""
    n_blocks = len(fusions)
    fig = plt.figure(figsize=(19, 5.6 * n_blocks))
    outer = gridspec.GridSpec(n_blocks, 1, hspace=0.28, bottom=0.09, top=0.94)
    shown = set()
    ax_h_bottom = None
    for idx, fusion in enumerate(fusions):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 5, subplot_spec=outer[idx], width_ratios=[1.5, 1, 1, 1, 1], hspace=0.12, wspace=0.25
        )
        is_last = (idx == n_blocks - 1)
        letter = 'ABCD'[idx]
        block_title = f"{letter}   {FUSION_TITLES.get(fusion, fusion)}"
        ax_h_cand = draw_fig2_block(
            fig, inner, df, fusion, n_gen, perf_range, shown,
            show_col_titles=True,
            show_letters=False,
            show_xlabel=is_last,
            show_colorbar=is_last,
            block_title=block_title,
        )
        if is_last:
            ax_h_bottom = ax_h_cand

    handles = [Patch(facecolor=STATE_COLORS[k], label=STATE_LABELS[k]) for k in ['good', 'lost', 'bad']]
    handles += [Line2D([0], [0], color=c, linewidth=2.5, label=CLUSTER_NAMES[c])
                for c in [WORST_COLOR, BEST_COLOR] if c in shown]
    pos = ax_h_bottom.get_position()
    leg_y = pos.y0 - 0.27 * pos.height
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, bbox_to_anchor=(0.62, leg_y),
               fontsize=10.5, handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
    save(fig, out_name or f"fig2_supp_{n_gen}gen")


# ---------------------------------------------------------------------------------------------
# Fig. 3: Start near good target (left) and Start near bad target (right) ternary flow fields
# ---------------------------------------------------------------------------------------------
def draw_fig3_start_panel(fig, sub_spec, df, fusion, n_gen, cond, start_title, panel_letter_str=None,
                          block_title=None, fallback_div_gen=5):
    d = with_steps(df[(df['social_fusion'] == fusion) & (df['close_target_index'] == cond)])
    clusters = trapped_clusters(d)
    best = best_cluster(d)
    has_worst = len(clusters) > 0
    worst = d[d['unique_group'].isin(clusters[0][2])] if has_worst else d.iloc[0:0]

    div_gen = divergence_generation(best, worst) if (has_worst and len(best) > 0) else fallback_div_gen
    a_end = PANEL_A_END if div_gen < PANEL_A_END else n_gen

    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=sub_spec, hspace=0.14, wspace=0.12)
    periods = [(f"Gen 1–{div_gen + 1}", d['generation'] <= div_gen),
               (f"Gen {div_gen + 1}–{a_end}", (d['generation'] >= div_gen) & (d['generation'] < a_end))]

    rows = [
        ("Best", FREE_COLOR, FREE_CMAP, best, True),
        (CLUSTER_NAMES[WORST_COLOR], WORST_COLOR, WORST_CMAP, worst, has_worst),
    ]
    for i, (label, color, cmap, sub, show_row) in enumerate(rows):
        n = sub['unique_group'].nunique() if show_row else 0
        for j, (period, sel) in enumerate(periods):
            ax = fig.add_subplot(inner[i, j])
            if show_row:
                draw_flow(ax, sub[sel.loc[sub.index]], cmap)
                if j == 0:
                    ax.text(-0.14, H / 2, f"{label} ({n})",
                            rotation=90, ha='center', va='center', fontsize=10.5, color=color, weight='bold')
            else:
                ax.axis('off')
                if j == 0:
                    ax.text(1.05, 0.5, f"< {MIN_CLUSTER_GROUPS} trapped groups",
                            transform=ax.transAxes, ha='center', va='center', fontsize=10, color='#777777', style='italic')
            if i == 0:
                ax.set_title(period, fontsize=10.5, pad=4)
            if i == 0 and j == 0:
                if block_title is not None:
                    ax.text(-0.18, H + 0.52, block_title, fontsize=13.5, weight='bold', ha='left', va='bottom')
                header = f"{panel_letter_str}   {start_title}" if panel_letter_str else start_title
                header_size = 12.5 if panel_letter_str else 11.5
                ax.text(-0.18, H + 0.25, header, fontsize=header_size, weight='bold', ha='left', va='bottom')
    return div_gen


def draw_fig3_block(fig, outer_spec, df, fusion, n_gen, block_title=None, show_panel_letters=True):
    block_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_spec, wspace=0.18)
    # Compute bad-start divergence first so good-start can use it as fallback if < 10 trapped
    d_bad = with_steps(df[(df['social_fusion'] == fusion) & (df['close_target_index'] == 1)])
    bad_clusters = trapped_clusters(d_bad)
    bad_best = best_cluster(d_bad)
    bad_div = divergence_generation(bad_best, d_bad[d_bad['unique_group'].isin(bad_clusters[0][2])]) if (bad_clusters and len(bad_best) > 0) else 5

    for idx, (cond, start_label) in enumerate(STARTS):
        letter = 'AB'[idx] if show_panel_letters else None
        draw_fig3_start_panel(fig, block_gs[idx], df, fusion, n_gen, cond, start_label,
                              panel_letter_str=letter,
                              block_title=(block_title if idx == 0 else None),
                              fallback_div_gen=bad_div)


def fig3(df, fusion, n_gen):
    fig = plt.figure(figsize=(14.5, 6.8))
    outer = gridspec.GridSpec(1, 1, bottom=0.10, top=0.88)
    draw_fig3_block(fig, outer[0], df, fusion, n_gen, block_title=None, show_panel_letters=True)
    leg = [Patch(facecolor=plt.get_cmap(FREE_CMAP)(0.6), label=CLUSTER_NAMES[BEST_COLOR]),
           Patch(facecolor=plt.get_cmap(WORST_CMAP)(0.6), label=CLUSTER_NAMES[WORST_COLOR]),
           Line2D([0], [0], color='black', linewidth=1.2, label='Flow')]
    fig.legend(handles=leg, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.08), fontsize=10,
               handlelength=1.6, columnspacing=1.4, handletextpad=0.5)
    save(fig, ordered_name("fig3", fusion, n_gen))


def fig3_supp(df, fusions, n_gen, out_name=None):
    """Combined supplementary Figure 3 stacked vertically (2 rules per figure for readability)."""
    n_blocks = len(fusions)
    fig = plt.figure(figsize=(14.5, 6.1 * n_blocks))
    outer = gridspec.GridSpec(n_blocks, 1, hspace=0.34, bottom=0.07, top=0.91)
    for idx, fusion in enumerate(fusions):
        letter = 'ABCD'[idx]
        block_title = f"{letter}   {FUSION_TITLES.get(fusion, fusion)}"
        draw_fig3_block(fig, outer[idx], df, fusion, n_gen, block_title=block_title, show_panel_letters=False)
    leg = [Patch(facecolor=plt.get_cmap(FREE_CMAP)(0.6), label=CLUSTER_NAMES[BEST_COLOR]),
           Patch(facecolor=plt.get_cmap(WORST_CMAP)(0.6), label=CLUSTER_NAMES[WORST_COLOR]),
           Line2D([0], [0], color='black', linewidth=1.2, label='Flow')]
    fig.legend(handles=leg, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.04), fontsize=11,
               handlelength=1.6, columnspacing=1.5, handletextpad=0.5)
    save(fig, out_name or f"fig3_supp_{n_gen}gen")


# ---------------------------------------------------------------------------------------------
# New Supplementary Figure: Simplex density heatmap snapshots (no flow)
# Rows: each initialization x each social fusion rule; Columns: generations 0, 25, 50, 100, 200, 400, 800
# ---------------------------------------------------------------------------------------------
def fig_simplex_snapshots(df, fusions, n_gen, snap_gens=SNAPSHOT_GENS):
    n_fusions = len(fusions)
    n_cols = len(snap_gens)
    xb = np.linspace(-0.05, 1.05, 55)
    yb = np.linspace(-0.05, H + 0.05, 55)

    # Precompute densities so all panels share a common colour scale
    occ_grid = {}
    global_max = 0.0
    sub_snaps = df[df['generation'].isin(snap_gens)]
    for cond, _ in STARTS:
        for sf in fusions:
            d_sf = sub_snaps[(sub_snaps['close_target_index'] == cond) & (sub_snaps['social_fusion'] == sf)]
            for g in snap_gens:
                dg = d_sf[d_sf['generation'] == g]
                occ = gaussian_filter(
                    binned_statistic_2d(dg['mx'], dg['my'], dg['mx'], 'count', bins=[xb, yb])[0].T, 1.2
                )
                s = occ.sum()
                occ = occ / s if s > 0 else occ
                occ_grid[(cond, sf, g)] = occ
                if occ.max() > global_max:
                    global_max = float(occ.max())

    vmin = global_max * 0.002
    norm = mcolors.LogNorm(vmin=vmin, vmax=global_max)

    fig = plt.figure(figsize=(2.4 * n_cols + 1.4, 2.05 * n_fusions * len(STARTS) + 1.4))
    outer = gridspec.GridSpec(len(STARTS), 1, hspace=0.14, left=0.09, right=0.92, top=0.95, bottom=0.04)

    im_ref = None
    for s_idx, (cond, start_label) in enumerate(STARTS):
        inner = gridspec.GridSpecFromSubplotSpec(n_fusions, n_cols, subplot_spec=outer[s_idx], hspace=0.12, wspace=0.06)
        for r_idx, sf in enumerate(fusions):
            for c_idx, g in enumerate(snap_gens):
                ax = fig.add_subplot(inner[r_idx, c_idx])
                show_vert_labels = (r_idx == n_fusions - 1 and c_idx == 0)
                tri = ternary_frame(ax, fontsize=7.5, show_labels=show_vert_labels)
                occ = occ_grid[(cond, sf, g)]
                im = ax.imshow(
                    ma.array(occ, mask=occ < vmin),
                    extent=[xb[0], xb[-1], yb[0], yb[-1]],
                    origin='lower',
                    cmap='Greys',
                    norm=norm,
                    alpha=0.85,
                    interpolation='bilinear',
                    zorder=1,
                )
                im.set_clip_path(tri)
                im_ref = im

                if r_idx == 0:
                    ax.set_title(f"Gen {g}", fontsize=10, pad=4)
                if c_idx == 0:
                    ax.text(-0.18, H / 2, FUSION_SHORT_LABELS.get(sf, sf),
                            ha='right', va='center', fontsize=9.5, weight='bold')
                if r_idx == 0 and c_idx == 0:
                    letter = 'AB'[s_idx]
                    ax.text(-0.60, H + 0.25, f"{letter}   {start_label}",
                            ha='left', va='bottom', fontsize=12.5, weight='bold')

    if im_ref is not None:
        cax = fig.add_axes([0.935, 0.35, 0.015, 0.30])
        cb = fig.colorbar(im_ref, cax=cax)
        cb.set_label("Group density", fontsize=10)
        cb.ax.tick_params(labelsize=8.5)

    save(fig, f"fig_simplex_snapshots_{n_gen}gen")


for n_gen in args.generations:
    df = load(n_gen)
    perf_range = (df[PERF_COL].min(), df[PERF_COL].max())
    for fusion in args.fusions:
        if "fig2" in args.figures:
            fig2(df, fusion, n_gen, perf_range)
        if "fig3" in args.figures:
            fig3(df, fusion, n_gen)
    if "supp" in args.figures:
        supp_present = [f for f in SUPP_FUSIONS if f in df['social_fusion'].unique()]
        if supp_present:
            supp_parts = [supp_present[i:i + 2] for i in range(0, len(supp_present), 2)]
            for part_idx, part_fusions in enumerate(supp_parts, start=1):
                fig2_supp(df, part_fusions, n_gen, perf_range, out_name=f"fig2_supp_{part_idx}_{n_gen}gen")
                fig3_supp(df, part_fusions, n_gen, out_name=f"fig3_supp_{part_idx}_{n_gen}gen")
    if "snapshots" in args.figures:
        all_present = [f for f in ALL_FUSIONS if f in df['social_fusion'].unique()]
        if all_present:
            fig_simplex_snapshots(df, all_present, n_gen)
print("Done.")
