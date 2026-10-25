"""Paper figures 2 and 3 for every social fusion rule and run length.

fig2: reward of every group, where groups are, and channel use of the worst trapped, the best trapped (if more than
      N_CLUSTER groups are trapped) and the best groups.
fig3 (bad start): A, flow of channel use of not-trapped and the worst trapped groups before and after they diverge;
      B, where all groups spend their time and the flow of the worst and best trapped groups around trap onset.

Data (per group x generation, T = 1000): download_trap_split_wandb_data.py
  --max-ticks 200000 --eval-interval 1000   ->  200 generations
  (defaults)                                 ->  1000 generations
Figures go to figures/paper_figures/.
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

parser = argparse.ArgumentParser(description="Paper figures 2 and 3.")
parser.add_argument("--fusions", nargs="+", default=["naive", "ci", "firsthand"])
parser.add_argument("--generations", nargs="+", type=int, default=[200, 1000])
parser.add_argument("--figures", nargs="+", default=["fig2", "fig3"])
args = parser.parse_args()

wandb_dir = os.environ.get("WANDB_DIR", "./")
DATASETS = {200: "trap_split_T1000_200gen", 1000: "trap_split_T1000_long"}
OUT = os.path.join('figures', 'paper_figures')
os.makedirs(OUT, exist_ok=True)

# Trap rule: a group is trapped if its median reward, averaged over the last TRAP_WINDOW generations,
# is below TRAP_REWARD (valley between the two modes of the final reward).
PERF_COL = 'median_avg_fitness'
TRAP_REWARD = 0.7
TRAP_WINDOW = 20
# Trap onset: first generation of the final unbroken stretch in which the centred ONSET_SMOOTH-generation
# average of the median reward is below TRAP_REWARD (0 = never left the bad target)
ONSET_SMOOTH = 10

# Clusters, ranked by mean median reward over the run: the worst N_CLUSTER trapped groups (all trapped groups if fewer),
# the best N_CLUSTER trapped groups not among them (if more than N_CLUSTER are trapped) and the best N_CLUSTER groups
# overall (fig. 2 only)
N_CLUSTER = 100
WORST_COLOR, BEST_TRAPPED_COLOR, BEST_COLOR = '#a3135f', '#2c7bb6', '#4d9221'
CLUSTER_NAMES = {WORST_COLOR: "Worst trapped", BEST_TRAPPED_COLOR: "Best trapped", BEST_COLOR: f"Best {N_CLUSTER}"}

# Fig. 3
PANEL_A_END = 100       # panel A: generations 1-divergence and divergence-PANEL_A_END
DIVERGENCE = 0.05       # sets diverge when their mean channel use differs by more than this in the simplex
ONSET_SPAN = (5, 200)   # panel B: generations before / after onset used for each trapped group's flow
FLOW_DENSITY_SCALE = 0.3   # streamline density = scale / sqrt(share of the triangle the cluster's flow covers)
FLOW_MIN_SHARE = 0.02     # fig. 3: draw the flow only where at least this share of the set's groups pass
MIN_CLUSTER_GROUPS = 3

# File names sort as 200 then 1000 generations, and naive -> ci -> first-hand
GEN_ORDER = {200: 1, 1000: 2}
FUSION_ORDER = {"naive": 1, "ci": 2, "firsthand": 3}

STARTS = [(0, "Good start"), (1, "Bad start")]
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
    df['unique_group'] = df['run_id'] + '_' + df['group_id'].astype(str)
    df['mx'] = df['prob_none'] + 0.5 * df['prob_soc']
    df['my'] = H * df['prob_soc']
    trapped = pd.concat([d.pivot(index='unique_group', columns='generation', values=PERF_COL)
                         .iloc[:, -TRAP_WINDOW:].mean(axis=1) < TRAP_REWARD for _, d in df.groupby('run_id')])
    df['outcome'] = np.where(trapped.reindex(df['unique_group']).values, "Trapped", "Not trapped")
    return df


def with_steps(d):
    d = d.sort_values(['unique_group', 'generation']).copy()
    d['dmx'] = d.groupby('unique_group')['mx'].diff()
    d['dmy'] = d.groupby('unique_group')['my'].diff()
    return d


def trap_onset(d):
    """Per trapped group: first generation of its final trapped stretch (0 if it was never free)."""
    perf = d[d['outcome'] == "Trapped"].pivot(index='unique_group', columns='generation', values=PERF_COL)
    free = ~(perf.T.rolling(ONSET_SMOOTH, center=True, min_periods=1).mean().T < TRAP_REWARD)
    return free.apply(lambda r: r[r].index.max() + 1 if r.any() else 0, axis=1)


def trapped_clusters(d):
    """Trapped groups by mean median reward as (label, colour, groups): the worst N_CLUSTER and the best N_CLUSTER
    not among them."""
    perf = d[d['outcome'] == "Trapped"].groupby('unique_group')[PERF_COL].mean().sort_values()
    n = len(perf)
    n_worst = min(N_CLUSTER, n)
    clusters = [(f"Worst {n_worst} trapped", WORST_COLOR, perf.index[:n_worst])]
    n_best = min(N_CLUSTER, n - n_worst)
    if n_best > 0:
        clusters.append((f"Best {n_best} trapped", BEST_TRAPPED_COLOR, perf.index[n - n_best:]))
    return clusters


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


def ternary_frame(ax, fontsize=9):
    tri = Polygon([[0, 0], [1, 0], [0.5, H]], fill=False, edgecolor='black', linewidth=1.5, zorder=4)
    ax.add_patch(tri)
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
    return f"{figure}_{GEN_ORDER[n_gen]}_{n_gen}gen_{FUSION_ORDER[fusion]}_{fusion}"


def save(fig, name):
    path = os.path.join(OUT, name) + '.pdf'
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------------------------
# Fig. 2: rows = start; A reward of every group, B where groups are,
# C-E channel use of the worst trapped, the best trapped and the best groups
# ---------------------------------------------------------------------------------------------
def perf_clusters(d):
    """Groups ordered for the heatmap (trapped first, each block by mean median reward, ascending), and the
    clusters as (label, colour, groups): the worst and best trapped groups and the best groups."""
    info = d.groupby('unique_group').agg(mean_perf=(PERF_COL, 'mean'), outcome=('outcome', 'first'))
    info = info.iloc[np.lexsort((info['mean_perf'].values, info['outcome'].values != "Trapped"))]
    clusters = trapped_clusters(d) + [(f"Best {N_CLUSTER}", BEST_COLOR, info['mean_perf'].nlargest(N_CLUSTER).index)]
    return info.index, clusters


def fig2(df, fusion, n_gen, perf_range):
    fig = plt.figure(figsize=(19, 6.2))
    gs = gridspec.GridSpec(2, 5, width_ratios=[1.5, 1, 1, 1, 1], hspace=0.12, wspace=0.25)
    titles = ["Group reward", "Group location"] + [label for _, label in CHANNELS]
    shown = set()  # colours of the clusters drawn in any row
    for r, (cond, start) in enumerate(STARTS):
        d = df[(df['social_fusion'] == fusion) & (df['close_target_index'] == cond)]
        order, clusters = perf_clusters(d)
        last = r == len(STARTS) - 1
        axes = [fig.add_subplot(gs[r, c]) for c in range(5)]

        ax_h = axes[0]
        heat = d.pivot(index='unique_group', columns='generation', values=PERF_COL).reindex(order)
        image = ax_h.imshow(heat, aspect='auto', cmap=perf_cmap, vmin=perf_range[0], vmax=perf_range[1],
                            interpolation='none', extent=[1, n_gen, 0, len(heat)], origin='lower', rasterized=True)
        box = dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.2)
        row = pd.Series(np.arange(len(heat)), index=heat.index)
        # Each cluster is a contiguous block of rows: worst trapped at the bottom, best trapped at the top of the
        # trapped block, best groups at the top
        for label, color, groups in clusters:
            if len(groups) == 0:
                continue
            y0, y1 = row[groups].min(), row[groups].max() + 1
            for y in (y0, y1):
                if 0 < y < len(heat):
                    ax_h.axhline(y, color='black', linewidth=0.9)
            ax_h.text(n_gen * 0.98, (y0 + y1) / 2, label, ha='right', va='center', fontsize=8.5, color=color,
                      weight='bold', bbox=box)
        ax_h.set_yticks([])
        ax_h.set_ylabel(f"{start}\n\nGroups")
        gen_axis(ax_h, n_gen, label=last)

        ax_s = axes[1]
        shares = d.groupby('generation')[['frac_on_good', 'frac_on_bad']].mean()
        x = shares.index.values + 1
        ax_s.stackplot(x, shares['frac_on_good'], 1 - shares.sum(axis=1), shares['frac_on_bad'],
                       colors=[STATE_COLORS[k] for k in ['good', 'lost', 'bad']], linewidth=0)
        ax_s.set_ylim(0, 1)
        ax_s.set_ylabel("Share of groups")
        gen_axis(ax_s, n_gen, label=last)

        for ax, (col, _) in zip(axes[2:], CHANNELS):
            for _, color, groups in clusters:
                if len(groups) >= MIN_CLUSTER_GROUPS:
                    median_iqr(ax, d[d['unique_group'].isin(groups)], col, color)
                    shown.add(color)
            ax.set_ylim(0, 0.5 if col == 'prob_priv' else 1)
            ax.grid(True, alpha=0.3)
            gen_axis(ax, n_gen, label=last)
        axes[2].set_ylabel("Probability")

        if r == 0:
            for c_idx, a in enumerate(axes):
                a.set_title(titles[c_idx], fontsize=12)
                panel_letter(a, 'ABCDE'[c_idx])
        if last:
            cax = ax_h.inset_axes([0.0, -0.3, 1.0, 0.045])
            cb = fig.colorbar(image, cax=cax, orientation='horizontal')
            cb.set_label("Median reward", fontsize=9, labelpad=1)
            cb.ax.tick_params(labelsize=8)

    handles = [Patch(facecolor=STATE_COLORS[k], label=STATE_LABELS[k]) for k in ['good', 'lost', 'bad']]
    handles += [Line2D([0], [0], color=c, linewidth=2.5, label=CLUSTER_NAMES[c])
                for c in [WORST_COLOR, BEST_TRAPPED_COLOR, BEST_COLOR] if c in shown]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, bbox_to_anchor=(0.62, 0.035),
               fontsize=9.5, handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
    save(fig, ordered_name("fig2", fusion, n_gen))


# ---------------------------------------------------------------------------------------------
# Fig. 3 (bad start). A: flow of not-trapped and the worst trapped groups before and after they diverge.
# B: where all groups spend their time, and the flow of the worst and best trapped groups around trap onset.
# ---------------------------------------------------------------------------------------------
def cohort_flow(ax, sub, color, tri, res=36):
    """Flow field of one cluster, drawn in a single colour wherever at least FLOW_MIN_SHARE of the cluster's
    groups pass; fewer lines where the flow covers much of the triangle, more where it is confined to a strip.
    Returns whether any line was drawn."""
    xb, yb, X, Y, inside = simplex_grid(res)
    sub = sub.dropna(subset=['dmx', 'dmy'])
    count = binned_statistic_2d(sub['mx'], sub['my'], sub['dmx'], 'count', bins=[xb, yb])[0].T
    N = gaussian_filter(count, 1.2)
    sm = lambda col: gaussian_filter(np.nan_to_num(
        binned_statistic_2d(sub['mx'], sub['my'], sub[col], 'mean', bins=[xb, yb])[0].T) * count, 1.2) / np.maximum(N, 1e-9)
    mask = flow_mask(sub, xb, yb, inside)
    covered = (~mask).sum() / inside.sum()
    density = float(np.clip(FLOW_DENSITY_SCALE / np.sqrt(max(covered, 1e-3)), 0.4, 3.0))
    try:
        s = ax.streamplot(X, Y, ma.array(sm('dmx'), mask=mask), ma.array(sm('dmy'), mask=mask), color=color,
                          linewidth=1.5, density=density, arrowsize=1.2, zorder=4)
        s.lines.set_clip_path(tri)
        s.arrows.set_clip_path(tri)
        return len(s.lines.get_segments()) > 0
    except ValueError:
        return False


def fig3(df, fusion, n_gen, cond=1):
    d = with_steps(df[(df['social_fusion'] == fusion) & (df['close_target_index'] == cond)])
    clusters = trapped_clusters(d)
    free, worst = d[d['outcome'] == "Not trapped"], d[d['unique_group'].isin(clusters[0][2])]
    if len(clusters[0][2]) == 0 or len(free) == 0:
        print(f"  {fusion} {n_gen} gen: only one outcome, skipping fig3")
        return
    rows = [("Not trapped", FREE_COLOR, FREE_CMAP, free), (CLUSTER_NAMES[WORST_COLOR], WORST_COLOR, WORST_CMAP, worst)]
    div_gen = divergence_generation(free, worst)
    onset = trap_onset(d).clip(upper=n_gen - 1)
    a_end = PANEL_A_END if div_gen < PANEL_A_END else n_gen

    fig = plt.figure(figsize=(15, 7.2))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.08)

    # A: rows = not trapped / worst trapped, columns = before / after divergence
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0.12, wspace=0.12)
    periods = [(f"Gen 1–{div_gen + 1}", d['generation'] <= div_gen),
               (f"Gen {div_gen + 1}–{a_end}", (d['generation'] >= div_gen) & (d['generation'] < a_end))]
    for i, (label, color, cmap, sub) in enumerate(rows):
        n = sub['unique_group'].nunique()
        for j, (period, sel) in enumerate(periods):
            ax = fig.add_subplot(inner[i, j])
            draw_flow(ax, sub[sel.loc[sub.index]], cmap)
            if i == 0:
                ax.set_title(period, fontsize=11)
            if j == 0:
                ax.text(-0.14, H / 2, f"{label} ({n})",
                        rotation=90, ha='center', va='center', fontsize=11, color=color, weight='bold')
            if i == 0 and j == 0:
                ax.text(-0.2, H + 0.25, "A", fontsize=15, weight='bold')

    # B: time spent by all groups, and the flow of trapped groups around their onset, one colour per cluster
    ax_b = fig.add_subplot(outer[1])
    tri = ternary_frame(ax_b, fontsize=12)
    xb = np.linspace(-0.05, 1.05, 60)
    yb = np.linspace(-0.05, H + 0.05, 60)
    occ = gaussian_filter(binned_statistic_2d(d['mx'], d['my'], d['mx'], 'count', bins=[xb, yb])[0].T, 1.0)
    occ = occ / occ.sum()
    im = ax_b.imshow(ma.array(occ, mask=occ < occ.max() * 0.001), extent=[xb[0], xb[-1], yb[0], yb[-1]],
                     origin='lower', cmap='Greys', norm=mcolors.LogNorm(occ.max() * 0.001, occ.max()),
                     alpha=0.6, interpolation='bilinear', zorder=1)
    im.set_clip_path(tri)

    handles = []
    for label, color, groups in clusters:
        drawn = False
        if len(groups) >= MIN_CLUSTER_GROUPS:
            sub = d[d['unique_group'].isin(groups)].copy()
            sub['rel'] = sub['generation'] - sub['unique_group'].map(onset)
            drawn = cohort_flow(ax_b, sub[sub['rel'].between(-ONSET_SPAN[0], ONSET_SPAN[1])], color, tri)
        handles.append(Line2D([0], [0], color=color, linewidth=2.5, alpha=1 if drawn else 0.3,
                              label=f"{CLUSTER_NAMES[color]} ({len(groups)})"))
    ax_b.legend(handles=handles, loc='upper right', frameon=False, fontsize=9.5,
                title=f"Trapped, gen −{ONSET_SPAN[0]} to +{ONSET_SPAN[1]}\naround trap onset",
                title_fontsize=10, bbox_to_anchor=(1.04, 1.0))
    ax_b.text(-0.08, H + 0.06, "B", fontsize=15, weight='bold')
    cax = ax_b.inset_axes([0.05, 0.62, 0.025, 0.3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Time spent, all groups", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    leg = [Patch(facecolor=plt.get_cmap(FREE_CMAP)(0.6), label="Not trapped"),
           Patch(facecolor=plt.get_cmap(WORST_CMAP)(0.6), label=CLUSTER_NAMES[WORST_COLOR])]
    leg += [Line2D([0], [0], color='black', linewidth=1.2, label='Flow')]
    fig.legend(handles=leg, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.29, 0.13), fontsize=9.5,
               handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
    save(fig, ordered_name("fig3", fusion, n_gen))


for n_gen in args.generations:
    df = load(n_gen)
    perf_range = (df[PERF_COL].min(), df[PERF_COL].max())
    for fusion in args.fusions:
        if "fig2" in args.figures:
            fig2(df, fusion, n_gen, perf_range)
        if "fig3" in args.figures:
            fig3(df, fusion, n_gen)
print("Done.")
