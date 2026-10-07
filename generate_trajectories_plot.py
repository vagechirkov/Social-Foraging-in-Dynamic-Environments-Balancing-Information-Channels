import os
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import numpy.ma as ma
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Polygon
from sklearn.cluster import KMeans
from scipy.special import softmax
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Construct the path using WANDB_DIR
wandb_dir = os.environ.get("WANDB_DIR", "./")
data_path = os.path.join(wandb_dir, "data", "v9arred3_jmrbbem4", "merged_agent_stats.parquet")
dist_data_path = os.path.join(wandb_dir, "data", "v9arred3_jmrbbem4", "merged_group_distances.parquet")

print(f"Loading data from {data_path}...")
try:
    # Load the parquet file (much faster for large datasets)
    df = pd.read_parquet(data_path)
except FileNotFoundError:
    # Fallback to the CSV if parquet isn't available for some reason
    csv_path = data_path.replace(".parquet", ".csv.gz")
    print(f"Parquet not found. Falling back to CSV: {csv_path}")
    df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")

# Ensure group_id is always an integer to avoid '0.0' vs '0' mismatch
if 'group_id' in df.columns:
    df['group_id'] = df['group_id'].astype(float).astype(int)

# Filter for the requested condition
# df_flow = df[df['close_target_index'] == 1].copy()
df_flow = df.copy()

# Determine success for each group AT EACH STEP
# If any agent in the group reached current_fitness > 0.8 at this specific step
group_success = df_flow.groupby(['run_id', 'group_id', 'step'])['current_fitness'].max() > 0.85
group_success.name = 'success'
df_flow = df_flow.merge(group_success, on=['run_id', 'group_id', 'step'])

# 1. Compute softmax probabilities from the raw gene values
genes_cols = ['gene_priv', 'gene_soc', 'gene_none']
probs = softmax(df_flow[genes_cols].values, axis=1)

# Assign probabilities back to the dataframe
df_flow['prob_priv'] = probs[:, 0]
df_flow['prob_soc'] = probs[:, 1]
df_flow['prob_none'] = probs[:, 2]

# 2. Aggregate to group-level step data
# We include 'success' so it's available for splitting the flow fields
group_cols = ['run_id', 'group_id', 'step', 'success']
df_group_steps = df_flow.groupby(group_cols)[['prob_priv', 'prob_soc', 'prob_none', 'gene_eve']].mean().reset_index()

# Ensure chronological ordering
df_group_steps = df_group_steps.sort_values(['run_id', 'group_id', 'step'])
df_gs_all = df_group_steps.copy()

print("Calculating raw median performance per group from df...")
group_perf = df.groupby(['run_id', 'group_id', 'step', 'close_target_index'])['current_fitness'].median().reset_index()
group_perf = group_perf[group_perf['step'] < 199000]
group_perf['unique_group'] = group_perf['run_id'].astype(str) + '_' + group_perf['group_id'].astype(str)

import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, Rectangle
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import numpy.ma as ma
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns


plot_version = 'v5' # 'v4_10k_clust'
def draw_streamplot(ax, data, cmap, title, H=np.sqrt(3)/2):
    triangle = Polygon([[0, 0], [1, 0], [0.5, H]], fill=False, edgecolor='black', linewidth=2.5, zorder=4)
    ax.add_patch(triangle)

    grid_res = 50
    x_bins = np.linspace(-0.05, 1.05, grid_res)
    y_bins = np.linspace(-0.05, H + 0.05, grid_res)
    
    X, Y = np.meshgrid(x_bins[:-1] + np.diff(x_bins)/2, y_bins[:-1] + np.diff(y_bins)/2)
    
    if len(data) >= 5 and 'dmx' in data.columns:
        U, _, _, _ = binned_statistic_2d(data['mx'], data['my'], data['dmx'], statistic='mean', bins=[x_bins, y_bins])
        V, _, _, _ = binned_statistic_2d(data['mx'], data['my'], data['dmy'], statistic='mean', bins=[x_bins, y_bins])
        Counts, _, _, _ = binned_statistic_2d(data['mx'], data['my'], data['dmx'], statistic='count', bins=[x_bins, y_bins])

        U, V, Counts = U.T, V.T, Counts.T

        U_smooth = gaussian_filter(np.nan_to_num(U, 0), sigma=1.0)
        V_smooth = gaussian_filter(np.nan_to_num(V, 0), sigma=1.0)

        mask = Counts < 2
        U_masked = ma.array(U_smooth, mask=mask)
        V_masked = ma.array(V_smooth, mask=mask)

        density = gaussian_filter(Counts, sigma=1.5)
        density_masked = ma.array(density, mask=mask)

        try:
            ax.streamplot(X, Y, U_masked, V_masked, color=density_masked, cmap=cmap, 
                          linewidth=1.2, density=1.2, arrowsize=1.2, zorder=2)
        except ValueError:
            pass

    ax.text(0.5, H + 0.02, "Social", ha='center', va='bottom', fontsize=16, weight='bold')
    ax.text(0.1, -0.02, "Private", ha='right', va='top', fontsize=16, weight='bold')
    ax.text(0.9, -0.02, "None", ha='left', va='top', fontsize=16, weight='bold')

    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=24, weight='bold', pad=25)

n_clusters = 6
palette = sns.color_palette("coolwarm_r", as_cmap=True)
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom', ['#E69F00', '#0072B2'])

print("Generating clustered trajectories based on distances to targets over time...")

try:
    df_dist = pd.read_parquet(dist_data_path)
except FileNotFoundError:
    print(f"Distances data not found at {dist_data_path}. Falling back to CSV or skipping.")
    try:
        df_dist = pd.read_csv(dist_data_path.replace(".parquet", ".csv.gz"))
    except FileNotFoundError:
        df_dist = None

if df_dist is not None:
    if 'group_id' in df_dist.columns:
        df_dist['group_id'] = df_dist['group_id'].astype(float).astype(int)
    df_dist['unique_group'] = df_dist['run_id'].astype(str) + '_' + df_dist['group_id'].astype(str)

conditions = sorted(group_perf['close_target_index'].unique())
cluster_groups_by_cond = {}

for i, cond in enumerate(conditions):
    cond_data = group_perf[group_perf['close_target_index'] == cond].copy()
    
    heatmap_data = cond_data.pivot(index='unique_group', columns='step', values='current_fitness')
    
    # === PREVIOUS CLUSTERING CODE (based on median reward) ===
    # X_cluster = heatmap_data.fillna(0).values
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_cluster)
    # cluster_labels = kmeans.labels_
    # ==========================================================
    
    if df_dist is not None and plot_version != 'v5':
        # dist_good = dist_cond_data.pivot(index='unique_group', columns='tick', values='mean_good_dist')
        # dist_bad = dist_cond_data.pivot(index='unique_group', columns='tick', values='mean_bad_dist')
        
        # # Forward fill in case of missing ticks, then fill remaining NAs with 0
        # dist_good = dist_good.ffill(axis=1).fillna(0)
        # dist_bad = dist_bad.ffill(axis=1).fillna(0)
        
        # # Align rows exactly with heatmap_data
        # dist_good = dist_good.reindex(heatmap_data.index).fillna(0)
        # dist_bad = dist_bad.reindex(heatmap_data.index).fillna(0)

        dist_cond_data = df_dist[df_dist['unique_group'].isin(heatmap_data.index)].copy()
        
        # 1. Filter to first 100k ticks
        dist_100k = dist_cond_data[dist_cond_data['tick'] <= 10_000].copy()
        
        # 2. Extract features
        dist_mean = dist_100k.groupby('unique_group')[['mean_good_dist', 'mean_bad_dist']].mean()
        dist_mean.columns = ['mean_good', 'mean_bad']
        
        # Convergence point (mean over 80k-100k)
        dist_late = dist_100k[dist_100k['tick'] >= 8000].groupby('unique_group')[['mean_good_dist', 'mean_bad_dist']].mean()
        dist_late.columns = ['late_good', 'late_bad']
        
        # Overall standard deviation
        dist_std = dist_100k.groupby('unique_group')[['mean_good_dist', 'mean_bad_dist']].std()
        dist_std.columns = ['std_good', 'std_bad']
        
        # Oscillation proxy (std of tick-to-tick changes)
        dist_100k = dist_100k.sort_values(['unique_group', 'tick'])
        dist_100k['diff_good'] = dist_100k.groupby('unique_group')['mean_good_dist'].diff()
        dist_100k['diff_bad'] = dist_100k.groupby('unique_group')['mean_bad_dist'].diff()
        dist_osc = dist_100k.groupby('unique_group')[['diff_good', 'diff_bad']].std()
        dist_osc.columns = ['osc_good', 'osc_bad']
        
        # Combine features
        features = dist_mean.join(dist_std) # .join(dist_late).join(dist_osc)
        features = features.reindex(heatmap_data.index).fillna(0)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        X_cluster = StandardScaler().fit_transform(features.values)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_cluster)
        cluster_labels = kmeans.labels_
    else:
        # Fallback if no distance data
        X_cluster = heatmap_data.fillna(0).values[:, :100_000]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_cluster)
        cluster_labels = kmeans.labels_

    group_mean_perf = heatmap_data.mean(axis=1).values
    cluster_avg_perf = {c: group_mean_perf[cluster_labels == c].mean() for c in range(n_clusters)}
    sorted_clusters = sorted(cluster_avg_perf.keys(), key=lambda c: cluster_avg_perf[c])

    sorted_indices = []
    cluster_boundaries = []
    current_y = 0
    
    cluster_groups = []
    
    for c in sorted_clusters:
        idx_in_cluster = np.where(cluster_labels == c)[0]
        sorted_idx_in_cluster = idx_in_cluster[np.argsort(group_mean_perf[idx_in_cluster])]
        sorted_indices.extend(sorted_idx_in_cluster)
        
        size = len(idx_in_cluster)
        cluster_boundaries.append((current_y, current_y + size))
        current_y += size
        
        groups = heatmap_data.index[sorted_idx_in_cluster].tolist()
        cluster_groups.append(groups)
    
    cluster_groups_by_cond[cond] = cluster_groups

    heatmap_data = heatmap_data.iloc[sorted_indices]
    heatmap_cols = heatmap_data.columns
    
    fig = plt.figure(figsize=(38, 20))
    gs = gridspec.GridSpec(4, n_clusters, height_ratios=[2, 0.3, 1, 1], hspace=0.05, wspace=0.05)
    
    ax_main = fig.add_subplot(gs[0, :])

    extent = [heatmap_cols.min()/1000, heatmap_cols.max()/1000, 0, heatmap_data.shape[0]]
    
    custom_perf_cmap = mcolors.LinearSegmentedColormap.from_list('custom_perf', [
        '#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7',
        '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419'
    ])
    
    cax = ax_main.imshow(
        heatmap_data, 
        aspect='auto', 
        cmap=custom_perf_cmap, 
        interpolation='none', 
        extent=extent, 
        origin='lower', 
        rasterized=True
        )

    ax_main.set_xlabel("Generation (1k steps)", fontsize=32, labelpad=0)
    # custom_ticks now stops before 200
    custom_ticks = [1] + list(np.arange(25, 200, 25))
    ax_main.set_xticks(custom_ticks)
    ax_main.set_xticklabels([str(x) for x in custom_ticks])
    ax_main.tick_params(labelsize=24)
    ax_main.axvline(x=100, color='white', linestyle=':', linewidth=3)

    title_str = "Good Target Initialisation" if cond == 0 else "Bad Target Initialisation"
    ax_main.set_yticks([])
    ax_main.set_ylabel(f"Groups {title_str}", fontsize=28, labelpad=70) # , weight='bold'  (Clustered & Sorted)

    for idx, (y_min, y_max) in enumerate(cluster_boundaries):
        rect = Rectangle((extent[0], y_min), extent[1]-extent[0], y_max-y_min, 
                         linewidth=6, edgecolor='black', facecolor='none', zorder=5)
        ax_main.add_patch(rect)
        
        ax_main.text(extent[0] - 2, (y_min + y_max)/2, f'{n_clusters-idx}', 
                     color='black', fontsize=36, weight='bold', va='center', ha='right',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    cbar_ax = fig.add_axes([0.91, 0.53, 0.015, 0.35])
    
    cbar = fig.colorbar(cax, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=28, length=5)
    cbar.set_label('Median Performance', fontsize=32, labelpad=5) #  , weight='bold'

    info_genes = ['prob_priv', 'prob_soc', 'prob_none']
    info_labels = ['Private', 'Social', 'None']
    info_colors = ['#0072B2', '#E69F00', '#009E73'] 
    
    if 'gene_eve_scaled' not in df_gs_all.columns:
        df_gs_all['gene_eve_scaled'] = (1 / (1 + np.exp(-df_gs_all['gene_eve']))) * (0.1 - 0.01) + 0.01
    noise_gene = 'gene_eve_scaled'
    noise_label = 'Process Noise'
    noise_color = '#CC79A7' 
    
    if 'unique_group' not in df_gs_all.columns:
        df_gs_all['unique_group'] = df_gs_all['run_id'].astype(str) + '_' + df_gs_all['group_id'].astype(str)
        
    for cluster_col, (cluster_idx, groups) in enumerate(reversed(list(enumerate(cluster_groups)))):
        ax_info = fig.add_subplot(gs[2, cluster_col])
        ax_noise = fig.add_subplot(gs[3, cluster_col])
        # ax_flow = fig.add_subplot(gs[4, cluster_col])
        
        cluster_df = df_gs_all[df_gs_all['unique_group'].isin(groups)]
        
        for gene_idx, gene_name in enumerate(info_genes):
            color = info_colors[gene_idx]
            step_agg = cluster_df.groupby('step')[gene_name].agg(['mean', 'std']).reset_index()
            x_vals = (step_agg['step'] + 1) / 1000
            y_mean = step_agg['mean'].values
            y_std = step_agg['std'].values
            
            ax_info.plot(x_vals, y_mean, color=color, linewidth=2, label=info_labels[gene_idx])
            ax_info.fill_between(x_vals, np.clip(y_mean - y_std, 0, 1), np.clip(y_mean + y_std, 0, 1), color=color, alpha=0.2)
            
        ax_info.set_title(f'Cluster {cluster_col+1}', fontsize=28, weight='bold', color='black')
        ax_info.set_ylim(-0.05, 1.05)
        ax_info.tick_params(labelsize=18)
        ax_info.set_xticks([1] + list(np.arange(50, 250, 50)))
        ax_info.set_xticklabels([])
        ax_info.axvline(x=100, color='gray', linestyle=':', linewidth=2)
        ax_info.grid(True, linestyle='--', alpha=0.5)
        
        if cluster_col == 0:
            ax_info.set_ylabel("Probabilities", fontsize=22)
            ax_info.legend(fontsize=20, loc='best')
        else:
            ax_info.set_yticklabels([])
            
        step_agg = cluster_df.groupby('step')[noise_gene].agg(['mean', 'std']).reset_index()
        if not step_agg.empty:
            x_vals = (step_agg['step'] + 1) / 1000
            y_mean = step_agg['mean'].values
            y_std = step_agg['std'].values
            
            ax_noise.plot(x_vals, y_mean, color=noise_color, linewidth=2, label=noise_label)
            ax_noise.fill_between(x_vals, np.clip(y_mean - y_std, 0.04, 0.09), np.clip(y_mean + y_std, 0.04, 0.09), color=noise_color, alpha=0.2)
        
        ax_noise.set_ylim(0.04, 0.09)
        ax_noise.set_xlabel("Generation (1k steps)", fontsize=22)
        ax_noise.tick_params(labelsize=18)
        ax_noise.set_xticks([1] + list(np.arange(50, 250, 50)))
        ax_noise.axvline(x=100, color='gray', linestyle=':', linewidth=2)
        ax_noise.grid(True, linestyle='--', alpha=0.5)
        
        if cluster_col == 0:
            ax_noise.set_ylabel("Process Noise", fontsize=22)
        else:
            ax_noise.set_yticklabels([])

        # Flow Diagrams (Row 4)
        # flow_df = cluster_df[cluster_df['step'] < 7000].copy()
        # if len(flow_df) > 0:
        #     if 'mx' not in flow_df.columns:
        #         H = np.sqrt(3)/2
        #         flow_df['mx'] = flow_df['prob_none'] + 0.5 * flow_df['prob_soc']
        #         flow_df['my'] = H * flow_df['prob_soc']
        #     
        #     flow_df = flow_df.sort_values(['run_id', 'group_id', 'step'])
        #     flow_df['dmx'] = flow_df.groupby(['run_id', 'group_id'])['mx'].diff()
        #     flow_df['dmy'] = flow_df.groupby(['run_id', 'group_id'])['my'].diff()
        #     flow_df = flow_df.dropna(subset=['dmx', 'dmy'])
        #     
        # draw_streamplot(ax_flow, flow_df, 'Blues', 'Flow Gen 1-7')

    os.makedirs('figures', exist_ok=True)
    filename = f'figures/clustered_trajectories_gene_ts_perf_{"good" if cond == 0 else "bad"}_{plot_version}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    # # plt.show()
    plt.close(fig)


# Separate Flow Diagram for Good Target Initialization (using Bad Target separatrix)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from sklearn.svm import LinearSVC
import numpy.ma as ma
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("Inferring Divergence Generation and Separatrix...")

if 'mx' not in df_gs_all.columns:
    H = np.sqrt(3)/2
    df_gs_all['mx'] = df_gs_all['prob_none'] + 0.5 * df_gs_all['prob_soc']
    df_gs_all['my'] = H * df_gs_all['prob_soc']

infer_separately = True

def infer_separatrix_for_cond(cond_val):
    target_best_groups = cluster_groups_by_cond[cond_val][5]
    target_worst_groups = cluster_groups_by_cond[cond_val][0]
    
    target_best_df = df_gs_all[df_gs_all['unique_group'].isin(target_best_groups)]
    target_worst_df = df_gs_all[df_gs_all['unique_group'].isin(target_worst_groups)]
    
    best_means = target_best_df.groupby('step')[['mx', 'my']].mean()
    worst_means = target_worst_df.groupby('step')[['mx', 'my']].mean()
    merged_means = best_means.join(worst_means, lsuffix='_best', rsuffix='_worst').dropna()
    merged_means['dist'] = np.sqrt((merged_means['mx_best'] - merged_means['mx_worst'])**2 + (merged_means['my_best'] - merged_means['my_worst'])**2)
    
    div_step = merged_means[merged_means['dist'] > 0.05].index.min()
    div_gen = (div_step + 1) / 1000 if div_step is not pd.NA else 6.0
    
    post_best = target_best_df[target_best_df['step'] >= div_step][['mx', 'my']].dropna()
    post_worst = target_worst_df[target_worst_df['step'] >= div_step][['mx', 'my']].dropna()
    X_svm = np.vstack([post_best.values, post_worst.values])
    y_svm = np.array([1]*len(post_best) + [0]*len(post_worst))
    
    svm = LinearSVC(random_state=42, C=1.0)
    svm.fit(X_svm, y_svm)
    return div_step, div_gen, svm.coef_[0], svm.intercept_[0]

if not infer_separately:
    # Use Bad Target for inference globally
    diverge_step, diverge_gen, w, b = infer_separatrix_for_cond(1)
    print(f"Divergence Generation (from Bad Target): {diverge_gen}")
    print(f"Separatrix equation: {w[0]:.4f} * mx + {w[1]:.4f} * my + {b:.4f} = 0")

# Use Good Target and Bad Target for plotting flow diagrams
for target_type, cond_val in [('good', 0), ('bad', 1)]:
    if infer_separately:
        diverge_step, diverge_gen, w, b = infer_separatrix_for_cond(cond_val)
        print(f"Divergence Generation (from {target_type.capitalize()} Target): {diverge_gen}")
        print(f"Separatrix equation: {w[0]:.4f} * mx + {w[1]:.4f} * my + {b:.4f} = 0")
    target_best_groups = cluster_groups_by_cond[cond_val][5]
    target_worst_groups = cluster_groups_by_cond[cond_val][0]

    target_clusters = [
        ("Cluster 1 (Best)", target_best_groups, 'Blues'),
        ("Cluster 6 (Worst)", target_worst_groups, 'Reds')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    H = np.sqrt(3)/2

    periods = [
        (f'Gen 0-{int(diverge_gen)} (Pre-Divergence)', lambda d: d['step'] < diverge_step),
        (f'Gen {int(diverge_gen)}-100 (Post-Divergence)', lambda d: d['step'] >= diverge_step)
    ]

    grid_res = 50
    x_bins = np.linspace(-0.05, 1.05, grid_res)
    y_bins = np.linspace(-0.05, H + 0.05, grid_res)
    X, Y = np.meshgrid(x_bins[:-1] + np.diff(x_bins)/2, y_bins[:-1] + np.diff(y_bins)/2)

    sep_x = np.linspace(0, 1, 500)
    sep_y = -(w[0] * sep_x + b) / (w[1] + 1e-10)

    # Mask separatrix to triangle bounds
    mask_triangle = (sep_y >= 0) & (sep_y <= np.sqrt(3) * sep_x) & (sep_y <= -np.sqrt(3) * (sep_x - 1))
    sep_x = sep_x[mask_triangle]
    sep_y = sep_y[mask_triangle]

    for row, (cluster_title, groups, cmap) in enumerate(target_clusters):
        cluster_df = df_gs_all[df_gs_all['unique_group'].isin(groups)].copy()
        
        cluster_df = cluster_df.sort_values(['run_id', 'group_id', 'step'])
        cluster_df['dmx'] = cluster_df.groupby(['run_id', 'group_id'])['mx'].diff()
        cluster_df['dmy'] = cluster_df.groupby(['run_id', 'group_id'])['my'].diff()
        
        for col, (period_title, filter_func) in enumerate(periods):
            ax = axes[row, col]
            
            triangle = Polygon([[0, 0], [1, 0], [0.5, H]], fill=False, edgecolor='black', linewidth=2.5, zorder=4)
            ax.add_patch(triangle)
            
            phase_data = cluster_df[filter_func(cluster_df)].dropna(subset=['dmx', 'dmy'])
            
            if len(phase_data) > 0:
                U, _, _, _ = binned_statistic_2d(phase_data['mx'], phase_data['my'], phase_data['dmx'], statistic='mean', bins=[x_bins, y_bins])
                V, _, _, _ = binned_statistic_2d(phase_data['mx'], phase_data['my'], phase_data['dmy'], statistic='mean', bins=[x_bins, y_bins])
                Counts, _, _, _ = binned_statistic_2d(phase_data['mx'], phase_data['my'], phase_data['dmx'], statistic='count', bins=[x_bins, y_bins])

                U, V, Counts = U.T, V.T, Counts.T

                U_smooth = gaussian_filter(np.nan_to_num(U, 0), sigma=1.5)
                V_smooth = gaussian_filter(np.nan_to_num(V, 0), sigma=1.5)

                # Ensure arrows are visible for small clusters by masking based on smoothed density
                arrow_density = gaussian_filter(Counts, sigma=1.5)
                threshold = arrow_density.max() * 0.15
                mask = arrow_density < threshold
                U_masked = ma.array(U_smooth, mask=mask)
                V_masked = ma.array(V_smooth, mask=mask)

                # Add background smooth heatmap showing distribution at the END of the period
                end_data = phase_data.groupby(['run_id', 'group_id']).last().reset_index()
                Counts_end, _, _, _ = binned_statistic_2d(end_data['mx'], end_data['my'], end_data['mx'], statistic='count', bins=[x_bins, y_bins])
                Counts_end = Counts_end.T
                density_end = gaussian_filter(Counts_end, sigma=1.5)
                
                heatmap_data = ma.array(density_end, mask=(density_end < 1e-4))
                extent = [x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()]
                mesh = ax.imshow(heatmap_data, extent=extent, origin='lower', cmap=cmap, alpha=0.6, interpolation='bilinear', zorder=1)
                mesh.set_clip_path(triangle)

                try:
                    stream = ax.streamplot(X, Y, U_masked, V_masked, color='black', 
                                  linewidth=1.5, density=1.5, arrowsize=1.5, zorder=2)
                    stream.lines.set_clip_path(triangle)
                    stream.arrows.set_clip_path(triangle)
                except ValueError:
                    pass
                    
            # Draw Separatrix
            ax.plot(sep_x, sep_y, color='black', linestyle='--', linewidth=2.5, zorder=5, label='Inferred Separatrix' if col == 1 else "")
                    
            ax.text(0.5, H + 0.02, "Social", ha='center', va='bottom', fontsize=14, weight='bold')
            ax.text(0.1, -0.02, "Private", ha='right', va='top', fontsize=14, weight='bold')
            ax.text(0.9, -0.02, "None", ha='left', va='top', fontsize=14, weight='bold')
            
            ax.axis('off')
            ax.set_aspect('equal')
            
            if row == 0:
                ax.set_title(period_title, fontsize=20, pad=15) # , weight='bold'
                
            if col == 0:
                ax.text(-0.2, H/2, cluster_title, va='center', ha='right', rotation=90, fontsize=20) # weight='bold'
                
            if col == 1 and row == 0:
                ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.15, wspace=0.1, hspace=0.1)
    plt.suptitle(f"Flow Diagrams ({target_type.capitalize()} Target Initialization)", fontsize=24) # , weight='bold'
    plt.savefig(f'figures/flow_clusters_1_6_{target_type}_target_{plot_version}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/flow_clusters_1_6_{target_type}_target_{plot_version}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


print("Generating distance stream plots...")

if df_dist is not None:
    # df_dist is already loaded and unique_group is assigned
    # We want a stream plot: X-axis = mean_good_dist, Y-axis = mean_bad_dist
    df_dist = df_dist.sort_values(['unique_group', 'tick'])
    
    # Calculate differentials
    df_dist['d_good'] = df_dist.groupby(['unique_group'])['median_good_dist'].diff()
    df_dist['d_bad'] = df_dist.groupby(['unique_group'])['median_bad_dist'].diff()
    
    # Filter valid rows
    df_dist = df_dist.dropna(subset=['d_good', 'd_bad'])
    
    max_dist = 10  # max(df_dist['mean_good_dist'].max(), df_dist['mean_bad_dist'].max()) * 1.1
    
    from matplotlib.collections import LineCollection
    from sklearn.cluster import KMeans

    print("Generating average trajectories colored by time (clustered by trajectory)...")
    
    for max_tick, suffix in [(10000, "0_to_10k"), (25000, "0_to_25k"), (100000, "0_to_100k"), (200000, "0_to_200k")]:
        phase_data_filtered = df_dist[df_dist['tick'] <= max_tick].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        cond_titles = {0: "Init on Good Target", 1: "Init on Bad Target"}
        
        for col, cond_val in enumerate([0, 1]):
            ax = axes[col]
            phase_data = phase_data_filtered[phase_data_filtered['close_target_index'] == cond_val].copy()
            
            if len(phase_data) == 0:
                continue
                
            # Use clusters from the performance/genetic space
            cluster_groups = cluster_groups_by_cond.get(cond_val, [])
            
            for c, groups in enumerate(cluster_groups):
                cluster_df = phase_data[phase_data['unique_group'].isin(groups)]
                if len(cluster_df) == 0: continue
                
                measure = "median"
                # Aggregate by tick
                agg_df = cluster_df.groupby('tick')[[f'{measure}_good_dist', f'{measure}_bad_dist']].mean().reset_index()
                
                # Smooth
                x = agg_df[f'{measure}_good_dist'].rolling(window=100, min_periods=1).mean().values
                y = agg_df[f'{measure}_bad_dist'].rolling(window=100, min_periods=1).mean().values
                ticks = agg_df['tick'].values
                
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(ticks.min(), ticks.max()), linewidth=3.5, alpha=0.9)
                lc.set_array(ticks[:-1])
                ax.add_collection(lc)
                
                # Start and End markers
                ax.scatter(x[0], y[0], marker='o', color='black', s=80, edgecolor='white', zorder=3)
                ax.scatter(x[-1], y[-1], marker='X', color='black', s=120, edgecolor='white', zorder=3)
                ax.text(x[-1], y[-1], f" C{6-c}", fontsize=12, fontweight='bold', color='black', verticalalignment='bottom', zorder=4)

            ax.plot([0, max_dist], [0, max_dist], 'k--', alpha=0.5, label='Equal Distance' if col==0 else "")
            
            ax.set_xlim(0, max_dist)
            ax.set_ylim(0, max_dist)
            ax.set_xlabel("Mean Distance to Good Target", fontsize=14)
            if col == 0:
                ax.set_ylabel("Mean Distance to Bad Target", fontsize=14)
            ax.set_aspect('equal')
            ax.set_title(cond_titles[cond_val], fontsize=16, pad=15)
            ax.grid(True, linestyle='--', alpha=0.5)
            if col == 0:
                ax.legend(loc='upper left')
                
        # Add colorbar globally to the figure to avoid overlap
        cbar = fig.colorbar(lc, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
        cbar.set_label("Tick Time")

        plt.savefig(f'figures/distance_average_trajectories_{measure}_{suffix}_{plot_version}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'figures/distance_average_trajectories_{measure}_{suffix}_{plot_version}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("Generating aggregated distance over ticks plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    
    for col, cond_val in enumerate([0, 1]):
        cond_df = df_dist[df_dist['close_target_index'] == cond_val]
        
        # 1. Isolate tick >= 60k for clustering late-stage behavior
        late_df = cond_df[cond_df['tick'] >= 60000]
        if len(late_df) == 0:
            continue
            
        # 2. Aggregate distance for each unique_group over late ticks
        late_agg = late_df.groupby('unique_group')[['mean_good_dist', 'mean_bad_dist']].mean()
        
        # 3. Perform KMeans Clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        late_agg['cluster'] = kmeans.fit_predict(late_agg.values)
        
        # Sort clusters by distance to good target so colors are consistent across conditions
        cluster_means = late_agg.groupby('cluster')['mean_good_dist'].mean().sort_values()
        cmap = {old_id: new_id for new_id, old_id in enumerate(cluster_means.index)}
        late_agg['cluster'] = late_agg['cluster'].map(cmap)
        
        # 4. Map clusters back to the data
        cluster_map = late_agg['cluster'].to_dict()
        cond_df = cond_df.copy()
        cond_df['traj_cluster'] = cond_df['unique_group'].map(cluster_map)
        
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
        
        for c in range(4):
            cluster_df = cond_df[cond_df['traj_cluster'] == c]
            
            # Limit plotting data to 0-40k
            cluster_df = cluster_df[cluster_df['tick'] <= 40000]
            
            if len(cluster_df) == 0: continue
            
            agg_df = cluster_df.groupby('tick').agg({
                'mean_good_dist': 'mean',
                'mean_bad_dist': 'mean'
            })
            
            ticks = agg_df.index
            mean_good = agg_df['mean_good_dist']
            mean_bad = agg_df['mean_bad_dist']
            
            n_groups = len(late_agg[late_agg['cluster'] == c])
            cluster_label = f"Cluster {c+1} (n={n_groups})"
            
            # Row 0: Distance to Good
            axes[0, col].plot(ticks, mean_good, color=colors[c], linewidth=2.5, label=cluster_label)
            
            # Row 1: Distance to Bad
            axes[1, col].plot(ticks, mean_bad, color=colors[c], linewidth=2.5, label=cluster_label)

        axes[0, col].set_title(cond_titles[cond_val], fontsize=16)
        axes[1, col].set_xlabel("Tick", fontsize=14)
        if col == 0:
            axes[0, col].set_ylabel("Mean Distance to Good Target", fontsize=14)
            axes[1, col].set_ylabel("Mean Distance to Bad Target", fontsize=14)
            
        axes[0, col].set_xlim(0, 40000)
        axes[1, col].set_xlim(0, 40000)
        
        if col == 0:
            axes[0, col].legend(loc='upper right')
            
    plt.tight_layout()
    plt.savefig(f'figures/aggregated_distance_over_ticks_0_to_40000_mean_late_clustered_{plot_version}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/aggregated_distance_over_ticks_0_to_40000_mean_late_clustered_{plot_version}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

print("Done! All figures are saved in the 'figures' folder.")