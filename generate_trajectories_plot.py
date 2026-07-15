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
data_path = os.path.join(wandb_dir, "data", "lck2wafo_dment0yi", "merged_agent_stats.parquet")

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
custom_cmap = mcolors.ListedColormap(['#E69F00', '#0072B2'])
# Define the threshold at 0.7 (bounds encompass potential min and max values)
bounds = [-0.1, 0.6, 1] 
norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
print("Generating clustered trajectories based on raw median performance...")

conditions = sorted(group_perf['close_target_index'].unique())

for i, cond in enumerate(conditions):
    cond_data = group_perf[group_perf['close_target_index'] == cond].copy()
    
    heatmap_data = cond_data.pivot(index='unique_group', columns='step', values='current_fitness')
    
    X_cluster = heatmap_data.fillna(0).values
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

    heatmap_data = heatmap_data.iloc[sorted_indices]
    heatmap_cols = heatmap_data.columns
    
    fig = plt.figure(figsize=(38, 32))
    gs = gridspec.GridSpec(3, n_clusters, height_ratios=[2, 1, 1], hspace=0.18, wspace=0.05)
    
    ax_main = fig.add_subplot(gs[0, :])

    extent = [heatmap_cols.min()/1000, heatmap_cols.max()/1000, 0, heatmap_data.shape[0]]
    
    cax = ax_main.imshow(
        heatmap_data, 
        aspect='auto', 
        cmap=custom_cmap, 
        interpolation='none', 
        extent=extent, 
        norm=norm,
        origin='lower', 
        )

    ax_main.set_xlabel("Generation (1k steps)", fontsize=32, labelpad=0)
    # custom_ticks now stops before 200
    custom_ticks = [1] + list(np.arange(25, 200, 25))
    ax_main.set_xticks(custom_ticks)
    ax_main.set_xticklabels([str(x) for x in custom_ticks])
    ax_main.tick_params(labelsize=24)
    ax_main.axvline(x=100, color='white', linestyle=':', linewidth=3)

    title_str = "Init Good Target" if cond == 0 else "Init Bad Target"
    ax_main.set_yticks([])
    ax_main.set_ylabel(f"Groups {title_str} (Clustered & Sorted)", fontsize=28, weight='bold', labelpad=70)

    for idx, (y_min, y_max) in enumerate(cluster_boundaries):
        rect = Rectangle((extent[0], y_min), extent[1]-extent[0], y_max-y_min, 
                         linewidth=6, edgecolor='black', facecolor='none', zorder=5)
        ax_main.add_patch(rect)
        
        ax_main.text(extent[0] - 2, (y_min + y_max)/2, f'{n_clusters-idx}', 
                     color='black', fontsize=36, weight='bold', va='center', ha='right',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    cbar_ax = fig.add_axes([0.91, 0.53, 0.015, 0.35])
    
    cbar = fig.colorbar(cax, cax=cbar_ax, orientation='vertical', spacing='proportional')
    cbar.ax.tick_params(labelsize=28)
    cbar.set_ticks([0.3, 0.8])
    cbar.ax.set_yticklabels(['< 0.6', '> 0.6'], rotation=90, va='center')
    cbar.ax.tick_params(labelsize=28, length=0)
    cbar.set_label('Median Performance', fontsize=32, weight='bold', labelpad=5)

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
        ax_info = fig.add_subplot(gs[1, cluster_col])
        ax_noise = fig.add_subplot(gs[2, cluster_col])
        # ax_flow = fig.add_subplot(gs[3, cluster_col])
        
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

    filename = 'clustered_trajectories_gene_ts_perf_good.pdf' if cond == 0 else 'clustered_trajectories_gene_ts_perf_bad.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # # plt.show()
    plt.close(fig)


# Separate Flow Diagram for Bad Target Initialization (Clusters 1 and 6)
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

# Identify Best and Worst Groups
best_groups = cluster_groups[5]
worst_groups = cluster_groups[0]

best_df = df_gs_all[df_gs_all['unique_group'].isin(best_groups)]
worst_df = df_gs_all[df_gs_all['unique_group'].isin(worst_groups)]

# Calculate Divergence Step
best_means = best_df.groupby('step')[['mx', 'my']].mean()
worst_means = worst_df.groupby('step')[['mx', 'my']].mean()
merged_means = best_means.join(worst_means, lsuffix='_best', rsuffix='_worst').dropna()
merged_means['dist'] = np.sqrt((merged_means['mx_best'] - merged_means['mx_worst'])**2 + (merged_means['my_best'] - merged_means['my_worst'])**2)

diverge_step = merged_means[merged_means['dist'] > 0.05].index.min()
diverge_gen = (diverge_step + 1) / 1000 if diverge_step is not pd.NA else 6.0
print(f"Divergence Generation: {diverge_gen}")

# Infer Separatrix using SVM
post_best = best_df[best_df['step'] >= diverge_step][['mx', 'my']].dropna()
post_worst = worst_df[worst_df['step'] >= diverge_step][['mx', 'my']].dropna()
X_svm = np.vstack([post_best.values, post_worst.values])
y_svm = np.array([1]*len(post_best) + [0]*len(post_worst))

svm = LinearSVC(random_state=42, C=1.0)
svm.fit(X_svm, y_svm)
w = svm.coef_[0]
b = svm.intercept_[0]
print(f"Separatrix equation: {w[0]:.4f} * mx + {w[1]:.4f} * my + {b:.4f} = 0")

target_clusters = [
    ("Cluster 1 (Best)", best_groups, 'Blues'),
    ("Cluster 6 (Worst)", worst_groups, 'Reds')
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
        
        if len(phase_data) >= 5:
            U, _, _, _ = binned_statistic_2d(phase_data['mx'], phase_data['my'], phase_data['dmx'], statistic='mean', bins=[x_bins, y_bins])
            V, _, _, _ = binned_statistic_2d(phase_data['mx'], phase_data['my'], phase_data['dmy'], statistic='mean', bins=[x_bins, y_bins])
            Counts, _, _, _ = binned_statistic_2d(phase_data['mx'], phase_data['my'], phase_data['dmx'], statistic='count', bins=[x_bins, y_bins])

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
                              linewidth=1.5, density=2.5, arrowsize=1.5, zorder=2)
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
            ax.set_title(period_title, fontsize=20, weight='bold', pad=15)
            
        if col == 0:
            ax.text(-0.2, H/2, cluster_title, va='center', ha='right', rotation=90, fontsize=20, weight='bold')
            
        if col == 1 and row == 0:
            ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.9, left=0.15, wspace=0.1, hspace=0.1)
plt.suptitle("Flow Diagrams (Bad Target Initialization)", fontsize=24, weight='bold')
plt.savefig('flow_clusters_1_6_bad_target.pdf', dpi=300, bbox_inches='tight')
# # plt.show()
plt.close(fig)