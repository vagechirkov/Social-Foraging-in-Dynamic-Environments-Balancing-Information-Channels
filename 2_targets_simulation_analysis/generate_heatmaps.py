import time
start_time = time.time()

import os
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from scipy.special import softmax
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Construct the path using WANDB_DIR
wandb_dir = os.environ.get("WANDB_DIR", "./")
data_path = os.path.join(wandb_dir, "data", "v9arred3_jmrbbem4", "merged_agent_stats.parquet")

print(f"Loading data from {data_path}...")
try:
    df = pd.read_parquet(data_path)
except FileNotFoundError:
    csv_path = data_path.replace(".parquet", ".csv.gz")
    print(f"Parquet not found. Falling back to CSV: {csv_path}")
    df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")

if 'group_id' in df.columns:
    df['group_id'] = df['group_id'].astype(float).astype(int)

df_flow = df.copy()

# Compute softmax probabilities from the raw gene values
genes_cols = ['gene_priv', 'gene_soc', 'gene_none']
probs = softmax(df_flow[genes_cols].values, axis=1)

df_flow['prob_priv'] = probs[:, 0]
df_flow['prob_soc'] = probs[:, 1]
df_flow['prob_none'] = probs[:, 2]
df_flow['gene_eve_scaled'] = (1 / (1 + np.exp(-df_flow['gene_eve']))) * (0.1 - 0.01) + 0.01

print("Calculating gene aggregations (mean, median, max)...")
group_cols = ['run_id', 'group_id', 'step', 'close_target_index']
gene_cols = ['prob_priv', 'prob_soc', 'prob_none', 'gene_eve_scaled']

df_mean = df_flow.groupby(group_cols)[gene_cols].mean().reset_index()
df_median = df_flow.groupby(group_cols)[gene_cols].median().reset_index()
df_max = df_flow.groupby(group_cols)[gene_cols].max().reset_index()

for d in [df_mean, df_median, df_max]:
    d['unique_group'] = d['run_id'].astype(str) + '_' + d['group_id'].astype(str)

print("Calculating raw median performance per group from df...")
group_perf = df.groupby(group_cols)['current_fitness'].median().reset_index()
group_perf = group_perf[group_perf['step'] < 199000]
group_perf['unique_group'] = group_perf['run_id'].astype(str) + '_' + group_perf['group_id'].astype(str)

plot_version = 'v8_gene_heatmaps'
n_clusters = 6

print("Generating clustered trajectories based on median performance...")

conditions = sorted(group_perf['close_target_index'].unique())

os.makedirs('figures', exist_ok=True)

custom_perf_cmap = mcolors.LinearSegmentedColormap.from_list('custom_perf', [
    '#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7',
    '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419'
])

gene_cmaps = {
    'prob_priv': 'Blues',
    'prob_soc': 'Oranges',
    'prob_none': 'Greens',
    'gene_eve_scaled': 'Purples'
}
gene_titles = {
    'prob_priv': 'Private',
    'prob_soc': 'Social',
    'prob_none': 'None',
    'gene_eve_scaled': 'Process Noise'
}

for i, cond in enumerate(conditions):
    print(f"Processing condition {cond}...")
    cond_data = group_perf[group_perf['close_target_index'] == cond].copy()
    heatmap_data = cond_data.pivot(index='unique_group', columns='step', values='current_fitness')
    
    # Cluster based on median performance across all steps (as modified by user)
    X_cluster = heatmap_data.fillna(0).values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_cluster)
    cluster_labels = kmeans.labels_

    # Calculate mean performance 
    group_mean_perf = heatmap_data.mean(axis=1).values
    cluster_avg_perf = {c: group_mean_perf[cluster_labels == c].mean() for c in range(n_clusters)}
    sorted_clusters = sorted(cluster_avg_perf.keys(), key=lambda c: cluster_avg_perf[c])

    sorted_indices = []
    cluster_boundaries = []
    current_y = 0
    
    for c in sorted_clusters:
        idx_in_cluster = np.where(cluster_labels == c)[0]
        sorted_idx_in_cluster = idx_in_cluster[np.argsort(group_mean_perf[idx_in_cluster])]
        sorted_indices.extend(sorted_idx_in_cluster)
        
        size = len(idx_in_cluster)
        cluster_boundaries.append((current_y, current_y + size))
        current_y += size
    
    heatmap_data_sorted = heatmap_data.iloc[sorted_indices]
    heatmap_cols = heatmap_data_sorted.columns
    extent = [heatmap_cols.min()/1000, heatmap_cols.max()/1000, 0, heatmap_data_sorted.shape[0]]
    
    title_str = "Good Target Initialisation" if cond == 0 else "Bad Target Initialisation"
    
    # PLOT 0: Median Performance Heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    cax = ax.imshow(
        heatmap_data_sorted, 
        aspect='auto', 
        cmap=custom_perf_cmap, 
        interpolation='none', 
        extent=extent, 
        origin='lower', 
        rasterized=True
    )
    ax.set_title(f"Median Performance - {title_str}", fontsize=24, pad=20)
    ax.set_xlabel("Generation (1k steps)", fontsize=20)
    custom_ticks = [1] + list(np.arange(25, 200, 25))
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels([str(x) for x in custom_ticks])
    ax.tick_params(labelsize=16)
    ax.axvline(x=100, color='white', linestyle=':', linewidth=3)
    ax.set_yticks([])
    ax.set_ylabel("Groups", fontsize=20)

    for idx, (y_min, y_max) in enumerate(cluster_boundaries):
        rect = Rectangle((extent[0], y_min), extent[1]-extent[0], y_max-y_min, 
                         linewidth=4, edgecolor='black', facecolor='none', zorder=5)
        ax.add_patch(rect)
        ax.text(extent[0] - 2, (y_min + y_max)/2, f'{n_clusters-idx}', 
                     color='black', fontsize=20, weight='bold', va='center', ha='right',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                     
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Median Performance', fontsize=20)
    
    perf_filename = f'figures/heatmap_perf_med_{"good" if cond == 0 else "bad"}_{plot_version}.png'
    plt.savefig(perf_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # PLOT 1-4: Gene Heatmaps (Mean, Median, Max)
    df_mean_cond = df_mean[df_mean['close_target_index'] == cond]
    df_median_cond = df_median[df_median['close_target_index'] == cond]
    df_max_cond = df_max[df_max['close_target_index'] == cond]
    
    gene_pivots = {}
    for g in gene_cols:
        p_mean = df_mean_cond.pivot(index='unique_group', columns='step', values=g).reindex(heatmap_data.index).iloc[sorted_indices]
        p_median = df_median_cond.pivot(index='unique_group', columns='step', values=g).reindex(heatmap_data.index).iloc[sorted_indices]
        p_max = df_max_cond.pivot(index='unique_group', columns='step', values=g).reindex(heatmap_data.index).iloc[sorted_indices]
        
        p_mean = p_mean.reindex(columns=heatmap_cols)
        p_median = p_median.reindex(columns=heatmap_cols)
        p_max = p_max.reindex(columns=heatmap_cols)
        
        gene_pivots[g] = (p_mean, p_median, p_max)

    for g in gene_cols:
        p_mean, p_median, p_max = gene_pivots[g]
        
        fig, axes = plt.subplots(1, 3, figsize=(30, 12), sharey=True)
        fig.suptitle(f"{gene_titles[g]} Gene - {title_str}", fontsize=32, y=1.05)
        
        aggs = [('Mean', p_mean), ('Median', p_median), ('Max', p_max)]
        
        for ax_idx, (agg_name, p_data) in enumerate(aggs):
            ax = axes[ax_idx]
            
            vmin = 0 if 'eve' not in g else 0.01
            vmax = 1 if 'eve' not in g else 0.1
            
            cax = ax.imshow(
                p_data, 
                aspect='auto', 
                cmap=gene_cmaps[g], 
                interpolation='none', 
                extent=extent, 
                origin='lower', 
                rasterized=True,
                vmin=vmin,
                vmax=vmax
            )
            ax.set_title(f"{agg_name} {gene_titles[g]}", fontsize=28, pad=15)
            ax.set_xlabel("Generation (1k steps)", fontsize=24)
            ax.set_xticks(custom_ticks)
            ax.set_xticklabels([str(x) for x in custom_ticks])
            ax.tick_params(labelsize=20)
            ax.axvline(x=100, color='gray', linestyle=':', linewidth=3)
            
            if ax_idx == 0:
                ax.set_yticks([])
                ax.set_ylabel("Groups", fontsize=24)
                for idx, (y_min, y_max) in enumerate(cluster_boundaries):
                    rect = Rectangle((extent[0], y_min), extent[1]-extent[0], y_max-y_min, 
                                     linewidth=4, edgecolor='black', facecolor='none', zorder=5)
                    ax.add_patch(rect)
                    ax.text(extent[0] - 2, (y_min + y_max)/2, f'{n_clusters-idx}', 
                                 color='black', fontsize=20, weight='bold', va='center', ha='right',
                                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            else:
                for idx, (y_min, y_max) in enumerate(cluster_boundaries):
                    rect = Rectangle((extent[0], y_min), extent[1]-extent[0], y_max-y_min, 
                                     linewidth=4, edgecolor='black', facecolor='none', zorder=5)
                    ax.add_patch(rect)

        cbar = fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(f'{gene_titles[g]} Value', fontsize=24)
        
        gene_filename = f'figures/heatmap_gene_{g}_{"good" if cond == 0 else "bad"}_{plot_version}.png'
        plt.savefig(gene_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

print("Done! Heatmaps generated.")
end_time = time.time()
total_seconds = end_time - start_time
minutes = int(total_seconds // 60)
seconds = total_seconds % 60
print(f"Total execution time: {minutes} minutes and {seconds:.2f} seconds")
