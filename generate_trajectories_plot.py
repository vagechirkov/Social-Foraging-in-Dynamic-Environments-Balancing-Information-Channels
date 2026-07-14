#!/usr/bin/env python
# coding: utf-8

# # Reproduce Group Trajectories Plot
# 
# This notebook contains the minimal code necessary to process the simulation data and reproduce the group trajectory phase plots.

# ## 1. Imports and Setup
# Load the necessary libraries for data manipulation, mathematical operations, clustering, and plotting.

# In[1]:


import os
import numpy as np
import pandas as pd
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


# ## 2. Data Loading
# Load the `merged_agent_stats.parquet` dataset using paths from environment variables.

# In[2]:


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


# In[3]:


df[df['close_target_index'] == 1].head(10)





# ## 3. Data Preparation
# Filter for close_target_index=1, compute success metric (if any agent reached > 0.8 fitness), convert gene expression to probabilities, and aggregate to group-level step data.

# In[5]:


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
df_group_steps = df_flow.groupby(group_cols)[['prob_priv', 'prob_soc', 'prob_none']].mean().reset_index()

# Ensure chronological ordering
df_group_steps = df_group_steps.sort_values(['run_id', 'group_id', 'step'])


# ## 4. Trajectory and Gradient Calculations
# Map the ternary probabilities (Private, Social, None) into a 2D Cartesian triangle space, and compute the temporal gradients (how the probabilities shift over time).

# In[6]:


df_gs = df_group_steps.copy()

# Define the height of the equilateral triangle
H = np.sqrt(3) / 2

# Map ternary coordinates to 2D Cartesian coordinates (mx, my)
df_gs['mx'] = df_gs['prob_none'] + 0.5 * df_gs['prob_soc']
df_gs['my'] = H * df_gs['prob_soc']

# Sort chronologically to accurately compute differences
df_gs = df_gs.sort_values(['run_id', 'group_id', 'step'])

# Calculate the difference (gradient) between consecutive steps for each group
df_gs['dmx'] = df_gs.groupby(['run_id', 'group_id'])['mx'].diff()
df_gs['dmy'] = df_gs.groupby(['run_id', 'group_id'])['my'].diff()

# Drop rows with NaN (first step of each group)
df_gs = df_gs.dropna(subset=['dmx', 'dmy'])

# Normalize step units
df_gs['step'] -= 999
df_gs['step'] /= 1000

# Define time phases: Early (0-5k), Mid (5-12k), Late (12-200k)
max_step = df_gs['step'].max()
time_bins = [0, 5, 12, 200]
time_labels = ['Early', 'Mid', 'Late']

# Our two 'conditions' (clusters) are now True (Success) and False (Failure)
clusters = [True, False]
df_gs.shape



# ## 6. Robust State Identification & Transitions
# In this section, we apply a robust state identification pipeline: calculate fitness fraction, apply a 2-step rolling average, assign local states (Success >= 2/3, else Failure), and filter for blocks stable for at least 10 generations. Unstable blocks are marked as 'Mix'.

# In[8]:


print("Running robust state identification...")

# df_flow_all = df[df['step'] < 99000].copy()
df_flow_all = df.copy()

# 1. Raw Metric: fraction of agents tracking the target per group per step
df_flow_all['is_fit'] = df_flow_all['current_fitness'] > 0.6
frac_success = df_flow_all.groupby(['run_id', 'group_id', 'step'])['is_fit'].mean().reset_index()
frac_success = frac_success.sort_values(['run_id', 'group_id', 'step'])

# 2. Smoothing: 2-step rolling average
frac_success['smoothed_frac'] = frac_success.groupby(['run_id', 'group_id'])['is_fit'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())

# 3. Local State: Success if >= 2/3, else Failure
# Using np.where to assign states quickly
threshold = 2/3
frac_success['local_state'] = np.where(frac_success['smoothed_frac'] >= (threshold), 'Success', 'Failure')

# 4. Stability Filter: >= 10 generations
def apply_stability_filter(df_group):
    # Identify contiguous blocks
    blocks = (df_group['local_state'] != df_group['local_state'].shift()).cumsum()
    df_group['block_id'] = blocks

    # Calculate length of each block
    block_lengths = df_group.groupby('block_id')['step'].transform('count')

    # Assign final stable state
    df_group['stable_state'] = np.where(block_lengths >= 10, df_group['local_state'], 'Mix')
    return df_group

df_success = frac_success.groupby(['run_id', 'group_id'], group_keys=False).apply(apply_stability_filter)

df_flow_all = df_flow_all.merge(df_success[['run_id', 'group_id', 'step', 'stable_state']], on=['run_id', 'group_id', 'step'])

# Compute probabilities
genes_cols = ['gene_priv', 'gene_soc', 'gene_none']
probs_all = softmax(df_flow_all[genes_cols].values, axis=1)
df_flow_all['prob_priv'] = probs_all[:, 0]
df_flow_all['prob_soc'] = probs_all[:, 1]
df_flow_all['prob_none'] = probs_all[:, 2]

# Aggregate to group-level
group_cols = ['run_id', 'group_id', 'step', 'stable_state']
df_gs_all = df_flow_all.groupby(group_cols)[['prob_priv', 'prob_soc', 'prob_none']].mean().reset_index()
df_gs_all = df_gs_all.sort_values(['run_id', 'group_id', 'step'])

print("State distribution:")
print(df_gs_all['stable_state'].value_counts(normalize=True))


group_states_time = df_flow_all[['run_id', 'group_id', 'step', 'close_target_index', 'stable_state']].drop_duplicates()


import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Rectangle
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
import numpy.ma as ma
from sklearn.cluster import KMeans
import matplotlib.transforms as mtransforms

def draw_density_heatmap(ax, data, cmap, title, H=np.sqrt(3)/2):
    triangle = Polygon([[0, 0], [1, 0], [0.5, H]], fill=False, edgecolor='black', linewidth=3.5, zorder=4)
    ax.add_patch(triangle)

    grid_res = 50
    x_bins = np.linspace(-0.05, 1.05, grid_res)
    y_bins = np.linspace(-0.05, H + 0.05, grid_res)

    if len(data) >= 5:
        Counts, _, _, _ = binned_statistic_2d(data['mx'], data['my'], data['mx'], statistic='count', bins=[x_bins, y_bins])
        Counts = Counts.T

        density = gaussian_filter(Counts, sigma=1.5)
        if density.max() > 0:
            density = density / density.max()

        density_masked = ma.masked_where(density < 0.05, density)

        X, Y = np.meshgrid(x_bins, y_bins)
        colormesh = ax.pcolormesh(X, Y, density_masked, cmap=cmap, shading='flat', zorder=2, alpha=0.95, rasterized=True)

        clip_poly = Polygon([[0, 0], [1, 0], [0.5, H]], transform=ax.transData)
        colormesh.set_clip_path(clip_poly)

    ax.text(0.5, H + 0.02, "S", ha='center', va='bottom', fontsize=24, weight='bold')
    ax.text(-0.02, -0.02, "P", ha='right', va='top', fontsize=24, weight='bold')
    ax.text(1.02, -0.02, "N", ha='left', va='top', fontsize=24, weight='bold')

    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=24, weight='bold', pad=25)

state_map = {'Success': 2, 'Failure': 0, 'Mix': 1}
cmap = mcolors.ListedColormap(['#D55E00', '#CC79A7', '#0072B2'])

if 'mx' not in df_gs_all.columns:
    H = np.sqrt(3)/2
    df_gs_all['mx'] = df_gs_all['prob_none'] + 0.5 * df_gs_all['prob_soc']
    df_gs_all['my'] = H * df_gs_all['prob_soc']

conditions = sorted(group_states_time['close_target_index'].unique())

for i, cond in enumerate(conditions):
    cond_data = group_states_time[group_states_time['close_target_index'] == cond].copy()
    cond_data['unique_group'] = cond_data['run_id'].astype(str) + '_' + cond_data['group_id'].astype(str)
    heatmap_data = cond_data.pivot(index='unique_group', columns='step', values='stable_state')
    heatmap_int = heatmap_data.replace(state_map)

    X_cluster = heatmap_int.fillna(1).values
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_cluster)
    cluster_labels = kmeans.labels_

    group_success = (heatmap_data == 'Success').sum(axis=1).values
    cluster_avg_success = {c: group_success[cluster_labels == c].mean() for c in range(4)}
    sorted_clusters = sorted(cluster_avg_success.keys(), key=lambda c: cluster_avg_success[c])

    sorted_indices = []
    for c in sorted_clusters:
        idx_in_cluster = np.where(cluster_labels == c)[0]
        sorted_idx_in_cluster = idx_in_cluster[np.argsort(group_success[idx_in_cluster])]
        sorted_indices.extend(sorted_idx_in_cluster)

    heatmap_data = heatmap_data.iloc[sorted_indices]
    heatmap_int = heatmap_int.iloc[sorted_indices]
    heatmap_int_cols = heatmap_int.columns
    heatmap_int.columns = heatmap_int.columns / 1000

    import numpy as np
    fig = plt.figure(figsize=(38, 20))
    ax_main = fig.add_axes([0.15, 0.35, 0.58, 0.30])

    extent = [heatmap_int.columns.min(), heatmap_int.columns.max(), 0, heatmap_data.shape[0]]
    cax = ax_main.imshow(heatmap_int, aspect='auto', cmap=cmap, interpolation='none', vmin=-0.5, vmax=2.5, extent=extent, origin='lower')

    ax_main.set_xlabel("Generation (1k steps)", fontsize=32, labelpad=15)
    # Update x-ticks to be every 25
    ax_main.set_xlabel("Generation (1k steps)", fontsize=32, labelpad=15)
    # Update x-ticks to be every 25
    custom_ticks = [1] + list(np.arange(25, 200, 25)) + [200]
    ax_main.set_xticks(custom_ticks)
    ax_main.set_xticklabels([str(x) for x in custom_ticks])
    ax_main.tick_params(labelsize=24)
    ax_main.axvline(x=105, color='white', linestyle=':', linewidth=3)

    title_str = "Init Good Target" if cond == 0 else "Init Bad Target"
    ax_main.set_yticks([])
    ax_main.set_ylabel(f"{title_str}\n\nIndividual Groups\n(Clustered & Sorted)", fontsize=28, weight='bold', labelpad=40)


    num_groups = heatmap_data.shape[0]
    regions = []

    if cond == 0:
        top_windows = [[5, 15], [75, 85], [105, 115], [190, 200]]
        x_pos = [0.15, 0.32, 0.49, 0.66]
        for idx, gw in enumerate(top_windows):
            regions.append({
                'name': f'Gen {gw[0]}-{gw[1]}', 'gen_x': gw, 'y': [num_groups-100, num_groups], 
                'cmap': 'Blues', 'fig_x': x_pos[idx], 'fig_y': 0.73,
                'groups': heatmap_data.index[num_groups-100:num_groups]
            })

        bot_windows = [[5, 15], [75, 85], [105, 115], [190, 200]]
        for idx, gw in enumerate(bot_windows):
            regions.append({
                'name': f'Gen {gw[0]}-{gw[1]}', 'gen_x': gw, 'y': [0, 25], 
                'cmap': 'Reds', 'fig_x': x_pos[idx], 'fig_y': 0.10,
                'groups': heatmap_data.index[0:25]
            })
    else:
        top_bot_windows = [[1, 5], [5, 10], [10, 15], [75, 85], [105, 115], [190, 200]]
        x_pos_tb = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]

        for idx, gw in enumerate(top_bot_windows):
            regions.append({
                'name': f'Gen {gw[0]}-{gw[1]}', 'gen_x': gw, 'y': [num_groups-100, num_groups], 
                'cmap': 'Blues', 'fig_x': x_pos_tb[idx], 'fig_y': 0.73,
                'groups': heatmap_data.index[num_groups-100:num_groups]
            })
        for idx, gw in enumerate(top_bot_windows):
            regions.append({
                'name': f'Gen {gw[0]}-{gw[1]}', 'gen_x': gw, 'y': [0, 100], 
                'cmap': 'Reds', 'fig_x': x_pos_tb[idx], 'fig_y': 0.10,
                'groups': heatmap_data.index[0:100]
            })

    steps_fail = [s for s in heatmap_int_cols if 95000 <= s <= 105000]
    steps_succ = [s for s in heatmap_int_cols if 125000 <= s <= 135000]
    fail_mask = heatmap_data[steps_fail].eq('Failure').any(axis=1)
    succ_mask = heatmap_data[steps_succ].eq('Success').any(axis=1)
    recovering = fail_mask & succ_mask
    y_indices = np.where(recovering)[0]

    if len(y_indices) > 0:
        mid_idx = len(y_indices) // 2
        slice_start = max(0, mid_idx)
        slice_end = min(len(y_indices), mid_idx + 40)
        y_indices = y_indices[slice_start:slice_end]

        if len(y_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max() + 1
            recov_groups = heatmap_data.index[y_indices]

            mid_windows = [[95, 105], [105, 115], [115, 125], [125, 135]]
            y_pos_right = [0.73, 0.52, 0.31, 0.10] 
            for idx, gw in enumerate(mid_windows):
                regions.append({
                    'name': f'Gen {gw[0]}-{gw[1]}', 'gen_x': gw, 'y': [y_min, y_max], 
                    'cmap': 'Purples', 'fig_x': 0.8, 'fig_y': y_pos_right[idx],
                    'groups': recov_groups
                })

    for r_idx, r in enumerate(regions):
        box_num = r_idx + 1
        rect = Rectangle((r['gen_x'][0], r['y'][0]), r['gen_x'][1]-r['gen_x'][0], r['y'][1]-r['y'][0], 
                         linewidth=4.5, edgecolor='black', facecolor='none', zorder=5)
        ax_main.add_patch(rect)

        x_center = r['gen_x'][0] + (r['gen_x'][1] - r['gen_x'][0]) / 2
        ax_main.text(x_center, r['y'][1], str(box_num),
                     color='black', fontsize=18, weight='bold', ha='center', va='bottom',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.2', alpha=0.9), zorder=6)

        step_start = r['gen_x'][0] * 1000
        step_end = r['gen_x'][1] * 1000

        mask = (df_gs_all['step'] >= step_start) & (df_gs_all['step'] <= step_end)
        data_subset = df_gs_all[mask].copy()
        if 'unique_group' not in data_subset.columns:
            data_subset['unique_group'] = data_subset['run_id'].astype(str) + '_' + data_subset['group_id'].astype(str)
        data_subset = data_subset[data_subset['unique_group'].isin(r['groups'])]

        axins = fig.add_axes([r['fig_x'], r['fig_y'], 0.08, 0.15])
        title_with_num = f"({box_num}) {r['name']}"
        draw_density_heatmap(axins, data_subset, r['cmap'], title_with_num)

    cbar_ax = fig.add_axes([0.735, 0.35, 0.015, 0.30])
    cbar = fig.colorbar(cax, cax=cbar_ax, orientation='vertical', ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Failure', 'Mix', 'Success'], rotation=90, va='center')
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label('State', fontsize=32, weight='bold', labelpad=5)

    filename = 'individual_trajectories_annotated_good.pdf' if cond == 0 else 'individual_trajectories_annotated_bad.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)








