import os
import pandas as pd
import numpy as np
from scipy.special import softmax
from dotenv import load_dotenv

load_dotenv()
wandb_dir = os.environ.get("WANDB_DIR", "./")
data_path = os.path.join(wandb_dir, "data", "v9arred3_jmrbbem4", "merged_agent_stats.parquet")

print("Loading dataset...")
try:
    df = pd.read_parquet(data_path)
except FileNotFoundError:
    df = pd.read_csv(data_path.replace(".parquet", ".csv.gz"))

if 'group_id' in df.columns:
    df['group_id'] = df['group_id'].astype(float).astype(int)

# Filter for bad target
df_bad = df[df['close_target_index'] == 1].copy()

# Compute genes
genes_cols = ['gene_priv', 'gene_soc', 'gene_none']
probs = softmax(df_bad[genes_cols].values, axis=1)
df_bad['prob_priv'] = probs[:, 0]
df_bad['prob_soc'] = probs[:, 1]
df_bad['prob_none'] = probs[:, 2]
df_bad['gene_eve_scaled'] = (1 / (1 + np.exp(-df_bad['gene_eve']))) * (0.1 - 0.01) + 0.01
df_bad['unique_group'] = df_bad['run_id'].astype(str) + '_' + df_bad['group_id'].astype(str)

group_cols = ['unique_group', 'step']
agg_cols = ['current_fitness', 'prob_priv', 'prob_soc', 'prob_none', 'gene_eve_scaled']
df_group = df_bad.groupby(group_cols)[agg_cols].median().reset_index()

# 1. Get Late Performance (Gen 80 - 100)
df_late = df_group[(df_group['step'] >= 80000) & (df_group['step'] <= 100000)]
late_perf = df_late.groupby('unique_group')['current_fitness'].median()

# 2. Split into Success (>0.5) and Failure (<=0.5)
success_groups = late_perf[late_perf > 0.5].index
failure_groups = late_perf[late_perf <= 0.5].index

print(f"\nFound {len(success_groups)} groups with late performance > 0.5")
print(f"Found {len(failure_groups)} groups with late performance <= 0.5")

# 3. Get Initial Conditions (Gen 1: step <= 1000)
df_init = df_group[df_group['step'] <= 1000]
init_features = df_init.groupby('unique_group')[agg_cols].mean()

# 4. Compare the two groups
success_init = init_features.loc[success_groups].mean()
failure_init = init_features.loc[failure_groups].mean()

comparison = pd.DataFrame({
    'Success (Late > 0.5)': success_init,
    'Failure (Late <= 0.5)': failure_init
})
comparison['Difference'] = comparison['Success (Late > 0.5)'] - comparison['Failure (Late <= 0.5)']

print("\n--- Initial Conditions (Generation 1 Average) ---")
print(comparison.round(4).to_string())

comparison.round(4).to_csv('initial_conditions_comparison.csv')
print("\nStats saved to 'initial_conditions_comparison.csv'")
