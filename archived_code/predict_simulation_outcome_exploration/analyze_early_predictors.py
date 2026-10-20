import os
import time
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax
from dotenv import load_dotenv

start_time = time.time()

# Load environment variables
load_dotenv()
wandb_dir = os.environ.get("WANDB_DIR", "./")
data_path = os.path.join(wandb_dir, "data", "v9arred3_jmrbbem4", "merged_agent_stats.parquet")

print("Loading data...")
try:
    df = pd.read_parquet(data_path)
except FileNotFoundError:
    df = pd.read_csv(data_path.replace(".parquet", ".csv.gz"))

# Ensure group_id is integer
if 'group_id' in df.columns:
    df['group_id'] = df['group_id'].astype(float).astype(int)

# Filter for Bad Target Initialization
print("Filtering for Bad Target Initialization (close_target_index == 1)...")
df_bad = df[df['close_target_index'] == 1].copy()

# Compute properties
print("Computing gene probabilities...")
genes_cols = ['gene_priv', 'gene_soc', 'gene_none']
probs = softmax(df_bad[genes_cols].values, axis=1)
df_bad['prob_priv'] = probs[:, 0]
df_bad['prob_soc'] = probs[:, 1]
df_bad['prob_none'] = probs[:, 2]
df_bad['gene_eve_scaled'] = (1 / (1 + np.exp(-df_bad['gene_eve']))) * (0.1 - 0.01) + 0.01
df_bad['unique_group'] = df_bad['run_id'].astype(str) + '_' + df_bad['group_id'].astype(str)

print("Aggregating group-level data...")
group_cols = ['unique_group', 'step']
agg_cols = ['current_fitness', 'prob_priv', 'prob_soc', 'prob_none', 'gene_eve_scaled']
df_group = df_bad.groupby(group_cols)[agg_cols].median().reset_index()

# Late Target (Gen 80 - 100)
print("Extracting late performance (Gen 80-100)...")
df_late = df_group[(df_group['step'] >= 80000) & (df_group['step'] <= 100000)]
target_y = df_late.groupby('unique_group')['current_fitness'].mean().rename('late_performance')

os.makedirs('figures', exist_ok=True)

windows = {
    '10_gen': 10000,
    '20_gen': 20000
}

for win_name, max_step in windows.items():
    print(f"\n--- Analyzing Early Window: {win_name} (Steps 0 to {max_step}) ---")
    df_early = df_group[df_group['step'] <= max_step].copy()
    
    # Extract features
    print("Extracting early features...")
    # Mean of features
    early_mean = df_early.groupby('unique_group')[agg_cols].mean()
    early_mean.columns = [f'mean_{c}' for c in early_mean.columns]
    
    # Trend (Last - First) of features
    df_early_sorted = df_early.sort_values(['unique_group', 'step'])
    early_first = df_early_sorted.groupby('unique_group').first()
    early_last = df_early_sorted.groupby('unique_group').last()
    early_trend = (early_last[agg_cols] - early_first[agg_cols])
    early_trend.columns = [f'trend_{c}' for c in early_trend.columns]
    
    # Combine features and target
    features = pd.concat([early_mean, early_trend], axis=1)
    dataset = features.join(target_y, how='inner').dropna()
    
    X = dataset.drop(columns=['late_performance'])
    y = dataset['late_performance']
    
    print(f"Dataset shape for {win_name}: {X.shape}")
    
    # 1. Correlation Analysis
    print("Generating correlation heatmap...")
    corr = dataset.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr[['late_performance']].sort_values(by='late_performance', ascending=False), 
                annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"Correlation with Late Performance\n(Early Window: {win_name})")
    plt.tight_layout()
    plt.savefig(f'figures/corr_late_perf_bad_target_{win_name}.png', dpi=300)
    plt.close()
    
    # 2. Random Forest Feature Importance
    print("Training Random Forest to extract feature importance...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='skyblue')
    plt.title(f"Random Forest Feature Importance\n(Predicting Late Performance from {win_name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f'figures/rf_importance_bad_target_{win_name}.png', dpi=300)
    plt.close()
    
    # 3. Decision Tree Rules
    print("Training Decision Tree to extract explicit rules...")
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(X, y)
    
    plt.figure(figsize=(24, 12))
    plot_tree(dt, feature_names=X.columns, filled=True, rounded=True, precision=3, fontsize=12)
    plt.title(f"Decision Tree for Late Performance\n(Features from {win_name})", fontsize=20)
    plt.tight_layout()
    plt.savefig(f'figures/decision_tree_bad_target_{win_name}.png', dpi=300)
    plt.close()
    
print("\nDone! Analysis figures saved in 'figures/' directory.")
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
