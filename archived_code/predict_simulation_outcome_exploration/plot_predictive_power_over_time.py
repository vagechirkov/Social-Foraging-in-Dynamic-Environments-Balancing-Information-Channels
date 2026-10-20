import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
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

df_bad = df[df['close_target_index'] == 1].copy()
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

print("Calculating Late Performance...")
df_late = df_group[(df_group['step'] >= 80000) & (df_group['step'] <= 100000)]
target_y = df_late.groupby('unique_group')['current_fitness'].mean().rename('late_performance')

results = []
importances_list = []

print("Evaluating predictive power and feature importance from Generation 1 to 30...")

# Ensure sorted for trend calculations
df_group = df_group.sort_values(['unique_group', 'step'])

for gen in range(1, 31):
    max_step = gen * 1000
    df_early = df_group[df_group['step'] <= max_step]
    
    early_mean = df_early.groupby('unique_group')[agg_cols].mean()
    early_mean.columns = [f'mean_{c}' for c in early_mean.columns]
    
    early_first = df_early.groupby('unique_group').first()
    early_last = df_early.groupby('unique_group').last()
    early_trend = (early_last[agg_cols] - early_first[agg_cols])
    early_trend.columns = [f'trend_{c}' for c in early_trend.columns]
    
    features = pd.concat([early_mean, early_trend], axis=1)
    dataset = features.join(target_y, how='inner').dropna()
    
    X = dataset.drop(columns=['late_performance'])
    y = dataset['late_performance']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    r2_score = rf.score(X, y)
    
    print(f"Generation {gen:02d} | Explanatory Power (R²): {r2_score:.2%}")
    results.append({'Generation': gen, 'R2_Score': r2_score})
    
    imp_dict = {'Generation': gen}
    for feat, imp in zip(X.columns, rf.feature_importances_):
        imp_dict[feat] = imp
    importances_list.append(imp_dict)

results_df = pd.DataFrame(results)
importances_df = pd.DataFrame(importances_list)

print("Saving statistics...")
results_df.to_csv('predictive_power_1_to_30_gen.csv', index=False)
importances_df.to_csv('feature_importances_1_to_30_gen.csv', index=False)

with open('predictive_power_1_to_30_gen.md', 'w') as f:
    f.write("# Predictive Power and Feature Importances of Early Generations (1 to 30)\n\n")
    f.write("## Overall Explanatory Power (R²)\n")
    f.write("| Generation | Explanatory Power (R²) |\n")
    f.write("|---|---|\n")
    for _, row in results_df.iterrows():
        f.write(f"| {int(row['Generation'])} | {row['R2_Score']:.2%} |\n")
        
    f.write("\n## Feature Importances (at Gen 10, 20, 30)\n")
    f.write("| Feature | Gen 10 | Gen 20 | Gen 30 |\n")
    f.write("|---|---|---|---|\n")
    
    # We may not have perfectly gen 10/20/30 if data missing, but range guarantees it
    gen10 = importances_df[importances_df['Generation'] == 10].iloc[0]
    gen20 = importances_df[importances_df['Generation'] == 20].iloc[0]
    gen30 = importances_df[importances_df['Generation'] == 30].iloc[0]
    
    for feat in X.columns:
        f.write(f"| {feat} | {gen10[feat]:.2%} | {gen20[feat]:.2%} | {gen30[feat]:.2%} |\n")

print("Generating graphs...")
os.makedirs('figures', exist_ok=True)
from matplotlib.ticker import PercentFormatter

# Plot 1: R2 Score
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")
plt.plot(results_df['Generation'], results_df['R2_Score'], marker='o', linestyle='-', linewidth=3, color='#0072B2')
plt.fill_between(results_df['Generation'], results_df['R2_Score'], 0, alpha=0.1, color='#0072B2')

plt.title("Explanatory Power (R²) of Late Performance over Early Generations\n(Bad Target Initialization)", fontsize=18, pad=15)
plt.xlabel("Number of Early Generations Provided to the Model", fontsize=14)
plt.ylabel("R² Score (Explanatory Power)", fontsize=14)
plt.ylim(0, 1.05)
plt.xlim(1, 30)
plt.xticks(range(1, 31))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()
plt.savefig('figures/predictive_power_over_time_1_to_30.png', dpi=300)
plt.savefig('figures/predictive_power_over_time_1_to_30.pdf', dpi=300)
plt.close()

# Plot 2: Feature Importances Over Time
plt.figure(figsize=(16, 9))
sns.set_style("whitegrid")

# Sort features by their average importance to have a clean legend order
avg_importances = importances_df.drop('Generation', axis=1).mean().sort_values(ascending=False)
features_sorted = avg_importances.index

# Use a nice categorical colormap
colors = sns.color_palette("tab10", n_colors=len(features_sorted))

for idx, feat in enumerate(features_sorted):
    plt.plot(importances_df['Generation'], importances_df[feat], marker='o', markersize=4, linestyle='-', linewidth=2, label=feat, color=colors[idx])

plt.title("Feature Importance Change over Early Generations\n(Predicting Late Performance)", fontsize=18, pad=15)
plt.xlabel("Number of Early Generations Provided to the Model", fontsize=14)
plt.ylabel("Feature Importance (adds to 100%)", fontsize=14)
plt.ylim(0, importances_df.drop('Generation', axis=1).max().max() * 1.1)
plt.xlim(1, 30)
plt.xticks(range(1, 31))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('figures/feature_importances_over_time_1_to_30.png', dpi=300)
plt.savefig('figures/feature_importances_over_time_1_to_30.pdf', dpi=300)
plt.close()

print("\nDone! Saved stats to 'predictive_power_1_to_30_gen.md' and plots to 'figures/'.")
