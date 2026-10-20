import os
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

# Compute genes
genes_cols = ['gene_priv', 'gene_soc', 'gene_none']
probs = softmax(df_bad[genes_cols].values, axis=1)
df_bad['prob_priv'] = probs[:, 0]
df_bad['prob_soc'] = probs[:, 1]
df_bad['prob_none'] = probs[:, 2]
df_bad['gene_eve_scaled'] = (1 / (1 + np.exp(-df_bad['gene_eve']))) * (0.1 - 0.01) + 0.01
df_bad['unique_group'] = df_bad['run_id'].astype(str) + '_' + df_bad['group_id'].astype(str)

agg_cols = ['current_fitness', 'prob_priv', 'prob_soc', 'prob_none', 'gene_eve_scaled']

print("Aggregating wide features (mean, median, max, q90) across agents...")
def q90(x):
    return x.quantile(0.9)
q90.__name__ = 'q90'

# Aggregate across the agents within a group at a specific step
df_group_multi = df_bad.groupby(['unique_group', 'step'])[agg_cols].agg(['mean', 'median', 'max', q90])
# Flatten columns: 'prob_priv_mean', 'prob_priv_max', etc.
df_group_multi.columns = [f"{col}_{func}" for col, func in df_group_multi.columns]
df_group = df_group_multi.reset_index()

print("Calculating Late Performance...")
df_late = df_group[(df_group['step'] >= 80000) & (df_group['step'] <= 100000)]
# Use the group's median fitness (mean over the late time window) as the target
target_y = df_late.groupby('unique_group')['current_fitness_median'].mean().rename('late_performance')

results = []
importances_list = []

# List of all 20 the new feature columns (5 metrics * 4 aggregations)
feature_cols = [c for c in df_group.columns if c not in ['unique_group', 'step']]

print("Evaluating predictive power from Generation 1 to 30...")
df_group = df_group.sort_values(['unique_group', 'step'])

for gen in range(1, 31):
    max_step = gen * 1000
    df_early = df_group[df_group['step'] <= max_step]
    
    # 1. Mean over the early time window
    early_mean = df_early.groupby('unique_group')[feature_cols].mean()
    early_mean.columns = [f'time_mean_{c}' for c in early_mean.columns]
    
    # 2. Trend over the early time window
    early_first = df_early.groupby('unique_group').first()
    early_last = df_early.groupby('unique_group').last()
    early_trend = (early_last[feature_cols] - early_first[feature_cols])
    early_trend.columns = [f'time_trend_{c}' for c in early_trend.columns]
    
    # Total of 40 features per group
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

results_df.to_csv('predictive_power_wide_1_to_30.csv', index=False)
importances_df.to_csv('feature_importances_wide_1_to_30.csv', index=False)

print("Saving statistics to markdown...")
with open('predictive_power_wide_1_to_30.md', 'w') as f:
    f.write("# Predictive Power with Wide Features (1 to 30 Generations)\n\n")
    f.write("## Overall Explanatory Power (R²)\n")
    f.write("| Generation | Explanatory Power (R²) |\n")
    f.write("|---|---|\n")
    for _, row in results_df.iterrows():
        f.write(f"| {int(row['Generation'])} | {row['R2_Score']:.2%} |\n")
        
    f.write("\n## Top 15 Most Important Features at Generation 10\n")
    gen10 = importances_df[importances_df['Generation'] == 10].iloc[0].drop('Generation').sort_values(ascending=False)
    f.write("| Feature | Importance |\n|---|---|\n")
    for feat, imp in gen10.head(15).items():
        f.write(f"| {feat} | {imp:.2%} |\n")
        
    f.write("\n## Top 15 Most Important Features at Generation 30\n")
    gen30 = importances_df[importances_df['Generation'] == 30].iloc[0].drop('Generation').sort_values(ascending=False)
    f.write("| Feature | Importance |\n|---|---|\n")
    for feat, imp in gen30.head(15).items():
        f.write(f"| {feat} | {imp:.2%} |\n")

print("Generating graphs...")
os.makedirs('figures', exist_ok=True)
from matplotlib.ticker import PercentFormatter

# Plot 1: R2 Score
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")
plt.plot(results_df['Generation'], results_df['R2_Score'], marker='o', linestyle='-', linewidth=3, color='#D55E00')
plt.fill_between(results_df['Generation'], results_df['R2_Score'], 0, alpha=0.1, color='#D55E00')

plt.title("Explanatory Power (R²) of Late Performance over Early Generations\n(Wide Features: Mean, Median, Max, Q90)", fontsize=18, pad=15)
plt.xlabel("Number of Early Generations Provided to the Model", fontsize=14)
plt.ylabel("R² Score (Explanatory Power)", fontsize=14)
plt.ylim(0, 1.05)
plt.xlim(1, 30)
plt.xticks(range(1, 31))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()
plt.savefig('figures/predictive_power_wide_over_time.png', dpi=300)
plt.close()

# Plot 2: Top 10 Feature Importances Over Time
# Find the overall top 10 features across all generations to plot cleanly
top_features = importances_df.drop('Generation', axis=1).mean().sort_values(ascending=False).head(10).index

plt.figure(figsize=(16, 9))
sns.set_style("whitegrid")
colors = sns.color_palette("tab10", n_colors=len(top_features))

for idx, feat in enumerate(top_features):
    plt.plot(importances_df['Generation'], importances_df[feat], marker='o', markersize=4, linestyle='-', linewidth=2, label=feat, color=colors[idx])

plt.title("Top 10 Feature Importances Change over Early Generations (Wide Features)", fontsize=18, pad=15)
plt.xlabel("Number of Early Generations Provided to the Model", fontsize=14)
plt.ylabel("Feature Importance", fontsize=14)
plt.xlim(1, 30)
plt.xticks(range(1, 31))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('figures/feature_importances_wide_over_time.png', dpi=300)
plt.close()

print("\nDone! Saved stats to 'predictive_power_wide_1_to_30.md' and plots to 'figures/'.")
