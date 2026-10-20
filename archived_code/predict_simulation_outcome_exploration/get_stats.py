import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax
from dotenv import load_dotenv

load_dotenv()

wandb_dir = os.environ.get("WANDB_DIR", "./")
data_path = os.path.join(wandb_dir, "data", "v9arred3_jmrbbem4", "merged_agent_stats.parquet")

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

df_late = df_group[(df_group['step'] >= 80000) & (df_group['step'] <= 100000)]
target_y = df_late.groupby('unique_group')['current_fitness'].mean().rename('late_performance')

windows = {'10_gen': 10000, '20_gen': 20000}

with open('early_predictor_stats.md', 'w') as f:
    f.write("# Early Predictors of Late Performance (Bad Target Initialization)\n\n")
    
    for win_name, max_step in windows.items():
        df_early = df_group[df_group['step'] <= max_step].copy()
        
        early_mean = df_early.groupby('unique_group')[agg_cols].mean()
        early_mean.columns = [f'mean_{c}' for c in early_mean.columns]
        
        df_early_sorted = df_early.sort_values(['unique_group', 'step'])
        early_first = df_early_sorted.groupby('unique_group').first()
        early_last = df_early_sorted.groupby('unique_group').last()
        early_trend = (early_last[agg_cols] - early_first[agg_cols])
        early_trend.columns = [f'trend_{c}' for c in early_trend.columns]
        
        features = pd.concat([early_mean, early_trend], axis=1)
        dataset = features.join(target_y, how='inner').dropna()
        
        X = dataset.drop(columns=['late_performance'])
        y = dataset['late_performance']
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        r2_score = rf.score(X, y)
        
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        corr = dataset.corr()['late_performance'].drop('late_performance').sort_values(ascending=False)
        
        f.write(f"## Early Window: {win_name.replace('_gen', ' Generations')} (Steps 0 to {max_step})\n")
        f.write(f"**Overall Model Explanatory Power (R²):** {r2_score:.2%}\n\n")
        f.write("### Feature Importances (Random Forest)\n")
        f.write("| Feature | Importance (Adds up to 100%) |\n|---|---|\n")
        for idx, val in importances.items():
            f.write(f"| {idx} | {val:.2%} |\n")
        
        f.write("\n### Linear Correlations (Pearson)\n")
        f.write("| Feature | Correlation Coefficient |\n|---|---|\n")
        for idx, val in corr.items():
            f.write(f"| {idx} | {val:.3f} |\n")
        f.write("\n---\n\n")
