import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import scipy.stats as stats
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

# 3. Use only the initial strategy parameters, no fitness
genes_cols = ['gene_priv', 'gene_soc', 'gene_none']
probs = softmax(df_bad[genes_cols].values, axis=1)
df_bad['prob_priv'] = probs[:, 0]
df_bad['prob_soc'] = probs[:, 1]
df_bad['prob_none'] = probs[:, 2]
df_bad['gene_eve_scaled'] = (1 / (1 + np.exp(-df_bad['gene_eve']))) * (0.1 - 0.01) + 0.01
df_bad['unique_group'] = df_bad['run_id'].astype(str) + '_' + df_bad['group_id'].astype(str)

# Using simple subset (no wide features, no fitness)
agg_cols = ['prob_priv', 'prob_soc', 'prob_none', 'gene_eve_scaled']
print("Aggregating simple group features...")
df_group = df_bad.groupby(['unique_group', 'step'])[agg_cols].median().reset_index()

# Target definition (classification)
df_late_fitness = df_bad[(df_bad['step'] >= 80000) & (df_bad['step'] <= 100000)].groupby('unique_group')['current_fitness'].median()
target_y = (df_late_fitness > 0.5).astype(int).rename('success')

report_lines = [
    "# Robustness Checks: Predicting Success (Simple Subset)",
    "",
    "## Methodology",
    "- **Target**: Binary classification (Success > 0.5 median late fitness)",
    "- **Features**: Only initial strategy parameters (`prob_priv`, `prob_soc`, `prob_none`, `gene_eve_scaled`). No fitness features.",
    "- **Cross-Validation**: 5-Fold Stratified K-Fold",
    "- **Shuffle Test**: Target outcomes randomized to test for overfitting.",
    ""
]

for target_gen in [1, 10]:
    print(f"Processing Generation {target_gen}...")
    max_step = target_gen * 1000
    df_early = df_group[df_group['step'] <= max_step]
    
    early_mean = df_early.groupby('unique_group')[agg_cols].mean()
    early_mean.columns = [f'mean_{c}' for c in early_mean.columns]
    
    early_first = df_early.groupby('unique_group').first()
    early_last = df_early.groupby('unique_group').last()
    early_trend = (early_last[agg_cols] - early_first[agg_cols])
    early_trend.columns = [f'trend_{c}' for c in early_trend.columns]
    
    features = pd.concat([early_mean, early_trend], axis=1)
    dataset = features.join(target_y, how='inner').dropna()
    
    # If Gen 1, trend features are all 0, so we can drop them to avoid correlation warnings
    if target_gen == 1:
        dataset = dataset.drop(columns=early_trend.columns)
        
    X = dataset.drop(columns=['success'])
    y = dataset['success']
    
    # 4. Univariate effect sizes (Point-biserial correlation)
    effect_sizes = {}
    for col in X.columns:
        corr, p_val = stats.pointbiserialr(y, X[col])
        effect_sizes[col] = corr
    
    # Base rate
    majority_class_rate = max(y.mean(), 1 - y.mean())
    
    # 1. CV AUC and Accuracy
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    auc_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
    
    # 2. Shuffled CV
    y_shuffled = y.sample(frac=1, random_state=42).reset_index(drop=True)
    y_shuffled.index = y.index # Align indices
    auc_shuffled = cross_val_score(rf, X, y_shuffled, cv=cv, scoring='roc_auc')
    acc_shuffled = cross_val_score(rf, X, y_shuffled, cv=cv, scoring='accuracy')
    
    report_lines.extend([
        f"## Generation {target_gen}",
        f"**Base Rate (Always Predicting Majority Class)**: {majority_class_rate:.2%}",
        "",
        "### 1 & 2. Cross-Validation & Overfitting Check",
        "| Metric | Real Outcomes | Shuffled Outcomes (Null) |",
        "|:---|:---|:---|",
        f"| **CV Accuracy** | {np.mean(acc_scores):.2%} ± {np.std(acc_scores):.2%} | {np.mean(acc_shuffled):.2%} ± {np.std(acc_shuffled):.2%} |",
        f"| **CV AUC** | {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f} | {np.mean(auc_shuffled):.4f} ± {np.std(auc_shuffled):.4f} |",
        "",
        "### 4. Univariate Effect Sizes",
        "Point-biserial correlation (r) measures the linear relationship between the continuous feature and the binary success outcome.",
        "",
        "| Feature | Point-Biserial Correlation (r) | Effect Direction |",
        "|:---|:---|:---|"
    ])
    
    for feat, corr in sorted(effect_sizes.items(), key=lambda item: abs(item[1]), reverse=True):
        direction = "Positive" if corr > 0 else "Negative"
        report_lines.append(f"| `{feat}` | {corr:.4f} | {direction} |")
    
    report_lines.append("")

with open('robustness_tests_report.md', 'w') as f:
    f.write("\n".join(report_lines))

print("Done! Tests written to 'robustness_tests_report.md'")
