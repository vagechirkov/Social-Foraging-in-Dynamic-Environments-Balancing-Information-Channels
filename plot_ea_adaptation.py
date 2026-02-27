import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re
from scipy.spatial import distance
from scipy.integrate import simpson

OUTPUT_DIR = "outputs/publication_plots"
CACHE_FILE = os.path.join(OUTPUT_DIR, "metrics_cache.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)

def jensen_shannon_divergence(p, q):
    """Calculates JSD between two probability distributions.
    scipy's jensenshannon returns the square root of JSD, so we square it.
    """
    # ensure they are numpy arrays and normalized
    p = np.array(p)
    q = np.array(q)
    # add small epsilon to avoid div by zero or log(0)
    epsilon = 1e-8
    p = p + epsilon
    q = q + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    return distance.jensenshannon(p, q, base=2.0)**2

def process_adaptation():
    print(f"Loading metrics from {CACHE_FILE}")
    if not os.path.exists(CACHE_FILE):
         print("Error: metrics_cache.csv not found!")
         return
    df = pd.read_csv(CACHE_FILE)
    
    # 1. Identify static optimal strategies S_solitary and S_collective
    # We'll average the last 500 generations of the static runs.
    static_df = df[df["mode"] == "static"]
    optimal_strategies = {}
    
    for (mut_prob, sel), group in static_df.groupby(["mutation_prob", "selection"]):
        # Solitary
        sol_group = group[group["switch_interval"] == "static-solitary"]
        if not sol_group.empty:
            # Average over last 500 generations (max generation is 3000)
            max_gen = sol_group["Generation"].max()
            sol_last = sol_group[sol_group["Generation"] >= max_gen - 500]
            s_sol = sol_last[["Private", "Belief", "None"]].mean().values
            s_sol = s_sol / np.sum(s_sol)
            f_sol = sol_last["Fitness"].mean()
        else:
            s_sol = np.array([1.0, 0.0, 0.0]) # fallback
            f_sol = 1.0
            
        # Collective
        col_group = group[group["switch_interval"] == "static-collective"]
        if not col_group.empty:
            max_gen = col_group["Generation"].max()
            col_last = col_group[col_group["Generation"] >= max_gen - 500]
            s_col = col_last[["Private", "Belief", "None"]].mean().values
            s_col = s_col / np.sum(s_col)
            f_col = col_last["Fitness"].mean()
        else:
            s_col = np.array([0.0, 1.0, 0.0]) # fallback
            f_col = 1.0
            
        optimal_strategies[(mut_prob, sel)] = {
            "Solitary": {"strategy": s_sol, "fitness": f_sol},
            "Collective": {"strategy": s_col, "fitness": f_col}
        }
        print(f"Optimal Static for mut={mut_prob}, sel={sel}: Solitary={s_sol} (F={f_sol:.3f}), Collective={s_col} (F={f_col:.3f})")

    # 2. Calculate JSD for dynamic runs
    dynamic_df = df[df["mode"] == "dynamic"].copy()
    dynamic_df["switch_interval"] = pd.to_numeric(dynamic_df["switch_interval"])
    
    jsd_records = []
    
    for idx, row in dynamic_df.iterrows():
        mut_prob = row["mutation_prob"]
        sel = row["selection"]
        gen = row["Generation"]
        sw_int = row["switch_interval"]
        
        current_strategy = np.array([row["Private"], row["Belief"], row["None"]])
        # Replace NaNs with 0 and normalize (though shouldn't be NaNs)
        current_strategy = np.nan_to_num(current_strategy)
        s_sum = np.sum(current_strategy)
        if s_sum > 0:
            current_strategy = current_strategy / s_sum
        else:
            current_strategy = np.array([1/3, 1/3, 1/3])
        
        phase_idx = (gen // int(sw_int)) % 2
        phase_text = "Solitary" if phase_idx == 0 else "Collective"
        
        target_info = optimal_strategies.get((mut_prob, sel), {}).get(phase_text, {"strategy": np.array([1/3, 1/3, 1/3]), "fitness": 1.0})
        target_strategy = target_info["strategy"]
        target_fitness = target_info["fitness"]
        
        dist = jensen_shannon_divergence(current_strategy, target_strategy)
        
        # We need to preserve the run information if there are multiple runs, 
        # but in df they are already separated by rows (we don't have run_id in df directly)
        # However, they might be averaged or there might be duplicates for same generation?
        # Let's just append to a list and reconstruct the dataframe
        jsd_records.append({
            "mutation_prob": mut_prob,
            "selection": sel,
            "switch_interval": sw_int,
            "Generation": gen,
            "JSD": dist,
            "Fitness": row["Fitness"],
            "Target_Fitness": target_fitness,
            "Phase": phase_text
        })
        
    jsd_df = pd.DataFrame(jsd_records)
    
    # 3. Plot Adaptation Dynamics
    sel_order = ["individual-global", "multilevel-local"]
    
    for mut_prob, mut_group in jsd_df.groupby("mutation_prob"):
        print(f"Plotting JSD for mutation_prob={mut_prob}")
        
        g = sns.relplot(
            data=mut_group,
            kind="line",
            x="Generation",
            y="JSD",
            row="selection",
            col="switch_interval",
            hue="Phase",
            palette={"Solitary": "#009E73", "Collective": "#56B4E9"},
            errorbar=('ci', 95),
            height=4, aspect=2,
            row_order=sel_order,
            facet_kws={'margin_titles': True}
        )
        g.set(ylim=(0, None), xlim=(0, 3000))
        g.set_axis_labels("Generation", "Jensen-Shannon Divergence")
        g.set_titles(row_template="{row_name}", col_template="Switch = {col_name}")
        
        sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
        
        for ax in g.axes.flat:
            col_match = re.search(r'switch_interval\s*=\s*(\d+)', ax.get_title())
            if col_match:
                dyn_sw_int = int(col_match.group(1))
                tick_interval = dyn_sw_int * 3 if dyn_sw_int == 100 else dyn_sw_int
            else:
                tick_interval = 300
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
            ax.tick_params(axis='x', rotation=45)
            
        plt.savefig(os.path.join(OUTPUT_DIR, f"adaptation_jsd_mut{mut_prob}.pdf"), dpi=300, bbox_inches='tight')
        plt.close(g.fig)

    # 4. Quantify Adaptation Speed (Half-life and AUC)
    print("\n--- Calculating Adaptation Speed ---")
    speed_records = []
    
    # We must treat each trace (run) separately for AUC and Half-life, but we lost run_id in caching...
    # Fortunately, seaborn bootstraps over identical (Generation, condition) rows.
    # We can group by (mutation_prob, selection, switch_interval, Generation) to get the MEAN JSD over runs.
    # Then calculate half-life, AUC JSD, and Cumulative Regret on the mean curve.
    
    mean_jsd_df = jsd_df.groupby(["mutation_prob", "selection", "switch_interval", "Generation"]).agg({
        "JSD": "mean", 
        "Fitness": "mean",
        "Target_Fitness": "first",
        "Phase": "first"
    }).reset_index()
    
    for (mut_prob, sel, sw_int), group in mean_jsd_df.groupby(["mutation_prob", "selection", "switch_interval"]):
        group = group.sort_values(by="Generation")
        max_gen = group["Generation"].max()
        
        num_switches = int(max_gen // sw_int)
        
        for k in range(1, num_switches): # exclude the very first phase (k=0) as it doesn't follow a switch
            start_gen_idx = k * sw_int
            end_gen_idx = (k + 1) * sw_int
            
            # Slice the interval
            interval_data = group[(group["Generation"] >= start_gen_idx) & (group["Generation"] < end_gen_idx)]
            if interval_data.empty:
                continue
                
            peak_jsd = interval_data.iloc[0]["JSD"]
            
            # Calculate Half-life
            # find first gen where JSD <= 0.5 * peak_jsd
            half_life = np.nan
            for _, row in interval_data.iterrows():
                if row["JSD"] <= 0.5 * peak_jsd:
                    half_life = row["Generation"] - start_gen_idx
                    break
                    
            if np.isnan(half_life):
                # If it never reaches half, we cap it at switch_interval or mark as NaN
                half_life = sw_int 
            
            # Calculate AUC of JSD
            if len(interval_data) > 1:
                auc = simpson(y=interval_data["JSD"].values, x=interval_data["Generation"].values)
            else:
                auc = 0.0
                
            # Calculate Cumulative Regret (Adaptive Load)
            regret = np.abs(interval_data["Target_Fitness"].values - interval_data["Fitness"].values)
            if len(interval_data) > 1:
                cum_regret = simpson(y=regret, x=interval_data["Generation"].values)
            else:
                cum_regret = 0.0
                
            phase_val = interval_data.iloc[0]["Phase"]
            transition_str = "Collective → Solitary" if phase_val == "Solitary" else "Solitary → Collective"
            
            speed_records.append({
                "mutation_prob": mut_prob,
                "selection": sel,
                "switch_interval": sw_int,
                "Switch_Index": k,
                "Phase": phase_val,
                "Transition": transition_str,
                "Peak_JSD": peak_jsd,
                "Half_Life": half_life,
                "AUC": auc,
                "Cumulative_Regret": cum_regret
            })
            
    speed_df = pd.DataFrame(speed_records)
    speed_df.to_csv(os.path.join(OUTPUT_DIR, "adaptation_speed.csv"), index=False)
    
    # Summary Table
    summary_speed = speed_df.groupby(["mutation_prob", "selection", "switch_interval"]).agg({
        "Half_Life": ["mean", "std"],
        "AUC": ["mean", "std"],
        "Cumulative_Regret": ["mean", "std"]
    }).reset_index()
    
    print("\nAdaptation Speed Summary:")
    print(summary_speed)
    summary_speed.to_csv(os.path.join(OUTPUT_DIR, "adaptation_speed_summary.csv"), index=False)

    # 5. Plot Adaptation Catplots
    # We will use catplot with x="switch_interval", y=metric, hue="Transition", row="selection", col="mutation_prob"

    # Make switch_interval categorical for better display
    speed_df["switch_interval"] = speed_df["switch_interval"].astype(str)

    # Reorder conditions for uniform plotting
    transition_order = ["Solitary → Collective", "Collective → Solitary"]
    switch_order = ["100", "300"]
    
    for metric in ["Half_Life", "AUC", "Cumulative_Regret"]:
        with sns.plotting_context("paper", font_scale=1.2):
            g = sns.catplot(
                data=speed_df,
                kind="point",
                x="switch_interval",
                y=metric,
                hue="Transition",
                row="selection",
                col="mutation_prob",
                row_order=sel_order,
                order=switch_order,
                hue_order=transition_order,
                palette="Set2",
                height=4.0,
                aspect=1.0,
                errorbar=('ci', 95),
                capsize=0.1,
                err_kws={'linewidth': 1.5},
                margin_titles=True,
                dodge=True,
                markers=["o", "s"],
                join=False
            )
            
            g.set_axis_labels("Switch Interval", metric.replace("_", " "))
            g.set_titles(row_template="{row_name}", col_template="Mut Prob = {col_name}")
            
            # Adjust legend and layout
            sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, title="Transition")
            g.fig.subplots_adjust(wspace=0.05, hspace=0.1)
            
            plt.savefig(os.path.join(OUTPUT_DIR, f"adaptation_catplot_{metric}.pdf"), dpi=300, bbox_inches='tight')
            plt.close(g.fig)
        
    print(f"\nSaved speed metrics to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_adaptation()
