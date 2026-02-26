import os
import io
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wandb
import numpy as np
import re

WANDB_PROJECT = "dynamic_evolution_v1"
OUTPUT_DIR = "outputs/publication_plots"
CACHE_FILE = os.path.join(OUTPUT_DIR, "metrics_cache.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

api = wandb.Api()

switch_intervals = [100, 300]
mutation_probs = [0.2, 0.8]
selections = ["individual-local", "individual-global"]

def get_selection_name(sel, multi_level):
    if sel == "individual-global":
        return "individual-global"
    elif sel == "individual-local":
        if multi_level:
            return "multilevel-local"
        else:
            return "individual-local"
    return sel

print("Querying WandB...")
runs = api.runs(WANDB_PROJECT, filters={
    "$and": [
        {"config.evolution.mutation_prob": {"$in": mutation_probs}},
        {"config.evolution.selection": {"$in": selections}},
        {"config.use_gpu": True},
        {"$or": [
            {"config.environment.mode": "static"},
            {"$and": [
                {"config.environment.mode": "dynamic"},
                {"config.evolution.switch_interval": {"$in": switch_intervals}}
            ]}
        ]}
    ]
})

print(f"Found {len(runs)} runs matching criteria.")

def process_and_plot(runs):
    # 1. Fetch metrics or load from cache
    if os.path.exists(CACHE_FILE):
        print("\n--- Loading Metrics from Cache ---")
        df = pd.read_csv(CACHE_FILE)
        # Reconstruct valid_runs to just have references for gif download
        # since we won't need to re-query history
        valid_runs = [run for run in runs if run.state == "finished"]
    else:
        print("\n--- Fetching Metrics ---")
        all_metrics_data = []
        valid_runs = []
        for run in runs:
            if run.state != "finished":
                print(f"Skipping run {run.id} as it is in state {run.state}")
                continue

            valid_runs.append(run)
            print(f"Fetching metrics for run: {run.id} ({run.name})")
            
            cfg = run.config
            mode = cfg.get("environment", {}).get("mode", "Unknown")
            static_cat = cfg.get("environment", {}).get("static_category", "Unknown")
            
            if mode == "dynamic":
                sw_int = cfg.get("evolution", {}).get("switch_interval", "Unknown")
            else:
                sw_int = f"static-{static_cat}"
                
            mut_prob = cfg.get("evolution", {}).get("mutation_prob", "Unknown")
            sel = cfg.get("evolution", {}).get("selection", "Unknown")
            multi_level = cfg.get("evolution", {}).get("multi_level_selection", False)
            
            sel_name = get_selection_name(sel, multi_level)
            
            num_runs = cfg.get("num_runs", 1)

            keys_to_fetch = []
            for r in range(num_runs):
                keys_to_fetch.extend([
                    f"run_{r}/global_mean",
                    f"run_{r}/avg/prob_Priv",
                    f"run_{r}/avg/prob_Belief",
                    f"run_{r}/avg/prob_None"
                ])
                
            history = list(run.scan_history(keys=["_step"] + keys_to_fetch))
            
            step_metrics = {}
            for row in history:
                step = row["_step"]
                if step not in step_metrics:
                    step_metrics[step] = {}
                for k in keys_to_fetch:
                    if k in row and row[k] is not None:
                        step_metrics[step][k] = row[k]
                        
            for step, metrics in step_metrics.items():
                for r in range(num_runs):
                    if f"run_{r}/global_mean" in metrics:
                        all_metrics_data.append({
                            "mode": mode,
                            "switch_interval": sw_int,
                            "mutation_prob": mut_prob,
                            "selection": sel_name,
                            "Generation": step,
                            "Fitness": metrics.get(f"run_{r}/global_mean", np.nan),
                            "Private": metrics.get(f"run_{r}/avg/prob_Priv", np.nan),
                            "Belief": metrics.get(f"run_{r}/avg/prob_Belief", np.nan),
                            "None": metrics.get(f"run_{r}/avg/prob_None", np.nan),
                        })
                        
        if not all_metrics_data:
            print("No metrics data found.")
            return
            
        df = pd.DataFrame(all_metrics_data)
        df.to_csv(CACHE_FILE, index=False)
        print(f"Metrics cached to {CACHE_FILE}")

    print("\n--- Generating Plots ---")
    
    # We want to combine dynamic environments and static extremes (static-solitary, static-collective)
    # The 'switch_interval' column has '100', '300', 'static-solitary', 'static-collective'
    # Group by mutation probability
    
    for mut_prob, mut_group in df.groupby("mutation_prob"):
        print(f"Plotting for mutation_prob={mut_prob}")
        
        # Determine the dynamic intervals present for this mutation
        dynamic_intervals = [sw for sw in mut_group["switch_interval"].unique() if isinstance(sw, (int, float)) or (isinstance(sw, str) and sw.isdigit())]
        
        for dyn_sw in dynamic_intervals:
            print(f"  Generating Fitness Plot combining static extremes and dynamic sw={dyn_sw}")
            
            # Filter the dataframe to just the static defaults + this dynamic switch
            target_intervals = [dyn_sw, "static-solitary", "static-collective"]
            sub_df = mut_group[mut_group["switch_interval"].isin(target_intervals)].copy()
            
            # Plot Fitness (Stack Vertically)
            g = sns.relplot(
                data=sub_df, 
                kind="line",
                x="Generation", 
                y="Fitness", 
                row="selection", # Stacking vertically
                hue="switch_interval",
                style="switch_interval",
                errorbar=('ci', 95),
                n_boot=50,
                palette="Set1",
                height=4, aspect=1.8 # Taller stacked plots
            )
            g.set(ylim=(0.35, 0.75), xlim=(0, 3000))
            g.fig.suptitle(f"Mean Fitness (Mut Prob: {mut_prob}, Dynamic Env vs Static sw={dyn_sw})", y=1.05)
            
            dyn_sw_int = int(dyn_sw)
            tick_interval = dyn_sw_int * 3 if dyn_sw_int == 100 else dyn_sw_int
            for ax in g.axes.flat:
                # Show every 3rd tick for 100 to avoid clutter
                ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
                ax.tick_params(axis='x', rotation=45)
            
            plt.savefig(os.path.join(OUTPUT_DIR, f"fitness_combined_sw{dyn_sw}_mut{mut_prob}.png"), dpi=300, bbox_inches='tight')
            plt.close(g.fig)

            # Channel Probs Plots for the dynamic environment (split by selection)
            for sel, sel_group in sub_df.groupby("selection"):
                group_melted = sel_group.melt(
                    id_vars=["Generation", "switch_interval"], 
                    value_vars=["Private", "Belief", "None"],
                    var_name="Channel",
                    value_name="Probability"
                )
                
                # CI Plot
                plt.figure(figsize=(12, 6))
                ax2 = sns.lineplot(
                    data=group_melted,
                    x="Generation",
                    y="Probability",
                    hue="Channel",
                    style="switch_interval",
                    errorbar=('ci', 95),
                    n_boot=50,
                    palette=["#009E73", "#56B4E9", "#E69F00"] # Distinct color palette for channels
                )
                plt.ylim(0, 1)
                plt.xlim(0, 3000)
                ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
                plt.xticks(rotation=45)
                
                plt.title(f"Channel Probabilities CI ({sel}, Env Switch: {dyn_sw}, Mut Prob: {mut_prob})")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"channel_probs_ci_{sel}_sw{dyn_sw}_mut{mut_prob}.png"), dpi=300)
                plt.close()
                
                # SD Plot to highlight emergence of frequency-dependent strategies
                # Show individual run variance directly instead of just CI bounding boxes
                plt.figure(figsize=(12, 6))
                ax3 = sns.lineplot(
                    data=group_melted,
                    x="Generation",
                    y="Probability",
                    hue="Channel",
                    style="switch_interval",
                    estimator='mean',
                    errorbar='sd',
                    err_kws={'alpha': 0.3}, # Highlight variance / strategy divergence 
                    palette=["#009E73", "#56B4E9", "#E69F00"] 
                )
                plt.ylim(0, 1)
                plt.xlim(0, 3000)
                ax3.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
                plt.xticks(rotation=45)
                
                plt.title(f"Channel Probabilities SD ({sel}, Env Switch: {dyn_sw}, Mut Prob: {mut_prob})")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"channel_probs_sd_{sel}_sw{dyn_sw}_mut{mut_prob}.png"), dpi=300)
                plt.close()

    print(f"Plots saved to {OUTPUT_DIR}")

    print("\n--- Generating GIFs ---")
    for run in valid_runs:
        cfg = run.config
        mode = cfg.get("environment", {}).get("mode", "Unknown")
        sw_int = cfg.get("evolution", {}).get("switch_interval", "static") if mode == "dynamic" else "static"
        mut_prob = cfg.get("evolution", {}).get("mutation_prob", "Unknown")
        sel = cfg.get("evolution", {}).get("selection", "Unknown")
        multi_level = cfg.get("evolution", {}).get("multi_level_selection", False)
        
        sel_name = get_selection_name(sel, multi_level)

        if mode == "dynamic":
            download_gifs(run, sw_int, mut_prob, sel_name)


def extract_gen(fname):
    m = re.search(r'ternary_density_(?:top_k|all)_(\d+)_', fname)
    if m:
        return int(m.group(1))
    m = re.search(r'gen_(\d+)', fname)
    return int(m.group(1)) if m else -1

def extract_prefix(fname):
    m = re.search(r'(run_\d+)/ternary', fname)
    return m.group(1) if m else "global_"

def download_gifs(run, sw_int, mut_prob, sel_name):
    # The user wants GIFs between 1800 and 2400 generations.
    print(f"  Downloading ternary density plots for GIF for {run.id} (between gens 1800-2400)...")
    files = list(run.files())
    ternary_files = [f for f in files if "ternary_density_top_k" in f.name and f.name.endswith(".png")]
    if not ternary_files:
        print(f"  No ternary density files found for {run.id}.")
        return

    # Filter to only the target generations: 1800 <= gen <= 2400
    filtered_files = [f for f in ternary_files if 1800 <= extract_gen(f.name) <= 2400]

    # Force sort by generation correctly
    filtered_files.sort(key=lambda x: extract_gen(x.name))
    
    global_files = [f for f in filtered_files if extract_prefix(f.name) == "global_"]
    if len(global_files) < 2:
        print(f"  Not enough global frames for GIF in required range for {run.id}.")
        return
        
    print(f"  Creating GIF for {run.id} with {len(global_files)} frames...")
    images = []
    
    for f in global_files:
        file_path = f.download(replace=True, root="/tmp/wandb_dl").name
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # Fetch default font or load one. PIL default might be too small, but it works.
            # Calculate Phase to find out if it's solitary or collective
            gen = extract_gen(f.name)
            
            if isinstance(sw_int, int) or (isinstance(sw_int, str) and sw_int.isdigit()):
                phase_idx = (gen // int(sw_int)) % 2
                phase_text = "Solitary" if phase_idx == 0 else "Collective"
            else:
                phase_text = "Static"
                
            # Draw text on image with a small bbox to ensure readability
            try:
                # Add text to bottom right or top left
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
            except IOError:
                font = ImageFont.load_default()
                
            # Outline/shadow for readability
            text_pos = (20, 20)
            draw.text((text_pos[0]-2, text_pos[1]-2), f"{phase_text} Phase (Gen {gen})", font=font, fill="white")
            draw.text((text_pos[0]+2, text_pos[1]-2), f"{phase_text} Phase (Gen {gen})", font=font, fill="white")
            draw.text((text_pos[0]-2, text_pos[1]+2), f"{phase_text} Phase (Gen {gen})", font=font, fill="white")
            draw.text((text_pos[0]+2, text_pos[1]+2), f"{phase_text} Phase (Gen {gen})", font=font, fill="white")
            draw.text(text_pos, f"{phase_text} Phase (Gen {gen})", font=font, fill="black")
            
            images.append(np.array(img))
            
    if images:
        gif_name = f"ternary_density_{sel_name}_sw{sw_int}_mut{mut_prob}_{run.id}.gif"
        gif_path = os.path.join(OUTPUT_DIR, gif_name)
        imageio.mimsave(gif_path, images, fps=5)
        print(f"  Saved GIF: {gif_path}")

process_and_plot(runs)
