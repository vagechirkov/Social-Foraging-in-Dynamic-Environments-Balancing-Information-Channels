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
import math

WANDB_PROJECT = "dynamic_evolution_v1"
OUTPUT_DIR = "outputs/publication_plots"
CACHE_FILE = os.path.join(OUTPUT_DIR, "metrics_cache.csv")
HIST_CACHE_FILE = os.path.join(OUTPUT_DIR, "hist_cache.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)

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
    if os.path.exists(CACHE_FILE) and os.path.exists(HIST_CACHE_FILE):
        print("\n--- Loading Metrics from Cache ---")
        df = pd.read_csv(CACHE_FILE)
        df_hist = pd.read_csv(HIST_CACHE_FILE)
        # Drop rows where std could not be calculated
        df_hist = df_hist.dropna(subset=['Private_SD', 'Belief_SD', 'None_SD'])
        valid_runs = [run for run in runs if run.state in ["finished", "running"]]
    else:
        print("\n--- Fetching Metrics ---")
        all_metrics_data = []
        all_hist_data = [] # We'll store mean and std directly instead of samples to fix SD shade rendering
        valid_runs = []
        for run in runs:
            if run.state not in ["finished", "running"]:
                print(f"Skipping run {run.id} as it is in state {run.state}")
                continue

            valid_runs.append(run)
            print(f"Fetching metrics for run: {run.id} ({run.name}) [state: {run.state}]")
            
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
            
            # Use `hist/xxx` as requested
            keys_to_fetch = [
                "hist/prob_global_Priv",
                "hist/prob_global_Belief",
                "hist/prob_global_None"
            ]
            
            num_runs = cfg.get("num_runs", 1)
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
                
                # Global Hist Data parsing
                priv_hist = metrics.get(f"hist/prob_global_Priv")
                belief_hist = metrics.get(f"hist/prob_global_Belief")
                none_hist = metrics.get(f"hist/prob_global_None")
                
                def calc_hist_stats(h):
                    if not isinstance(h, dict) or 'bins' not in h or 'values' not in h:
                        return np.nan, np.nan
                    bins = np.array(h['bins'])
                    values = np.array(h['values'])
                    midpoints = (bins[:-1] + bins[1:]) / 2.0
                    total = values.sum()
                    if total == 0:
                        return np.nan, np.nan
                    mean = np.sum(midpoints * values) / total
                    var = np.sum(values * (midpoints - mean)**2) / total
                    return mean, math.sqrt(var)

                p_mean, p_sd = calc_hist_stats(priv_hist)
                b_mean, b_sd = calc_hist_stats(belief_hist)
                n_mean, n_sd = calc_hist_stats(none_hist)
                
                if not np.isnan(p_sd) and not np.isnan(b_sd) and not np.isnan(n_sd):
                   all_hist_data.append({
                       "mode": mode,
                       "switch_interval": sw_int,
                       "mutation_prob": mut_prob,
                       "selection": sel_name,
                       "Generation": step,
                       "Private_Mean": p_mean,
                       "Private_SD": p_sd,
                       "Belief_Mean": b_mean,
                       "Belief_SD": b_sd,
                       "None_Mean": n_mean,
                       "None_SD": n_sd
                   })

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
        df_hist = pd.DataFrame(all_hist_data)
        df.to_csv(CACHE_FILE, index=False)
        df_hist.to_csv(HIST_CACHE_FILE, index=False)
        print(f"Metrics cached.")

    print("\n--- Generating Plots ---")
    
    channel_base_colors = {
        "Private": {"dynamic": "#009E73", "static-solitary": "#70deb8", "static-collective": "#00664b"},
        "Belief": {"dynamic": "#56B4E9", "static-solitary": "#a6daff", "static-collective": "#1a7ab0"},
        "None": {"dynamic": "#E69F00", "static-solitary": "#ffd275", "static-collective": "#cc8d00"}
    }
    
    # Plotting styles - list of pixels for seaborn dashes. [dash_points, space_points], or "" for solid line.
    line_dashes = {"dynamic": [2, 2], "static-solitary": "", "static-collective": ""}
    
    # Pre-process dataset to be combined dynamically correctly
    plot_df = []
    
    for mut_prob, mut_group in df.groupby("mutation_prob"):
        dynamic_intervals = [sw for sw in mut_group["switch_interval"].unique() if isinstance(sw, (int, float)) or (isinstance(sw, str) and sw.isdigit())]
        
        for dyn_sw in dynamic_intervals:
            target_intervals = [dyn_sw, "static-solitary", "static-collective"]
            sub_df = mut_group[mut_group["switch_interval"].isin(target_intervals)].copy()
            
            sub_df["Environment"] = sub_df["switch_interval"].apply(
                lambda x: "dynamic" if str(x) == str(dyn_sw) else str(x)
            )
            sub_df["Dynamic_Switch"] = int(dyn_sw) # Keep track of which grid panel this belongs to
            
            plot_df.append(sub_df)
            
    plot_df = pd.concat(plot_df)
    
    # Ensure standard ordering for selection panels
    sel_order = ["individual-global", "multilevel-local"]
            
    for mut_prob, sub_mut_group in plot_df.groupby("mutation_prob"):
        print(f"Plotting for mutation_prob={mut_prob}")

        # --- Fitness Plot 2x2 Grid (Rows = Selection, Cols = Switch Interval)
        g = sns.relplot(
            data=sub_mut_group, 
            kind="line",
            x="Generation", 
            y="Fitness", 
            row="selection", 
            col="Dynamic_Switch",
            hue="Environment",
            style="Environment",
            dashes=line_dashes,
            errorbar=('ci', 95),
            n_boot=30,
            palette={"dynamic": "#D55E00", "static-collective": "#0072B2", "static-solitary": "#009E73"},
            height=6, aspect=2, # Wider figures
            row_order=sel_order,
            col_order=[100, 300],
            linewidth=2.5,
            facet_kws={'margin_titles': True}
        )
        g.set(ylim=(0.4, 0.7), xlim=(0, 3000))
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        
        sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.05), title="", ncol=3)

        # We configure ticks dynamically based on the column
        for ax in g.axes.flat:
            # Try to grab the col title (e.g. Dynamic_Switch = 100)
            col_match = re.search(r'Dynamic_Switch\s*=\s*(\d+)', ax.get_title())
            if col_match:
                dyn_sw_int = int(col_match.group(1))
                tick_interval = dyn_sw_int * 3 if dyn_sw_int == 100 else dyn_sw_int
            else:
                tick_interval = 300  # Fallback

            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
            ax.tick_params(axis='x', rotation=45)
        
        plt.savefig(os.path.join(OUTPUT_DIR, f"fitness_grid_mut{mut_prob}.png"), dpi=300, bbox_inches='tight')
        plt.close(g.fig)

        # --- Channel Probs Plots CI (Average) 2x2 Grid
        for channel in ["Private", "Belief", "None"]:
            g_ci = sns.relplot(
                data=sub_mut_group,
                kind="line",
                x="Generation",
                y=channel,
                row="selection",
                col="Dynamic_Switch",
                hue="Environment",
                style="Environment",
                dashes=line_dashes,
                errorbar=('ci', 95),
                n_boot=30,
                palette=channel_base_colors[channel],
                height=6, aspect=2,
                linewidth=2.5,
                row_order=sel_order,
                facet_kws={'margin_titles': True}
            )
            g_ci.set(ylim=(0, 1), xlim=(0, 3000))
            g_ci.set_titles(row_template="{row_name}", col_template="{col_name}")
            sns.move_legend(g_ci, "lower center", bbox_to_anchor=(0.5, -0.05), title="", ncol=3)

            for ax in g_ci.axes.flat:
                col_match = re.search(r'Dynamic_Switch\s*=\s*(\d+)', ax.get_title())
                if col_match:
                    dyn_sw_int = int(col_match.group(1))
                    tick_interval = dyn_sw_int * 3 if dyn_sw_int == 100 else dyn_sw_int
                else:
                    tick_interval = 300
                ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
                ax.tick_params(axis='x', rotation=45)
                
            plt.savefig(os.path.join(OUTPUT_DIR, f"channel_probs_ci_{channel}_grid_mut{mut_prob}.png"), dpi=300, bbox_inches='tight')
            plt.close(g_ci.fig)
            
    # Process custom SD plots separately since they use standard deviations directly calculated over WandB global distributions
    # We must melt differently to plot actual means with SD ranges
    
    plot_sd_df = []
    for mut_prob, mut_group in df_hist.groupby("mutation_prob"):
        dynamic_intervals = [sw for sw in mut_group["switch_interval"].unique() if isinstance(sw, (int, float)) or (isinstance(sw, str) and sw.isdigit())]
        
        for dyn_sw in dynamic_intervals:
            target_intervals = [dyn_sw, "static-solitary", "static-collective"]
            sub_hist_df = mut_group[mut_group["switch_interval"].isin(target_intervals)].copy()
            
            sub_hist_df["Environment"] = sub_hist_df["switch_interval"].apply(
                lambda x: "dynamic" if str(x) == str(dyn_sw) else str(x)
            )
            sub_hist_df["Dynamic_Switch"] = dyn_sw
            plot_sd_df.append(sub_hist_df)
            
    if plot_sd_df:
        plot_sd_df = pd.concat(plot_sd_df)
        
        for mut_prob, sub_sd_mut_group in plot_sd_df.groupby("mutation_prob"):
            for channel in ["Private", "Belief", "None"]:
                
                # To plot SD as an envelope using sns, we need upper/lower bounds.
                # However sns.relplot directly doesn't allow parsing explicit upper/lower bounds elegantly.
                # We will draw it manually on a FacetGrid
                
                g_sd = sns.FacetGrid(
                    data=sub_sd_mut_group, 
                    row="selection", 
                    col="Dynamic_Switch",
                    hue="Environment",
                    palette=channel_base_colors[channel],
                    height=6, aspect=2,
                    row_order=sel_order,
                    margin_titles=True
                )
                
                # Define plotting function map
                def plot_sd_envelope(data, **kwargs):
                    ax = plt.gca()
                    env = data['Environment'].iloc[0]
                    color = channel_base_colors[channel][env]
                    ls = line_dashes[env] if line_dashes[env] != "" else "solid"
                    if ls != "solid":
                        # Convert to loosely dashed tuple expected by plt.plot dashes arg if it's not a generic style string
                        pass
                    
                    # Sort data
                    data = data.sort_values(by="Generation")
                    
                    # Calculate mean across rows for the exact same generation
                    # (since we collapsed all runs into separate distributions)
                    g_data = data.groupby("Generation").agg({
                        f"{channel}_Mean": "mean",
                        f"{channel}_SD": "mean" 
                    }).reset_index()
                    
                    gens = g_data["Generation"]
                    means = g_data[f"{channel}_Mean"]
                    sds = g_data[f"{channel}_SD"]
                    
                    if ls == "solid" or ls == "-":
                        ax.plot(gens, means, color=color, linestyle="-", linewidth=2.5, label=env)
                    else:
                        ax.plot(gens, means, color=color, dashes=line_dashes[env], linewidth=2.5, label=env)
                        
                    ax.fill_between(gens, np.clip(means - sds, 0, 1), np.clip(means + sds, 0, 1), color=color, alpha=0.3)
                
                g_sd.map_dataframe(plot_sd_envelope)
                g_sd.set(ylim=(0, 1), xlim=(0, 3000))
                g_sd.set_titles(row_template="{row_name}", col_template="{col_name}")
                g_sd.add_legend(title="")
                sns.move_legend(g_sd, "lower center", bbox_to_anchor=(0.5, -0.05), title="", ncol=3)
                
                for ax in g_sd.axes.flat:
                    col_match = re.search(r'Dynamic_Switch\s*=\s*(\d+)', ax.get_title())
                    if col_match:
                        dyn_sw_int = int(col_match.group(1))
                        tick_interval = dyn_sw_int * 3 if dyn_sw_int == 100 else dyn_sw_int
                    else:
                        tick_interval = 300
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
                    ax.tick_params(axis='x', rotation=45)
                    ax.set_ylabel(channel)
                    ax.set_xlabel("Generation")

                plt.savefig(os.path.join(OUTPUT_DIR, f"channel_probs_global_sd_{channel}_grid_mut{mut_prob}.png"), dpi=300, bbox_inches='tight')
                plt.close(g_sd.fig)

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
    gif_name = f"ternary_density_{sel_name}_sw{sw_int}_mut{mut_prob}_{run.id}.gif"
    gif_path = os.path.join(OUTPUT_DIR, gif_name)
    if os.path.exists(gif_path):
        print(f"  {gif_path} exists, skipping GIF generation per user request.")
        return

    # Keep logic just in case the file doesn't exist
    print(f"  Downloading ternary density plots for GIF for {run.id} (between gens 1800-2400)...")
    files = list(run.files())
    ternary_files = [f for f in files if "ternary_density_top_k" in f.name and f.name.endswith(".png")]
    if not ternary_files:
        return

    filtered_files = [f for f in ternary_files if 1800 <= extract_gen(f.name) <= 2400]
    filtered_files.sort(key=lambda x: extract_gen(x.name))
    
    global_files = [f for f in filtered_files if extract_prefix(f.name) == "global_"]
    if len(global_files) < 2:
        return
        
    print(f"  Creating GIF for {run.id} with {len(global_files)} frames...")
    images = []
    
    for f in global_files:
        file_path = f.download(replace=True, root="/tmp/wandb_dl").name
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            gen = extract_gen(f.name)
            if isinstance(sw_int, int) or (isinstance(sw_int, str) and sw_int.isdigit()):
                phase_idx = (gen // int(sw_int)) % 2
                phase_text = "Solitary" if phase_idx == 0 else "Collective"
            else:
                phase_text = "Static"
                
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
            except IOError:
                font = ImageFont.load_default()
                
            # Place textual label upper-middle left, just Solitary/Collective
            text_pos = (20, 150)
            text_str = phase_text
            draw.text((text_pos[0]-2, text_pos[1]-2), text_str, font=font, fill="black")
            draw.text((text_pos[0]+2, text_pos[1]-2), text_str, font=font, fill="black")
            draw.text((text_pos[0]-2, text_pos[1]+2), text_str, font=font, fill="black")
            draw.text((text_pos[0]+2, text_pos[1]+2), text_str, font=font, fill="black")
            draw.text(text_pos, text_str, font=font, fill="white")
            
            images.append(np.array(img))
            
    if images:
        # Loop GIFs loop=0 means infinite loop
        imageio.mimsave(gif_path, images, fps=5, loop=0)
        print(f"  Saved GIF: {gif_path}")

process_and_plot(runs)
