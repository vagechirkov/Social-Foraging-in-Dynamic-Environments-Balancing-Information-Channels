import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

# Configuration for the curves
curves_config = [
    # {
    #     "run_path": "chirkov/3_channels_abm/y6qd48ad",
    #     "key": "focal/social_mix_p0.40_b0.60/full_run/avg_fitness",
    #     "start": 0,
    #     "end": 990
    # },
    # {
    #     "run_path": "chirkov/3_channels_abm/y6qd48ad",
    #     "key": "focal/social_mix_p0.40_b0.60/full_run/avg_fitness",
    #     "start": 1000,
    #     "end": 1990
    # },
    # {
    #     "run_path": "chirkov/3_channels_abm/y6qd48ad",
    #     "key": "focal/asocial_p1.00_b0.00/full_run/avg_fitness",
    #     "start": 1000,
    #     "end": 1990
    # },
    {
        "run_path": "chirkov/ssga_2_targets/nftwovc4",  # 1000, cost 0.1 both
        "key": "tick/avg_fitness",
        "start": 199000,
        "end": 199990,
        "label": "R=1"
    },
    {
        "run_path": "chirkov/ssga_2_targets/90u21lkg",  # 1000, cost 0.1 both, history cleaned
        "key": "tick/avg_fitness",
        "start": 199000,
        "end": 199990,
        "label": "R=1, HR"
    },
    {
        "run_path": "chirkov/ssga_2_targets/hrw0yri8",  # 20_000, cost 0.1 both
        "key": "tick/avg_fitness",
        "start": 180000,
        "end": 180990,
        "label": "R=20"
    },
    {
        "run_path": "chirkov/ssga_2_targets/xm5yqik7",  # 50_000, cost 0.1 both
        "key": "tick/avg_fitness",
        "start": 150000,
        "end": 150990,
        "label": "R=50"
    },
    {
        "run_path": "chirkov/ssga_2_targets/fyd61jgo",  # MLS, 1000, cost 0.1 both
        "key": "tick/avg_fitness",
        "start": 199000,
        "end": 199990,
        "label": "MLS, R=1"
    },
    {
        "run_path": "chirkov/ssga_2_targets/jukv3alm",  # MLS, 1000, cost 0.1 both, history cleaned
        "key": "tick/avg_fitness",
        "start": 199000,
        "end": 199990,
        "label": "MLS, R=1, HR"
    },
    {
        "run_path": "chirkov/ssga_2_targets/5kxbfx9r",  # MLS, 20_000, cost 0.1 both
        "key": "tick/avg_fitness",
        "start": 180000,
        "end": 180990,
        "label": "MLS, R=20"
    },
    {
        "run_path": "chirkov/ssga_2_targets/9flryakq",  # MLS, 50_000, cost 0.1 both
        "key": "tick/avg_fitness",
        "start": 150000,
        "end": 150990,
        "label": "MLS, R=50"
    }
]

def main():
    # Scientific style settings
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "--"
    })
    
    # Wong's colorblind palette
    # "orange, sky blue, green, yellow, blue, red, pink, black"
    cb_palette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    plt.rcParams['axes.prop_cycle'] = cycler(color=cb_palette)

    api = wandb.Api()
    
    fig, ax = plt.subplots(figsize=(7, 4))

    for idx, config in enumerate(curves_config):
        print(f"Fetching curve {idx+1}: {config['run_path']} -> {config['key']} (steps {config['start']} to {config['end']})")
        run = api.run(config['run_path'])
        
        # We use scan_history to get all rows without sampling
        data = []
        for row in run.scan_history(keys=['_step', config['key']]):
            step = row.get('_step')
            val = row.get(config['key'])
            
            if step is not None and val is not None:
                if config['start'] <= step <= config['end']:
                    data.append({'_step': step, config['key']: val})
                    
        df = pd.DataFrame(data)
        
        if df.empty:
            print(f"  -> No data found for {config['key']} between {config['start']} and {config['end']}")
            continue
            
        # Sort by step to ensure correct plotting
        df = df.sort_values('_step')

        # Create a normalized step so all curves start at 0
        df['normalized_step'] = df['_step'] - config['start']
        
        # Plot without smoothing
        ax.plot(df['normalized_step'], df[config['key']], linewidth=1.5, alpha=0.9, label=config.get('label', ''))

    ax.set_xlabel("Relative Time Step")
    ax.set_ylabel("Average Fitness")
    # ax.set_title("Average Fitness from wandb Runs")

    # Put the legend
    ax.legend()

    # Save the plot as png
    output_filename = "wandb_curves_3.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved as {output_filename}")

if __name__ == "__main__":
    main()
