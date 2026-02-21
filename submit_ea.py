import os
import subprocess
import itertools
from omegaconf import OmegaConf
import argparse

def submit_ea_jobs():
    parser = argparse.ArgumentParser(description="Submit EA jobs (Dynamic pairs and Static baselines).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--gpu", action="store_true", help="Submit to GPU partition instead of CPU.")
    parser.add_argument("--replicates", type=int, help="Number of replicates")
    parser.add_argument("--generations", type=int, help="Number of generations")
    parser.add_argument("--switch_interval", type=int, help="Switch interval for dynamic environments")
    parser.add_argument("--selection", type=str, choices=["individual-global", "individual-local"], help="Selection method (individual-global or individual-local)")
    parser.add_argument("--multi_level_selection", type=str, choices=["true", "false", "True", "False"], help="Enable multi-level selection (true/false)")
    parser.add_argument("--mutation_prob", type=float, help="Mutation probability")
    args = parser.parse_args()

    extra_args = []
    if args.replicates is not None:
        extra_args.append(f"evolution.replicates={args.replicates}")
    if args.generations is not None:
        extra_args.append(f"evolution.generations={args.generations}")
    if args.switch_interval is not None:
        extra_args.append(f"evolution.switch_interval={args.switch_interval}")
    if args.selection is not None:
        extra_args.append(f"evolution.selection={args.selection}")
    if args.multi_level_selection is not None:
        extra_args.append(f"evolution.multi_level_selection={args.multi_level_selection.lower()}")
    if args.mutation_prob is not None:
        extra_args.append(f"evolution.mutation_prob={args.mutation_prob}")

    # Load the configuration
    config_path = "abm/ea_evaluation.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    cfg = OmegaConf.load(config_path)
    
    categories = list(cfg.environments.categories.keys())
    
    if args.gpu:
        sbatch_script = "ea_gpu_hpc_scioi.sh"
    else:
        sbatch_script = "ea_cpu_hpc_itb.sh"
    
    print(f"Found categories: {categories}")
    print(f"Using submit script: {sbatch_script}")

    # 1. Dynamic Environment Pipeline (Pairwise)
    print("\n--- Submitting Dynamic Environment Pipeline (Pairwise) ---")
    
    # Generate all pairs of environments
    # pairs = [("solitary", "collective")]
    pairs = list(itertools.combinations(categories, 2))
    
    for pair in pairs:
        pair_str = f"{pair[0]}-{pair[1]}"
        print(f"Processing pair: {pair_str}")
        
        # Command arguments for ea_cpu_hpc_itb.sh / ea_gpu_hpc_scioi.sh
        # $1: mode (dynamic)
        # $2: category (pair_str)
        
        cmd = ["sbatch", sbatch_script, "dynamic", pair_str] + extra_args
        
        if args.dry_run:
            print(f"  Command: {' '.join(cmd)}")
        else:
            subprocess.run(cmd)

    # 2. Static Environment Pipelines (Baselines)
    print("\n--- Submitting Static Environment Pipelines (Baselines) ---")
    
    for cat in categories:
        print(f"Processing static category: {cat}")
        
        # Command arguments for ea_cpu_hpc_itb.sh / ea_gpu_hpc_scioi.sh
        # $1: mode (static)
        # $2: category (cat)
        
        cmd = ["sbatch", sbatch_script, "static", cat] + extra_args
        
        if args.dry_run:
            print(f"  Command: {' '.join(cmd)}")
        else:
            subprocess.run(cmd)

if __name__ == "__main__":
    submit_ea_jobs()
