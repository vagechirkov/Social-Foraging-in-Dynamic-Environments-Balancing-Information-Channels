import os
import subprocess
import yaml
from omegaconf import OmegaConf

import argparse

def submit_jobs():
    parser = argparse.ArgumentParser(description="Submit EA jobs for each environment category.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    # Load the configuration
    config_path = "abm/ea_evaluation.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    cfg = OmegaConf.load(config_path)
    
    # Common Parameters (defaults from config or script constants)
    # Read defaults directly from the loaded YAML configuration
    defaults = {
        "dim": cfg.x_dim,
        "n_agents": cfg.n_agents,
        "n_targets": cfg.n_targets,
        "target_persistence": cfg.target_persistence,
        "relocation_interval": cfg.relocation_interval,
        "base_noise": cfg.base_noise,
        "process_noise_scale": cfg.process_noise_scale,
        "gamma_belief": cfg.social_trans_scale, # Mapping social_trans_scale to gamma_belief
        "belief_selectivity_array": str(cfg.belief_selectivity_threshold),
        "chan_y": "Belief",
        "process_noise_scale_het_ratio": cfg.process_noise_scale_het_ratio,
        "process_noise_scale_het_scale": cfg.process_noise_scale_het_scale,
        "cost_consensus": cfg.cost_consensus,
        "consensus_selectivity_threshold": cfg.consensus_selectivity_threshold,
        "bias_magnitude": cfg.bias_magnitude,
        "sbatch_script": "3_channel_cpu_hpc_itb.sh"
    }

    categories = cfg.environments.categories

    for category_name, params in categories.items():
        print(f"Processing category: {category_name}")
        
        # Extract specific parameters for the category
        t_speed = params.get("target_speed", 0.1) 
        cost_priv = params.get("cost_priv", 0.5)
        cost_belief = params.get("cost_belief", 0.1)
        spot_radius = params.get("spot_radius", 0.5)
        dist_noise_scale_priv = params.get("dist_noise_scale_priv", 0.5)
        
        if args.dry_run:
            print(f"  Target Speed: {t_speed}")
            print(f"  Cost Private: {cost_priv}")
            print(f"  Cost Belief: {cost_belief}")
            print(f"  Spot Radius: {spot_radius}")
            print(f"  Dist Noise Priv: {dist_noise_scale_priv}")

        cmd = [
            "sbatch", defaults["sbatch_script"],
            str(t_speed),
            str(cost_priv),
            str(cost_belief),
            str(defaults["dim"]),
            str(defaults["n_agents"]),
            str(defaults["gamma_belief"]),
            str(defaults["belief_selectivity_array"]),
            str(defaults["base_noise"]),
            str(dist_noise_scale_priv),
            str(defaults["target_persistence"]),
            str(defaults["process_noise_scale"]),
            str(defaults["relocation_interval"]),
            str(defaults["n_targets"]),
            str(defaults["process_noise_scale_het_ratio"]),
            str(defaults["process_noise_scale_het_scale"]),
            str(defaults["cost_consensus"]),
            str(defaults["consensus_selectivity_threshold"]),
            str(defaults["chan_y"]),
            str(defaults["bias_magnitude"]),
            str(spot_radius),
            str(category_name)
        ]
        
        # Execute
        if args.dry_run:
            print(f"  Command: {' '.join(cmd)}")
            print("-" * 40)
        else:
             subprocess.run(cmd)

if __name__ == "__main__":
    submit_jobs()
