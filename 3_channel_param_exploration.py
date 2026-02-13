import subprocess
import itertools
import sys
import os

# --- Parameter Definitions ---
# Consensus Parameters
consensus_selectivity_array = [1.0]

# Belief Parameters
belief_selectivity_array = [100.0]
gamma_belief_array = [0.01]  # Options: 0.1, 1, 10

# Channel Mode
channel_y_name_array = ["Belief"]

# Agent & Environment Dimensions
dim_array = [2]  # Options: 5, 10, 15
n_agent_array = [10]  # Options: 20, 30

# Target
n_targets_array = [1]
target_speed_array = [0.7]  # Options: 0.1, 0.3, 0.5
target_persistence_array = [20]
relocation_interval_array = [200]

# Costs
cost_priv_array = [0.5]  # Options: 0.5, 0.1, 0.02
cost_belief_array = [0.1, 0.5]  # Options: 0.01, 0.05, 0.5, 0.1, 0.02
cost_consensus_array = [0.01]

# Noise
base_noise_array = [0.1]
dist_noise_scale_priv_array = [0.1]  # 0.5, 1.0
process_noise_scale_array = [0.05]  # 0.1, 0.5, 1.0
process_noise_scale_het_ratio_array = [0]  # 0.5, 0.2, 0.8
process_noise_scale_het_scale_array = [10]  # 100
bias_magnitude_array = [0.1, 0.5, 1.0]

def run_exploration(dry_run=False):
    # Ensure current directory is in PYTHONPATH
    env = os.environ.copy()
    cwd = os.getcwd()
    env["PYTHONPATH"] = f"{cwd}:{env.get('PYTHONPATH', '')}"

    # Create all combinations
    # The order here matches the nesting order in the shell script:
    combinations = itertools.product(
        n_targets_array,
        process_noise_scale_array,
        target_persistence_array,
        dist_noise_scale_priv_array,
        base_noise_array,
        belief_selectivity_array,
        gamma_belief_array,
        n_agent_array,
        dim_array,
        target_speed_array,
        cost_priv_array,
        cost_belief_array,
        relocation_interval_array,
        process_noise_scale_het_ratio_array,
        process_noise_scale_het_scale_array,
        cost_consensus_array,
        consensus_selectivity_array,
        bias_magnitude_array,
        channel_y_name_array
    )

    count = 0
    for (
        n_targets,
        process_noise_scale,
        target_persistence,
        dist_noise_scale_priv,
        base_noise,
        belief_selectivity,
        gamma_belief,
        n_agents,
        dim,
        t_speed,
        c_priv,
        c_bel,
        rel_int,
        process_noise_scale_het_ratio,
        process_noise_scale_het_scale,
        cost_consensus,
        consensus_selectivity,
        bias_magnitude,
        channel_y_name
    ) in combinations:
        
        count += 1
        
        # Construct the command
        cmd = [
            "python", "abm/3_channels_abm_exploration.py",
            f"n_agents={n_agents}",
            f"n_targets={n_targets}",
            "max_steps=5000",
            "replicates=100",
            "run_name=costs",
            f"target_speed={t_speed}",
            f"cost_priv={c_priv}",
            f"cost_belief={c_bel}",
            f"x_dim={dim}",
            f"y_dim={dim}",
            f"social_trans_scale={gamma_belief}",
            f"belief_selectivity_threshold={belief_selectivity}",
            f"process_noise_scale={process_noise_scale}",
            f"base_noise={base_noise}",
            f"dist_noise_scale_priv={dist_noise_scale_priv}",
            f"target_persistence={target_persistence}",
            "target_movement_pattern=levy",
            f"relocation_interval={rel_int}",
            f"process_noise_scale_het_ratio={process_noise_scale_het_ratio}",
            f"process_noise_scale_het_scale={process_noise_scale_het_scale}",
            f"cost_consensus={cost_consensus}",
            f"consensus_selectivity_threshold={consensus_selectivity}",
            f"channel_y_name={channel_y_name}",
            f"bias_magnitude={bias_magnitude}"
        ]

        if dry_run:
            print(f"Combination {count}:")
            print(" ".join(cmd))
            print("-" * 20)
        else:
            print(f"Running combination {count}...")
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"Error running combination {count}: {e}")
                # decide if we want to stop or continue? 
                continue
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                sys.exit(1)

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_exploration(dry_run=dry_run)
