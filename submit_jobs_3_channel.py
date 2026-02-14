import subprocess
import itertools
import sys
import os

# --- Parameter Definitions (Matching 3_channel_cpu_driver_itb.sh) ---

# Consensus Parameters
consensus_selectivity_array = [1.0]

# Belief Parameters
belief_selectivity_array = [0.5, 1.0]
gamma_belief_array = [0.01]  # Options: 0.1 1 10

# Channel Mode
channel_y_name = "Belief"

# Agent & Environment Dimensions
dim_array = [2]  # Options: 5 10 15
n_agent_array = [15]  # Options: 20 30

# Target
n_targets_array = [1]
target_speed_array = [0.1, 0.5, 0.9]  # Options: 0.1 0.3 0.5
target_persistence_array = [25]
relocation_interval_array = [1000, 2500]

# Costs
cost_priv_array = [0.3]  # Options: 0.5 0.1 0.02
cost_belief_array = [0.05]  # Options: 0.01 0.05 0.5 0.1 0.02
cost_consensus_array = [0.01]

# Noise
base_noise_array = [0.1]
dist_noise_scale_priv_array = [0.05]  # 0.5 1.0
process_noise_scale_array = [0.05]  # 0.1 0.5 1.0
process_noise_scale_het_ratio_array = [0]  # 0.5 0.2 0.8
process_noise_scale_het_scale_array = [10]  # 100
bias_magnitude_array = [0.0]
spot_radius_array = [0.5]

def submit_jobs(dry_run=False):
    # Create all combinations
    # The order here matches the nesting order in the shell script:
    # n_targets, process_noise_scale, target_persistence, dist_noise_scale_priv, base_noise,
    # belief_selectivity, gamma_belief, n_agents, dim, t_speed, c_priv, c_bel, rel_int,
    # process_noise_scale_het_ratio, process_noise_scale_het_scale, cost_consensus,
    # consensus_selectivity, bias_magnitude, spot_radius
    
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
        spot_radius_array
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
        spot_radius
    ) in combinations:
        
        count += 1
        
        # Args for 3_channel_cpu_hpc_itb.sh in order:
        # 1. t_speed
        # 2. cost_priv
        # 3. cost_belief
        # 4. dim
        # 5. n_agents
        # 6. gamma_belief
        # 7. belief_selectivity_array (passed as scalar in loop)
        # 8. base_noise
        # 9. dist_noise_scale_priv
        # 10. target_persistence
        # 11. process_noise_scale
        # 12. relocation_interval
        # 13. n_targets
        # 14. process_noise_scale_het_ratio
        # 15. process_noise_scale_het_scale
        # 16. cost_consensus
        # 17. consensus_selectivity_threshold
        # 18. channel_y_name
        # 19. bias_magnitude

        args = [
            str(t_speed),
            str(c_priv),
            str(c_bel),
            str(dim),
            str(n_agents),
            str(gamma_belief),
            str(belief_selectivity),
            str(base_noise),
            str(dist_noise_scale_priv),
            str(target_persistence),
            str(process_noise_scale),
            str(rel_int),
            str(n_targets),
            str(process_noise_scale_het_ratio),
            str(process_noise_scale_het_scale),
            str(cost_consensus),
            str(consensus_selectivity),
            str(channel_y_name),
            str(bias_magnitude),
            str(spot_radius)
        ]
        
        cmd = ["sbatch", "3_channel_cpu_hpc_itb.sh"] + args

        if dry_run:
            print(f"Combination {count}:")
            print(" ".join(cmd))
            print("-" * 20)
        else:
            print(f"Submitting job {count}...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job {count}: {e}")
                # Optional: continue or break
                continue

    print(f"Total jobs {'generated' if dry_run else 'submitted'}: {count}")

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    submit_jobs(dry_run=dry_run)
