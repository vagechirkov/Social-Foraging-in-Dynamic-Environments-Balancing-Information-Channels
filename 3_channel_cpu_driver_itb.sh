#!/bin/bash

# Belief Parameters
belief_selectivity_array=(0.1 0.5 1.0)
gamma_belief_array=(0.01)                       # Options: 0.1 1 10

# Agent & Environment Dimensions
dim_array=(5)                                   # Options: 5 10 15
n_agent_array=(10)                              # Options: 20 30

# Target
n_targets_array=(3 1)
target_speed_array=(0.05)                       # Options: 0.3 0.5
target_persistence_array=(20)
relocation_interval_array=(500 1000)

# Costs
cost_priv=(0.5)                                 # Options: 0.5 0.1 0.02
cost_belief=(0.05)                              # Options: 0.01 0.05 0.5 0.1 0.02

# Noise
base_noise_array=(0.1)
dist_noise_scale_priv_array=(0.1 0.5 1.0)   # Options: 0 - inf
process_noise_scale_array=(0.1 0.5 1.0)
process_noise_scale_het_ratio_array=(0.5 0.2 0.8)
process_noise_scale_het_scale_array=(10 100)


# --- Execution Loops ---
for n_targets in "${n_targets_array[@]}"; do
  for process_noise_scale in "${process_noise_scale_array[@]}"; do
    for target_persistence in "${target_persistence_array[@]}"; do
      for dist_noise_scale_priv in "${dist_noise_scale_priv_array[@]}"; do
        for base_noise in "${base_noise_array[@]}"; do
          for belief_selectivity in "${belief_selectivity_array[@]}"; do
            for gamma_belief in "${gamma_belief_array[@]}"; do
              for n_agents in "${n_agent_array[@]}"; do
                for dim in "${dim_array[@]}"; do
                  for t_speed in "${target_speed_array[@]}"; do
                    for c_priv in "${cost_priv[@]}"; do
                      for c_bel in "${cost_belief[@]}"; do
                        for rel_int in "${relocation_interval_array[@]}"; do
                          for process_noise_scale_het_ratio in "${process_noise_scale_het_ratio_array[@]}"; do
                            for process_noise_scale_het_scale in "${process_noise_scale_het_scale_array[@]}"; do

                              # Submit job with arguments explicitly listed on new lines
                              sbatch 3_channel_cpu_hpc_itb.sh \
                                  "$t_speed" \
                                  "$c_priv" \
                                  "$c_bel" \
                                  "$dim" \
                                  "$n_agents" \
                                  "$gamma_belief" \
                                  "$belief_selectivity" \
                                  "$base_noise" \
                                  "$dist_noise_scale_priv" \
                                  "$target_persistence" \
                                  "$process_noise_scale" \
                                  "$rel_int" \
                                  "$n_targets" \
                                  "$process_noise_scale_het_ratio" \
                                  "$process_noise_scale_het_scale"

                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done