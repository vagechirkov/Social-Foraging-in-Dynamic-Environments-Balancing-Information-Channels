#!/bin/bash

belief_selectivity_array=(1.0 3.0 6.0)
gamma_belief_array=(0.01 0.1)  #  1 10
dim_array=(10)  # 5  10 15
n_agent_array=(10)  # 20 30
target_speed_array=(0.1)  #  0.3 0.5
cost_priv=(0.9)  # 0.5 0.1 0.02
cost_belief=(0.005 0.01 0.05)  # 0.5 0.1 0.02
base_noise_array=(1)
dist_noise_scale_priv_array=(4 6)  # 2 4 6
target_persistence_array=(1 5)

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
                                        sbatch 3_channel_cpu_hpc_itb.sh "$t_speed" "$c_priv" "$c_bel" "$dim" "$n_agents" "$gamma_belief" "$belief_selectivity" "$base_noise" "$dist_noise_scale_priv" "$target_persistence"
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