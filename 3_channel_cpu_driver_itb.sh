#!/bin/bash

belief_selectivity_array=(2.0 4.0)
gamma_belief_array=(1)
dim_array=(5)  # 10 15
n_agent_array=(10)  # 20 30
target_speed_array=(0.1 0.5)
cost_priv=(0.9 0.5 0.1 0.02)
cost_belief=(0.5 0.1 0.02)


for belief_selectivity in "${belief_selectivity_array[@]}"; do
    for gamma_belief in "${gamma_belief_array[@]}"; do
        for n_agents in "${n_agent_array[@]}"; do
            for dim in "${dim_array[@]}"; do
                for t_speed in "${target_speed_array[@]}"; do
                    for c_priv in "${cost_priv[@]}"; do
                        for c_bel in "${cost_belief[@]}"; do
                            sbatch 3_channel_cpu_hpc_itb.sh "$t_speed" "$c_priv" "$c_bel" "$dim" "$n_agents" "$gamma_belief" "$belief_selectivity"
                        done
                    done
                done
            done
        done
    done
done