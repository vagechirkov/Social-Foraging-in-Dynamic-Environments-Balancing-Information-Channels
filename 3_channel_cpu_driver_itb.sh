#!/bin/bash

target_speed_array=(0.1 0.3 0.5)
cost_priv=(0.5 0.1 0.02)
cost_belief=(0.5 0.1 0.02)

  for t_speed in "${target_speed_array[@]}"; do
    for c_priv in "${cost_priv[@]}"; do
        for c_bel in "${cost_belief[@]}"; do
            sbatch 3_channel_cpu_hpc_itb.sh "$t_speed" "$c_priv" "$c_bel"
        done
    done
done