#!/bin/bash

target_speed_array=(0.1 0.3 0.5)
gamma_bel_array=(1.0 2.0 5.0)
beta_bel_array=(1.0 0.1)

for beta_bel in "${beta_bel_array[@]}"; do
    for gamma_bel in "${gamma_bel_array[@]}"; do
        for t_speed in "${target_speed_array[@]}"; do
            sbatch 3_channel_cpu_hpc_itb.sh "$t_speed" "$gamma_bel" "$beta_bel"
        done
    done
done