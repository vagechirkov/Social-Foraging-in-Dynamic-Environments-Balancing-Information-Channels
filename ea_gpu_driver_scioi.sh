#!/bin/bash

dims_array=(5)
target_speed_array=(0.05 0.1 0.3 0.5)

for dim in "${dims_array[@]}"; do
    for t_speed in "${target_speed_array[@]}"; do
        sbatch ea_gpu_hpc_scioi.sh "$t_speed" "$dim"
        sleep 1
    done
done