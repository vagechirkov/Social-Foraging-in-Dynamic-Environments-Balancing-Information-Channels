#!/bin/bash

target_speed_array=(0.1 0.3 0.5)

for t_speed in "${target_speed_array[@]}"; do
    sbatch ea_cpu_hpc_itb.sh "$t_speed"
done