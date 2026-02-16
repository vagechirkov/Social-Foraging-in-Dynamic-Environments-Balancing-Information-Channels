#!/bin/bash

# 1. Dynamic Environment Pipeline
echo "Submitting Dynamic Environment Pipeline..."
# We use a specific run name to identify it easily in W&B
sbatch ea_cpu_hpc_itb.sh "dynamic" "baseline"

# 2. Static Environment Pipelines (Baselines)
# We ignore the second argument for static, or pass it as the specific category
categories=("baseline" "high_cost" "noisy_private" "fast_target")

echo "Submitting Static Environment Pipelines..."
for cat in "${categories[@]}"; do
    sbatch ea_cpu_hpc_itb.sh "static" "$cat"
done