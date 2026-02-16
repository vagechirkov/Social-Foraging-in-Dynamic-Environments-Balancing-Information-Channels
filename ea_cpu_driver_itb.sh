#!/bin/bash

# 1. Dynamic Environment Pipeline (Pairwise)
echo "Submitting Dynamic Environment Pipeline (Pairwise)..."

# Generate all pairs of environments
# using python to get combinations of ["baseline", "noisy_private", "fast_target"]
pairs=$(python3 -c 'import itertools; keys = ["baseline", "noisy_private", "fast_target"]; print(" ".join([f"{c[0]}-{c[1]}" for c in itertools.combinations(keys, 2)]))')

echo "Pairs to run: $pairs"

for pair in $pairs; do
    echo "Submitting pair: $pair"
    sbatch ea_cpu_hpc_itb.sh "dynamic" "$pair"
done

# 2. Static Environment Pipelines (Baselines)
# We ignore the second argument for static, or pass it as the specific category
categories=("baseline" "noisy_private" "fast_target")

echo "Submitting Static Environment Pipelines..."
for cat in "${categories[@]}"; do
    sbatch ea_cpu_hpc_itb.sh "static" "$cat"
done