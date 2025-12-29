#!/bin/bash

#SBATCH --job-name=ea-collective-tracking           # Job name
#SBATCH --output=job_%j.log                         # Output log file (%j will be replaced with job ID)
#SBATCH --error=job_%j.log                          # Error log file
#SBATCH --partition=long                            # Partition to submit to
#SBATCH --nodes=1                                   # Ensure it runs on one node
#SBATCH --ntasks=1                                  # Run a single task
#SBATCH --cpus-per-task=64                          # Request 64 CPUs
#SBATCH --mem=128G                                   # Memory allocation


source .venv/bin/activate

# add current directory to python path
export PYTHONPATH=$PYTHONPATH:.

t_speed=$1

uv run abm/info_channels_ea.py \
    --n_agents 100 \
    --target_speed="$t_speed" \
    --episode_len 3000 \
    --pop_size 64 \
    --ngen 1000 \
    --top_k 5 \
    --dim 3 \
    --costs 0.05 0.02 0.01 0.005 \
    --use_wandb \
    --run_name="init_explor"
