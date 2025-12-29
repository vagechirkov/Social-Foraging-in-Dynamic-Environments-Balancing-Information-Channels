#!/bin/bash

#SBATCH --job-name=ea-collective-tracking           # Job name
#SBATCH --output=job_%j.log                         # Output log file (%j will be replaced with job ID)
#SBATCH --error=job_%j.log                          # Error log file
#SBATCH --partition=long                            # Partition to submit to
#SBATCH --nodes=1                                   # Ensure it runs on one node
#SBATCH --ntasks=1                                  # Run a single task
#SBATCH --cpus-per-task=48                          # Request 48 CPUs
#SBATCH --mem=64G                                   # Memory allocation


source .venv/bin/activate

# add current directory to python path
eexport PYTHONPATH=$PYTHONPATH:.

t_speed=$1

uv run abm/info_channels_ea.py \
    --n_agents 10 \
    --target_speed="$t_speed" \
    --episode_len 1000 \
    --pop_size 1000 \
    --ngen 1000 \
    --costs 0.05 0.02 0.01 0.005 \
    --use_wandb \
    --run_name="init_explor"
