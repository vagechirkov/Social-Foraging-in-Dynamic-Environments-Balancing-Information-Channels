#!/bin/bash

#SBATCH --job-name=ea-collective-tracking           # Job name
#SBATCH --output=job_%j.log                         # Output log file (%j will be replaced with job ID)
#SBATCH --error=job_%j.log                          # Error log file
#SBATCH --partition=long                            # Partition to submit to
#SBATCH --nodes=1                                   # Ensure it runs on one node
#SBATCH --ntasks=1                                  # Run a single task
#SBATCH --cpus-per-task=30                          # Request 60 CPUs
#SBATCH --mem=128G                                   # Memory allocation


source .venv/bin/activate

# add current directory to python path
export PYTHONPATH=$PYTHONPATH:.

mode=$1
category=$2

echo "Running Evolutionary Pipeline:"
echo "Mode: $mode"
echo "Category: $category"

# WandB fault-tolerance configuration (tolerates up to 24h blackout)
export WANDB_HTTP_TIMEOUT=1200    # 20 minutes per request
export WANDB_INIT_TIMEOUT=86400   # 24 hours init timeout
export WANDB_HTTP_RETRIES=4000    # high number of retries
export WANDB__SERVICE_WAIT=86400  # service wait time

uv run abm/info_channels_ea.py \
    environment.mode="$mode" \
    environment.static_category="$category" \
    project_name="dynamic_evolution_v1" \
    run_name="ea_pop_30" \
    "${@:3}"
