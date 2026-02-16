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

mode=$1
category=$2

echo "Running Evolutionary Pipeline:"
echo "Mode: $mode"
echo "Category: $category"

uv run abm/info_channels_ea.py \
    environment.mode="$mode" \
    environment.static_category="$category" \
    +evolution.replicates=100 \
    +evolution.generations=1000 \
    project_name="dynamic_evolution_v1" \
    run_name="ea_pop_100"
