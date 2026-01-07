#!/bin/bash

#SBATCH --job-name=3-channel-exploration            # Job name
#SBATCH --output=job_%j.log                         # Output log file (%j will be replaced with job ID)
#SBATCH --error=job_%j.log                          # Error log file
#SBATCH --partition=oneday                          # Partition to submit to
#SBATCH --nodes=1                                   # Ensure it runs on one node
#SBATCH --ntasks=1                                  # Run a single task
#SBATCH --cpus-per-task=64                          # Request 64 CPUs
#SBATCH --mem=512G                                  # Memory allocation


source .venv/bin/activate

# add current directory to python path
export PYTHONPATH=$PYTHONPATH:.

t_speed=$1
cost_priv=$2
cost_belief=$3
dim=$4
n_agents=$5
gamma_belief=$6
belief_selectivity_array=$7

python abm/3_channels_abm_exploration.py \
    --m n_agents="$n_agents" \
    max_steps=5000 \
    replicates=100 \
    run_name="costs" \
    target_speed="$t_speed" \
    cost_priv="$cost_priv" \
    cost_belief="$cost_belief" \
    x_dim="$dim" \
    y_dim="$dim" \
    social_trans_scale="$gamma_belief" \
    belief_selectivity_threshold="$belief_selectivity_array" \
    process_noise_scale=0.1
