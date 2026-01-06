#!/bin/bash

#SBATCH --job-name=3-channel-exploration            # Job name
#SBATCH --output=job_%j.log                         # Output log file (%j will be replaced with job ID)
#SBATCH --error=job_%j.log                          # Error log file
#SBATCH --partition=oneday                          # Partition to submit to
#SBATCH --nodes=1                                   # Ensure it runs on one node
#SBATCH --ntasks=1                                  # Run a single task
#SBATCH --cpus-per-task=64                          # Request 64 CPUs
#SBATCH --mem=128G                                  # Memory allocation


source .venv/bin/activate

# add current directory to python path
export PYTHONPATH=$PYTHONPATH:.

t_speed=$1
gamma_bel=$2
beta_bel=$3

python abm/3_channels_abm_exploration.py \
    --m n_agents=10 \
    max_steps=1000 \
    replicates=100 \
    run_name="01_2026" \
    target_speed="$t_speed" \
    social_trans_scale="$gamma_bel" \
    dist_noise_scale_soc="$beta_bel"
