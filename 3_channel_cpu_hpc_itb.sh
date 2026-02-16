#!/bin/bash

#SBATCH --job-name=3-channel-exploration            # Job name
#SBATCH --output=job_%j.log                         # Output log file (%j will be replaced with job ID)
#SBATCH --error=job_%j.log                          # Error log file
#SBATCH --partition=long                            # Partition to submit to
#SBATCH --nodes=1                                   # Ensure it runs on one node
#SBATCH --ntasks=1                                  # Run a single task
#SBATCH --cpus-per-task=20                          # Request 64 CPUs
#SBATCH --mem=125G                                  # Memory allocation


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
base_noise=$8
dist_noise_scale_priv=$9
target_persistence=${10}
process_noise_scale=${11}
relocation_interval=${12}
n_targets=${13}
process_noise_scale_het_ratio=${14}
process_noise_scale_het_scale=${15}
cost_consensus=${16}
consensus_selectivity_threshold=${17}
channel_y_name=${18}
bias_magnitude=${19}
spot_radius=${20}
category_name=${21}

python abm/3_channels_abm_exploration.py \
    --m n_agents="$n_agents" \
    n_targets="$n_targets" \
    max_steps=1000 \
    replicates=100 \
    run_name="$category_name" \
    target_speed="$t_speed" \
    cost_priv="$cost_priv" \
    cost_belief="$cost_belief" \
    x_dim="$dim" \
    y_dim="$dim" \
    social_trans_scale="$gamma_belief" \
    belief_selectivity_threshold="$belief_selectivity_array" \
    process_noise_scale="$process_noise_scale" \
    base_noise="$base_noise" \
    dist_noise_scale_priv="$dist_noise_scale_priv" \
    target_persistence="$target_persistence" \
    target_movement_pattern="crw" \
    relocation_interval="$relocation_interval" \
    process_noise_scale_het_ratio="$process_noise_scale_het_ratio" \
    process_noise_scale_het_scale="$process_noise_scale_het_scale" \
    cost_consensus="$cost_consensus" \
    consensus_selectivity_threshold="$consensus_selectivity_threshold" \
    channel_y_name="$channel_y_name" \
    bias_magnitude="$bias_magnitude" \
    spot_radius="$spot_radius"
