#!/bin/bash

# Driver script to submit 3-channel parameter sweeps for specific environment categories

# Common Parameters (Defaults from ea_evaluation.yaml)
dim=2
n_agents=20
n_targets=1
target_persistence=20
relocation_interval=1000
base_noise=0.1
process_noise_scale=0.05
gamma_belief=0.01
belief_selectivity_array="0.25"
chan_y="Belief"
process_noise_scale_het_ratio=0.0
process_noise_scale_het_scale=1.0
cost_consensus=0.0
consensus_selectivity_threshold=3.0
bias_magnitude=0.0

# Function to submit job for a category
submit_category_job() {
    local category_name=$1
    local t_speed=$2
    local cost_priv=$3
    local cost_belief=$4
    local spot_radius=$5
    local dist_noise_scale_priv=$6

    echo "Submitting job for category: $category_name"
    echo "  Target Speed: $t_speed"
    echo "  Cost Private: $cost_priv"
    echo "  Cost Belief: $cost_belief"
    echo "  Spot Radius: $spot_radius"
    echo "  Dist Noise Priv: $dist_noise_scale_priv"

    sbatch 3_channel_cpu_hpc_itb.sh \
        "$t_speed" \
        "$cost_priv" \
        "$cost_belief" \
        "$dim" \
        "$n_agents" \
        "$gamma_belief" \
        "$belief_selectivity_array" \
        "$base_noise" \
        "$dist_noise_scale_priv" \
        "$target_persistence" \
        "$process_noise_scale" \
        "$relocation_interval" \
        "$n_targets" \
        "$process_noise_scale_het_ratio" \
        "$process_noise_scale_het_scale" \
        "$cost_consensus" \
        "$consensus_selectivity_threshold" \
        "$chan_y" \
        "$bias_magnitude" \
        "$spot_radius" \
        "$category_name"
}

# --- Category: Baseline ---
submit_category_job "baseline" 0.5 0.1 0.1 0.5 0.05

# --- Category: Fast Target ---
submit_category_job "fast_target" 0.9 0.5 0.1 0.5 0.25

# --- Category: Noisy Private ---
submit_category_job "noisy_private" 0.5 0.5 0.1 0.5 2.0
