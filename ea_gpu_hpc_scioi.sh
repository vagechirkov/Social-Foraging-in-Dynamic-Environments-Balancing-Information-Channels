#!/bin/bash

#SBATCH --job-name=ea-collective-tracking           # Job name
#SBATCH --output=job_%j.log                         # Output log file (%j will be replaced with job ID)
#SBATCH --error=job_%j.log                          # Error log file
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --partition=ex_scioi_gpu                    # Partition to submit to
#SBATCH --gres=gpu:1                                # Request 1 GPU
#SBATCH --cpus-per-task=10                          # Number of CPUs
#SBATCH --time=7-00:00:00                           # Maximum runtime (hh:mm:ss)
#SBATCH --mem=16G                                   # Memory allocation

module load nvidia/cuda/12.1

# assuming that current direction is Social-Foraging-in-Dynamic-Environments-Balancing-Information-Channels
# echo "Current working directory: $(pwd)"

# add current directory to python path
export PYTHONPATH=$PYTHONPATH:.

source ~/.bashrc
source .venv/bin/activate

# echo "uv sync --extra cu121..."
# uv sync --extra cu121 --verbose

t_speed=$1
dim=$2

# uv run
python abm/info_channels_ea.py \
    --m n_agents=30 \
    target_speed="$t_speed" \
    x_dim="$dim" \
    y_dim="$dim" \
    max_steps=1000 \
    n_envs=100 \
    ngen=2000 \
    top_k=10 \
    migration_freq=10 \
    n_migrants=3 \
    cost_priv=0.5
