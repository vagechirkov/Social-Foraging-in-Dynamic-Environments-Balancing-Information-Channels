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
echo "Current working directory: $(pwd)"

# add current directory to python path
export PYTHONPATH=$PYTHONPATH:.

source ~/.bashrc
source .venv/bin/activate

# Check which python and uv are being used
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "UV path: $(which uv)"
echo "Checking GPU availability..."
nvidia-smi

t_speed=$1

uv run abm/info_channels_ea.py \
    --n_agents 500 \
    --target_speed "$t_speed" \
    --episode_len 3000 \
    --pop_size 50 \
    --ngen 1000 \
    --top_k 5 \
    --dim 3 \
    --costs 0.05 0.02 0.01 0.005 \
    --use_wandb \
    --use_gpu \
    --run_name "init_explor" \
    --base_noise 0.01
