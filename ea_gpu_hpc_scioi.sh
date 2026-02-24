#!/bin/bash

#SBATCH --job-name=ea-collective-tracking           # Job name
#SBATCH --output=job_%j.log                         # Output log file (%j will be replaced with job ID)
#SBATCH --error=job_%j.log                          # Error log file
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --partition=gpu,ex_scioi_gpu,scioi_gpu,ex_scioi_a100nv # Partition to submit to
#SBATCH --gres=gpu:1                                # Request 1 GPU
#SBATCH --cpus-per-task=10                          # Number of CPUs
#SBATCH --time=3-00:00:00                           # Maximum runtime (hh:mm:ss)
#SBATCH --mem=128G                                   # Memory allocation

module load nvidia/cuda/12.1

# assuming that current direction is Social-Foraging-in-Dynamic-Environments-Balancing-Information-Channels
# echo "Current working directory: $(pwd)"

# add current directory to python path
export PYTHONPATH=$PYTHONPATH:.

source ~/.bashrc
source .venv2/bin/activate

# echo "uv sync --extra cu121..."
# uv sync --extra cu121 --verbose

mode=$1
category=$2

echo "Running Evolutionary Pipeline on GPU:"
echo "Mode: $mode"
echo "Category: $category"

# WandB fault-tolerance configuration (tolerates up to 24h blackout)
export WANDB_HTTP_TIMEOUT=1200    # 20 minutes per request
export WANDB_INIT_TIMEOUT=86400   # 24 hours init timeout
export WANDB_HTTP_RETRIES=4000    # high number of retries
export WANDB__SERVICE_WAIT=86400  # service wait time

python abm/info_channels_ea.py \
    environment.mode="$mode" \
    environment.static_category="$category" \
    project_name="dynamic_evolution_v1" \
    use_gpu=True \
    run_name="ea_gpu" \
    "${@:3}"
