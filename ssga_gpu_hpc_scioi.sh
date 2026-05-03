#!/bin/bash

#SBATCH --job-name=ssga-collective-tracking
#SBATCH --output=ssga_job_%j.log
#SBATCH --error=ssga_job_%j.log
#SBATCH --ntasks=1
#SBATCH --partition=gpu,ex_scioi_gpu,scioi_gpu,ex_scioi_a100nv
#SBATCH --qos=ex_scioi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G

module load cuda/12.8

export PYTHONPATH=$PYTHONPATH:.

source ~/.bashrc
source .venv/bin/activate

# WandB fault-tolerance
export WANDB_HTTP_TIMEOUT=1200
export WANDB_INIT_TIMEOUT=86400
export WANDB_HTTP_RETRIES=4000
export WANDB__SERVICE_WAIT=86400

# Arguments are passed directly to the python script
python ssga_2_targets.py "$@"
