#!/bin/bash

#SBATCH --job-name=env_optimization_v1
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G

source .venv/bin/activate

# add current directory to python path
export PYTHONPATH=$PYTHONPATH:.

# Run the optimization script
# You can override defaults here if needed, e.g. replicates=100
uv run python abm/optimize_environment.py
