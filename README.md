# Social Foraging in Dynamic Environments: Balancing Information Channels

## Installation
```bash
uv sync

# for gpu installation
uv sync --extra cu121

# for cpu installation
# uv sync --extra cpu

uv tree

# activate the environment
source .venv/bin/activate
```

See more info [here](https://docs.astral.sh/uv/) and [here](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies).



## Running Evolutionary Algorithm

### 1. Configuration
The experiment is configured in `abm/ea_evaluation.yaml`. Key parameters include:
- `evolution.generations`: Number of generations.
- `evolution.replicates`: Number of parallel replicates (managed internally).
- `environment.mode`: "dynamic" (switches environments) or "static" (fixed environment).

### 2. Local Execution (No Slurm)
You can run the full evolutionary algorithm locally. This is suitable if you have a powerful workstation (e.g., many CPU cores or a GPU) and want to avoid the queue.

**Full Experiment (Uses defaults from `abm/ea_evaluation.yaml`):**
```bash
# Run with default settings (dynamic mode)
PYTHONPATH=. uv run abm/info_channels_ea.py
```

**Testing / Debugging (Short run):**
```bash
# Run with overrides for a quick check
PYTHONPATH=. uv run abm/info_channels_ea.py \
    evolution.replicates=2 \
    evolution.generations=10 \
    max_steps=100 \
    project_name="local_test"
```

**Full Static Experiment:**
```bash
# Runs 1000 generations, 100 replicates, static mode (e.g., baseline)
PYTHONPATH=. uv run abm/info_channels_ea.py environment.mode="static" environment.static_category="baseline"
```

### 3. HPC Execution (Slurm)
To run the full suite of experiments (Dynamic + All Static Baselines) on a Slurm cluster:
```bash
sbatch ea_cpu_driver_itb.sh
```
This script submits:
- One job for the **Dynamic** pipeline.
- Separate jobs for each **Static** environment category (`baseline`, `noisy_private`, `fast_target`).

## Run 3 channels parameter sweep
```bash
uv run python submit_jobs_3_channel.py
```

## RUN EA on GPU cluster
```bash
bash ea_gpu_driver_scioi.sh 
```

## Testing
To run all tests:
```bash
uv run pytest
```

## Streamlit Visualization
To run the interactive visualization:
```bash
PYTHONPATH=.
uv run streamlit run abm/app.py
```

## Environment Optimization
To optimize environment parameters (target behavior, sensing noise, etc.) to maximize social benefit:

1. Run the optimization script:
```bash
uv run python abm/optimize_environment.py
```

2. Monitor progress with the Optuna Dashboard:
```bash
uv run optuna-dashboard sqlite:///env_optimization_v1.db
```

3. Run on Slurm Cluster:
```bash
sbatch optimize_environment_hpc.sh
```