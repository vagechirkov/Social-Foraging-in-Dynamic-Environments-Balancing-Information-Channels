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



## Run EA on CPU cluster
```bash
sbatch ea_cpu_driver_itb.sh
```

## Run 3 channels parameter sweep
```bash
uv run python submit_jobs_3_channel.py
```

```bash
uv run python3 3_channel_param_exploration
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