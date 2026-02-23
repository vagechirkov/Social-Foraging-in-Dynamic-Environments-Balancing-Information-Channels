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

**CPU Execution (Default)**
```bash
uv run python submit_ea.py --dry-run
uv run python submit_ea.py
```

**Customizing Evolution Parameters**
- `--replicates`: Number of replicates to run for each setup
- `--generations`: Number of generations for evolution
- `--switch_interval`: Number of steps between environment switches
- `--selection`: Selection method (`individual-global` or `individual-local`)
- `--multi_level_selection`: Enable multi-level selection (`true` or `false`)
- `--mutation_prob`: The probability of mutation during evolution

Example:
```bash
uv run python submit_ea.py --replicates 50 --generations 2000 --switch_interval 250 --selection individual-global --multi_level_selection True --mutation_prob 0.6
```

**GPU Execution**
To dispatch the evolutionary algorithm to the GPU nodes (e.g. `ex_scioi_gpu`), append the `--gpu` flag:
```bash
uv run --extra cu121 python submit_ea.py --gpu --dry-run
uv run --extra cu121 python submit_ea.py --gpu --replicates 2000 --generations 3000 --switch_interval 300 --selection individual-local --multi_level_selection True --mutation_prob 0.1
uv run --extra cu121 python submit_ea.py --gpu --replicates 2000 --generations 3000 --switch_interval 300 --selection individual-global --multi_level_selection False --mutation_prob 0.1

uv run --extra cu121 python submit_ea.py --gpu --replicates 2000 --generations 3000 --switch_interval 100 --selection individual-local --multi_level_selection True --mutation_prob 0.1
uv run --extra cu121 python submit_ea.py --gpu --replicates 2000 --generations 3000 --switch_interval 100 --selection individual-global --multi_level_selection False --mutation_prob 0.1
```

This script submits:
- One job for the **Dynamic** pipeline for each pair of environments.
- Separate jobs for each **Static** environment category (`solitary`, `collective`, `info_constrained`).

To check GPU usage:
```bash
srun --jobid=123456 nvidia-smi
```

## Run 3 channels parameter sweep
```bash
uv run python submit_jobs_3_channel.py
```

## Run 3 channels parameter sweep for each environment category
```bash
uv run python submit_categories.py
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


## Generate videos
```bash
uv run python abm/generate_videos.py --seed 42 --category solitary --p_private 0.8 --p_social 0.0 --p_none 0.2 --max_steps 1000
uv run python abm/generate_videos.py --seed 42 --category collective --p_private 0.4 --p_social 0.6 --p_none 0.0 --max_steps 1000
uv run python abm/generate_videos.py --seed 42 --category info_constrained --p_private 0.1 --p_social 0.1 --p_none 0.8 --max_steps 1000
```


## Clean up repo

```bash
rm job_*
```