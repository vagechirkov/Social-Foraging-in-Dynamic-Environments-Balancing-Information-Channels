# Social Foraging in Dynamic Environments: Balancing Information Channels

## Installation
```bash
uv sync

# for gpu installation
uv sync --extra cu121

# for cpu installation
# uv sync --extra cpu

uv tree
```

See more info [here](https://docs.astral.sh/uv/) and [here](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies).



## Run EA on CPU cluster
```bash
sbatch ea_cpu_driver_itb.sh
```


## RUN EA on GPU cluster
```bash
bash ea_gpu_driver_scioi.sh 
```