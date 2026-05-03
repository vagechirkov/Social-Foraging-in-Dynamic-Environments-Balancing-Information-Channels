import subprocess
import itertools
import argparse

def submit_ssga_jobs():
    parser = argparse.ArgumentParser(description="Submit SSGA parameter sweep jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    # Define parameter ranges
    switch_times = [500, 10000, 50000]
    cull_fractions = [0.1, 0.2, 0.3]
    n_agents_list = [30, 10, 60]

    sbatch_script = "ssga_gpu_hpc_scioi.sh"

    print(f"Submitting SSGA parameter sweep...")
    print(f"Switch times: {switch_times}")
    print(f"Cull fractions: {cull_fractions}")
    print(f"N agents: {n_agents_list}")
    print(f"Using submit script: {sbatch_script}")

    combinations = list(itertools.product(n_agents_list, switch_times, cull_fractions))
    print(f"Total jobs: {len(combinations)}")

    for n_agents, switch_time, cull_frac in combinations:
        run_name = f"ssga_st{switch_time}_cf{cull_frac}_nag{n_agents}"
        
        cmd = [
            "sbatch", 
            sbatch_script,
            f"switch_time={switch_time}",
            f"ssga.cull_fraction={cull_frac}",
            f"n_agents={n_agents}",
            f"run_name={run_name}",
            f"use_gpu=True",
            f"max_ticks={100_000_000_000}"
        ]
        
        if args.dry_run:
            print(f"  Command: {' '.join(cmd)}")
        else:
            subprocess.run(cmd)

if __name__ == "__main__":
    submit_ssga_jobs()
