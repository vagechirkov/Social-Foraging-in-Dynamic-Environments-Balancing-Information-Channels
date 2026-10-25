import os
import argparse
import glob
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import wandb
from scipy.special import softmax

from dotenv import load_dotenv

# 1. Load environment variables from .env if present
load_dotenv()

PROJECT = "chirkov/ssga_2_targets"

TRAP_CLASSES = ["good", "bad", "lost"]  # status codes 0, 1, 2 in ssga_2_targets.py
N_WORKERS = 8


def get_runs(api, max_ticks, eval_interval):
    filters = {
        "config.max_ticks": max_ticks,
        "config.ssga.eval_interval": eval_interval,
        "config.env_switch": False,
    }
    return list(api.runs(PROJECT, filters=filters))


def aggregate_agent_stats(artifact, tmp_root):
    """Download one agent_stats table and reduce it to one row per group."""
    download_dir = tempfile.mkdtemp(dir=tmp_root)
    try:
        # skip_cache: the raw tables are ~2.6 GB per run, don't keep them in ~/.cache
        path = artifact.download(root=download_dir, skip_cache=True)
        json_path = glob.glob(os.path.join(path, "**", "*.table.json"), recursive=True)[0]
        with open(json_path, "r") as f:
            data = json.load(f)
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)

    df = pd.DataFrame(data["data"], columns=data["columns"])
    probs = softmax(df[["gene_priv", "gene_soc", "gene_none"]].values, axis=1)
    df["prob_priv"], df["prob_soc"], df["prob_none"] = probs[:, 0], probs[:, 1], probs[:, 2]

    g = df.groupby("group_id")
    out = pd.concat([
        g[["current_fitness", "avg_fitness"]].median().add_prefix("median_"),
        g[["prob_priv", "prob_soc", "prob_none"]].mean(),
    ], axis=1).reset_index()
    out["group_id"] = out["group_id"].astype(int)
    out["generation"] = int(artifact.version[1:])
    return out


def download_occupancy(run, tmp_root):
    """Per group: final trap class. Per group x generation: fraction of the generation's
    trap blocks in which the group was on the good target, and on the bad target."""
    artifacts = [a for a in run.logged_artifacts() if a.name.startswith(f"occupancy_{run.id}")]
    if not artifacts:
        return None
    artifact = sorted(artifacts, key=lambda x: int(x.version[1:]))[-1]
    download_dir = tempfile.mkdtemp(dir=tmp_root)
    try:
        path = artifact.download(root=download_dir, skip_cache=True)
        npz_path = glob.glob(os.path.join(path, "*.npz"))[0]
        d = np.load(npz_path)
        final_class, status = d["final_class"], d["status"]
        blocks_per_gen = int(d["eval_interval"]) // int(d["trap_block"])
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)

    n_blocks, n_groups = status.shape
    per_gen_shape = (n_blocks // blocks_per_gen, blocks_per_gen, n_groups)
    on_good = (status == 0).reshape(per_gen_shape).mean(1)  # [Gen, Groups]
    on_bad = (status == 1).reshape(per_gen_shape).mean(1)
    final = pd.DataFrame({
        "group_id": np.arange(n_groups),
        "final_class": pd.Categorical.from_codes(final_class, TRAP_CLASSES),
    })
    per_gen = pd.DataFrame({
        "generation": np.repeat(np.arange(on_good.shape[0]), n_groups),
        "group_id": np.tile(np.arange(n_groups), on_good.shape[0]),
        "frac_on_good": on_good.flatten().astype(np.float32),
        "frac_on_bad": on_bad.flatten().astype(np.float32),
    })
    return final, per_gen


def process_run(run, per_run_dir, tmp_root):
    out_file = os.path.join(per_run_dir, f"{run.id}.parquet")
    cached = pd.read_parquet(out_file) if os.path.exists(out_file) else None
    occupancy_cols = ["frac_on_good", "frac_on_bad"]
    if cached is not None and all(c in cached for c in occupancy_cols):
        print(f"  Already processed, loading {out_file}")
        return cached

    occupancy = download_occupancy(run, tmp_root)
    if occupancy is None:
        print(f"  No occupancy artifact for {run.id} (state: {run.state}), skipping.")
        return None
    final, per_gen = occupancy

    if cached is not None:
        # Older cache without the on-target time series: add it without re-downloading the tables
        print(f"  Adding {occupancy_cols} to {out_file}")
        df = cached.drop(columns=occupancy_cols, errors="ignore").merge(per_gen, on=["generation", "group_id"], how="left")
        df.to_parquet(out_file, index=False)
        return df

    artifacts = [a for a in run.logged_artifacts() if "agent_stats" in a.name]
    artifacts = sorted(artifacts, key=lambda x: int(x.version[1:]))
    n_expected = run.config["max_ticks"] // run.config["ssga"]["eval_interval"]
    print(f"  Found {len(artifacts)} agent_stats tables (expected {n_expected}).")

    with ThreadPoolExecutor(N_WORKERS) as pool:
        dfs = []
        for i, df in enumerate(pool.map(lambda a: aggregate_agent_stats(a, tmp_root), artifacts)):
            dfs.append(df)
            if i > 0 and i % 50 == 0:
                print(f"  Processed {i} / {len(artifacts)} tables for {run.id}...")

    df = pd.concat(dfs, ignore_index=True).merge(final, on="group_id")
    df = df.merge(per_gen, on=["generation", "group_id"], how="left")
    # Table for generation g is logged at the last tick of that generation
    df["step"] = (df["generation"] + 1) * run.config["ssga"]["eval_interval"] - 1
    df.to_parquet(out_file, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Download per-group stats split by final trap state.")
    # Defaults: the long 1000-generation runs
    parser.add_argument("--max-ticks", type=int, default=1000 * 1000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--out-subfolder", default=None,
                        help="Defaults to trap_split_T<eval_interval>_long for the long runs, "
                             "trap_split_T<eval_interval>_<n_gen>gen otherwise.")
    args = parser.parse_args()

    n_gen = args.max_ticks // args.eval_interval
    out_subfolder = args.out_subfolder or (
        f"trap_split_T{args.eval_interval}_long" if args.max_ticks == 1000 * 1000
        else f"trap_split_T{args.eval_interval}_{n_gen}gen")

    api = wandb.Api()
    runs = get_runs(api, args.max_ticks, args.eval_interval)
    print(f"Found {len(runs)} runs matching the filters.")

    wandb_dir = os.environ.get("WANDB_DIR", ".")
    out_dir = os.path.join(wandb_dir, "data", out_subfolder)
    per_run_dir = os.path.join(out_dir, "per_run")
    tmp_root = os.path.join(wandb_dir, "tmp_downloads")
    os.makedirs(per_run_dir, exist_ok=True)
    os.makedirs(tmp_root, exist_ok=True)

    all_dfs = []
    for run in runs:
        meta = {
            "run_id": run.id,
            "social_fusion": run.config.get("social_fusion", "naive"),
            "close_target_index": run.config["close_target_index"],
        }
        print(f"Processing run: {run.id} {meta} ({run.state})")
        df = process_run(run, per_run_dir, tmp_root)
        if df is not None:
            all_dfs.append(df.assign(**meta))

    if not all_dfs:
        print("No runs processed.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    for col in ["run_id", "social_fusion"]:
        final_df[col] = final_df[col].astype("category")

    out_file = os.path.join(out_dir, "merged_group_stats.parquet")
    print(f"Saving merged dataframe {final_df.shape} to {out_file}...")
    final_df.to_parquet(out_file, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
