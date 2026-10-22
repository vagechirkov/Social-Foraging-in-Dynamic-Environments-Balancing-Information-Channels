import os
import numpy as np
import pandas as pd
import wandb

from dotenv import load_dotenv

# 1. Load environment variables from .env if present
load_dotenv()

PROJECT = "chirkov/ssga_2_targets"
OUT_SUBFOLDER = "occupancy_fusion_T_sweep"

# Run selection
EXCLUDED_MAX_TICKS = 1000 * 1000  # longer 1000-generation runs
EVAL_INTERVALS = [250, 1000, 4000]
CLOSE_TARGET_INDICES = [0, 1]

TRAP_CLASSES = ["good", "bad", "lost"]  # status codes 0, 1, 2 in ssga_2_targets.py
CHANNEL_NAMES = ["priv", "soc", "none"]


def get_runs(api):
    filters = {
        "config.max_ticks": {"$ne": EXCLUDED_MAX_TICKS},
        "config.ssga.eval_interval": {"$in": EVAL_INTERVALS},
        "config.close_target_index": {"$in": CLOSE_TARGET_INDICES},
        "config.env_switch": False,
    }
    runs = api.runs(PROJECT, filters=filters)
    # Re-check client-side in case the server-side filter misses nested keys
    return [
        r for r in runs
        if r.config.get("max_ticks") != EXCLUDED_MAX_TICKS
        and r.config.get("ssga", {}).get("eval_interval") in EVAL_INTERVALS
        and r.config.get("close_target_index") in CLOSE_TARGET_INDICES
        and r.config.get("env_switch") is False
    ]


def download_occupancy(run):
    artifacts = [a for a in run.logged_artifacts() if a.name.startswith(f"occupancy_{run.id}")]
    if not artifacts:
        return None
    artifact = sorted(artifacts, key=lambda x: int(x.version[1:]))[-1]

    wandb_dir = os.environ.get("WANDB_DIR", "./")
    download_dir = os.path.join(wandb_dir, "artifacts", artifact.name)
    path = artifact.download(root=download_dir)
    npz_files = [f for f in os.listdir(path) if f.endswith(".npz")]
    return np.load(os.path.join(path, npz_files[0]))


def main():
    api = wandb.Api()
    runs = get_runs(api)
    print(f"Found {len(runs)} runs matching the filters.")

    run_rows, block_dfs, prob_dfs, final_dfs = [], [], [], []

    for run in runs:
        cfg = run.config
        meta = {
            "run_id": run.id,
            "social_fusion": cfg.get("social_fusion", "naive"),
            "eval_interval": cfg["ssga"]["eval_interval"],
            "close_target_index": cfg["close_target_index"],
        }
        print(f"Processing run: {run.id} {meta}")

        d = download_occupancy(run)
        if d is None:
            print(f"  No occupancy artifact for {run.id} (state: {run.state}), skipping.")
            continue

        n_blocks, n_groups = d["status"].shape
        n_gen = d["probs"].shape[0]

        # Per block x group: occupancy and status
        df = pd.DataFrame({
            "block": np.repeat(np.arange(n_blocks), n_groups).astype(np.int32),
            "tick": np.repeat(d["block_end_tick"], n_groups).astype(np.int32),
            "group_id": np.tile(np.arange(n_groups), n_blocks).astype(np.int16),
            "occ_good": d["occ_good"].flatten().astype(np.float32),
            "occ_bad": d["occ_bad"].flatten().astype(np.float32),
            "status": pd.Categorical.from_codes(d["status"].flatten(), TRAP_CLASSES),
        })
        block_dfs.append(df.assign(**meta))

        # Per generation x group: group-mean channel use
        probs = d["probs"].reshape(n_gen * n_groups, 3)
        df = pd.DataFrame({
            "generation": np.repeat(np.arange(n_gen), n_groups).astype(np.int32),
            "tick": np.repeat(d["gen_end_tick"], n_groups).astype(np.int32),
            "group_id": np.tile(np.arange(n_groups), n_gen).astype(np.int16),
            **{f"prob_{c}": probs[:, i].astype(np.float32) for i, c in enumerate(CHANNEL_NAMES)},
        })
        prob_dfs.append(df.assign(**meta))

        # Per group: final trap class over the last trap_window blocks
        df = pd.DataFrame({
            "group_id": np.arange(n_groups).astype(np.int16),
            "final_class": pd.Categorical.from_codes(d["final_class"], TRAP_CLASSES),
        })
        final_dfs.append(df.assign(**meta))

        run_rows.append({
            **meta,
            "run_name": cfg.get("run_name"),
            "max_ticks": cfg.get("max_ticks"),
            "n_generations": n_gen,
            "n_groups": n_groups,
            "trap_block": int(d["trap_block"]),
            "trap_window": int(d["trap_window"]),
            "on_target_radius": float(d["on_target_radius"]),
            **{k: run.summary.get(k) for k in [
                "trapped_fraction", "trapped_bad_fraction", "trapped_lost_fraction",
                "n_final_good", "n_final_bad", "n_final_lost",
            ]},
        })

    wandb_dir = os.environ.get("WANDB_DIR", ".")
    out_dir = os.path.join(wandb_dir, "data", OUT_SUBFOLDER)
    os.makedirs(out_dir, exist_ok=True)

    tables = {
        "runs": pd.DataFrame(run_rows),
        "merged_occupancy_blocks": pd.concat(block_dfs, ignore_index=True),
        "merged_occupancy_probs": pd.concat(prob_dfs, ignore_index=True),
        "merged_occupancy_final": pd.concat(final_dfs, ignore_index=True),
    }
    for name, df in tables.items():
        for col in ["run_id", "social_fusion"]:
            if col in df:
                df[col] = df[col].astype("category")
        out_file = os.path.join(out_dir, f"{name}.parquet")
        print(f"Saving {name} dataframe {df.shape} to {out_file}...")
        df.to_parquet(out_file, index=False)

    tables["runs"].to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    print("Done!")


if __name__ == "__main__":
    main()
