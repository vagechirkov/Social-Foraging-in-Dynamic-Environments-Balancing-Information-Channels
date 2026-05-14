import os
import json
import wandb
import pandas as pd

from dotenv import load_dotenv

# 1. Load environment variables from .env if present
load_dotenv()

def main():
    api = wandb.Api()
    runs = ["chirkov/ssga_2_targets/lck2wafo", "chirkov/ssga_2_targets/dment0yi"]

    all_dfs = []

    for run_path in runs:
        print(f"Processing run: {run_path}")
        run = api.run(run_path)
        run_id = run.id
        close_target_index = run.config.get("close_target_index", None)
        
        # Get history to align steps
        print(f"Fetching history for {run_id}...")
        hist = run.history(keys=["_step", "ssga/agent_stats"], samples=100000)
        hist = hist.dropna(subset=["ssga/agent_stats"])
        hist = hist.sort_values(by="_step").reset_index(drop=True)
        
        # Get artifacts
        print(f"Fetching artifacts for {run_id}...")
        artifacts = [a for a in run.logged_artifacts() if "ssgaagent_stats" in a.name or "agent_stats" in a.name]
        artifacts = sorted(artifacts, key=lambda x: int(x.version[1:]))
        
        print(f"Found {len(artifacts)} table artifacts and {len(hist)} history steps.")
        
        for i, artifact in enumerate(artifacts):
            step = hist.loc[i, "_step"] if i < len(hist) else -1
            
            # Download the JSON file
            wandb_dir = os.environ.get("WANDB_DIR", "./")
            download_dir = os.path.join(wandb_dir, "artifacts", artifact.name)
            path = artifact.download(root=download_dir)
            json_path = os.path.join(path, "ssga", "agent_stats.table.json")
            
            with open(json_path, "r") as f:
                data = json.load(f)
                
            df = pd.DataFrame(data["data"], columns=data["columns"])
            df["run_id"] = run_id
            df["close_target_index"] = close_target_index
            df["step"] = step
            
            all_dfs.append(df)
            
            if i > 0 and i % 50 == 0:
                print(f"Processed {i} / {len(artifacts)} tables for {run_id}...")

    print("Concatenating all data...")
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Subfolder with run IDs
    run_ids = [r.split("/")[-1] for r in runs]
    subfolder_name = "_".join(run_ids)
    wandb_dir = os.environ.get("WANDB_DIR", ".")
    out_dir = os.path.join(wandb_dir, "data", subfolder_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save to CSV and Parquet
    out_file_csv = os.path.join(out_dir, "merged_agent_stats.csv.gz")
    out_file_parquet = os.path.join(out_dir, "merged_agent_stats.parquet")
    
    print(f"Saving dataframe to {out_file_csv}...")
    final_df.to_csv(out_file_csv, index=False, compression="gzip")
    
    print(f"Saving dataframe to {out_file_parquet}...")
    final_df.to_parquet(out_file_parquet, index=False)
    
    print(f"Done! Merged shape: {final_df.shape}")

if __name__ == "__main__":
    main()
