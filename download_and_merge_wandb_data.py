import os
import json
import wandb
import pandas as pd

from dotenv import load_dotenv

# 1. Load environment variables from .env if present
load_dotenv()

def main():
    api = wandb.Api()
    runs = ["chirkov/ssga_2_targets/v9arred3", "chirkov/ssga_2_targets/jmrbbem4"]

    all_dfs = []
    all_group_dist_dfs = []

    for run_path in runs:
        print(f"Processing run: {run_path}")
        run = api.run(run_path)
        run_id = run.id
        close_target_index = run.config.get("close_target_index", None)
        
        # Get history to align steps for agent_stats
        print(f"Fetching history for {run_id}...")
        hist = run.history(keys=["_step", "ssga/agent_stats"], samples=100000)
        hist = hist.dropna(subset=["ssga/agent_stats"])
        hist = hist.sort_values(by="_step").reset_index(drop=True)
        
        # Get artifacts for agent_stats
        print(f"Fetching agent_stats artifacts for {run_id}...")
        artifacts = [a for a in run.logged_artifacts() if "ssgaagent_stats" in a.name or "agent_stats" in a.name]
        artifacts = sorted(artifacts, key=lambda x: int(x.version[1:]))
        
        print(f"Found {len(artifacts)} agent_stats table artifacts and {len(hist)} history steps.")
        
        for i, artifact in enumerate(artifacts):
            step = hist.loc[i, "_step"] if i < len(hist) else -1
            
            # Download the JSON file
            wandb_dir = os.environ.get("WANDB_DIR", "./")
            download_dir = os.path.join(wandb_dir, "artifacts", artifact.name)
            path = artifact.download(root=download_dir)
            json_path = os.path.join(path, "ssga", "agent_stats.table.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(path, "agent_stats.table.json")
            
            with open(json_path, "r") as f:
                data = json.load(f)
                
            df = pd.DataFrame(data["data"], columns=data["columns"])
            df["run_id"] = run_id
            df["close_target_index"] = close_target_index
            df["step"] = step
            
            all_dfs.append(df)
            
            if i > 0 and i % 50 == 0:
                print(f"Processed {i} / {len(artifacts)} agent_stats tables for {run_id}...")
                

        # Get artifacts for group_distances
        print(f"Fetching group_distances artifacts for {run_id}...")
        group_dist_artifacts = [a for a in run.logged_artifacts() if a.name.startswith(f"group_distances_{run_id}_gen_")]
        
        print(f"Found {len(group_dist_artifacts)} group_distances artifacts.")
        
        for i, artifact in enumerate(group_dist_artifacts):
            wandb_dir_path = os.environ.get("WANDB_DIR", "./")
            download_dir = os.path.join(wandb_dir_path, "artifacts", artifact.name)
            path = artifact.download(root=download_dir)
            
            parquet_files = [f for f in os.listdir(path) if f.endswith(".parquet")]
            if parquet_files:
                parquet_path = os.path.join(path, parquet_files[0])
                df = pd.read_parquet(parquet_path)
                df["run_id"] = run_id
                df["close_target_index"] = close_target_index
                all_group_dist_dfs.append(df)
            
            if i > 0 and i % 50 == 0:
                print(f"Processed {i} / {len(group_dist_artifacts)} group_distances for {run_id}...")

    print("Concatenating all agent_stats data...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    

    print("Concatenating all group_distances data...")
    if all_group_dist_dfs:
        final_group_dist_df = pd.concat(all_group_dist_dfs, ignore_index=True)
    else:
        final_group_dist_df = pd.DataFrame()

    # Subfolder with run IDs
    run_ids = [r.split("/")[-1] for r in runs]
    subfolder_name = "_".join(run_ids)
    wandb_dir = os.environ.get("WANDB_DIR", ".")
    out_dir = os.path.join(wandb_dir, "data", subfolder_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save to CSV and Parquet for agent_stats
    out_file_csv = os.path.join(out_dir, "merged_agent_stats.csv.gz")
    out_file_parquet = os.path.join(out_dir, "merged_agent_stats.parquet")
    
    print(f"Saving agent_stats dataframe to {out_file_csv}...")
    final_df.to_csv(out_file_csv, index=False, compression="gzip")
    
    print(f"Saving agent_stats dataframe to {out_file_parquet}...")
    final_df.to_parquet(out_file_parquet, index=False)
    
    print(f"Done! Merged agent_stats shape: {final_df.shape}")


    # Save to CSV and Parquet for group_distances
    if not final_group_dist_df.empty:
        group_dist_file_csv = os.path.join(out_dir, "merged_group_distances.csv.gz")
        group_dist_file_parquet = os.path.join(out_dir, "merged_group_distances.parquet")
        
        print(f"Saving group distances dataframe to {group_dist_file_csv}...")
        final_group_dist_df.to_csv(group_dist_file_csv, index=False, compression="gzip")
        
        print(f"Saving group distances dataframe to {group_dist_file_parquet}...")
        final_group_dist_df.to_parquet(group_dist_file_parquet, index=False)
        
        print(f"Done! Merged group distances shape: {final_group_dist_df.shape}")

if __name__ == "__main__":
    main()
