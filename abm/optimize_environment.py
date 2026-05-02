import multiprocessing
import random
import math
import numpy as np
import torch
import hydra
import optuna
from omegaconf import DictConfig, OmegaConf
import joblib
import os
import sys

# Add project root to path to ensure abm module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abm.utils import VmasEvaluator, GenePersistenceTransform, SimpleAgent, ExperimentLogger

# Indices in CHANNEL_NAMES = ["Priv", "Belief", "Heading", "Pos", "None", "Consensus"]
IDX_PRIV = 0
IDX_BELIEF = 1
IDX_NONE = 4

def probs_to_logits(p_priv, p_y, p_none, channel_y_idx=IDX_BELIEF):
    """
    Converts desired probabilities to logits.
    """
    logits = [-100.0] * 6 # 6 channels
    
    def get_logit(p):
        return math.log(p) if p > 1e-6 else -100.0

    logits[IDX_PRIV] = get_logit(p_priv)
    logits[channel_y_idx] = get_logit(p_y)
    logits[IDX_NONE] = get_logit(p_none)

    return logits

@hydra.main(version_base=None, config_path=".", config_name="optimize_environment")
def run_optimization(cfg: DictConfig):
    
    # Study configuration
    storage_name = "sqlite:///env_optimization_v1.db"
    study_name = "env_optimization_v1_5_scenarios_priv_spotlight"
    n_trials = cfg.n_trials

    # Define the objective function
    def objective(trial):
        # 1. Sample Parameters of Interest
        # Target Behavior
        target_speed = trial.suggest_float("target_speed", 0.1, 2.0)
        relocation_interval = trial.suggest_int("relocation_interval", 100, cfg.max_steps)
        
        # Sensing & Physics
        dist_noise_scale_priv = trial.suggest_float("dist_noise_scale_priv", 0.0, 2.0)
        belief_selectivity_threshold = trial.suggest_float("belief_selectivity_threshold", 0.05, 2.0)
        spot_radius = trial.suggest_float("spot_radius", 0.1, 1.0)
        
        # Population Size
        n_agents = trial.suggest_int("n_agents", 5, 30)

        # 2. Update Configuration
        trial_cfg = cfg.copy()
        
        # Apply overrides
        trial_cfg.target_speed = target_speed
        trial_cfg.relocation_interval = relocation_interval
        trial_cfg.belief_selectivity_threshold = belief_selectivity_threshold
        
        trial_cfg.dist_noise_scale_priv = dist_noise_scale_priv
        trial_cfg.spot_radius = spot_radius
        
        trial_cfg.n_agents = n_agents
        
        # 3. Setup Experiment
        replicates = cfg.replicates
        
        # Define Scenarios
        # 1. Social (Target): Priv=0.1, Belief=0.9, None=0.0
        logits_social = probs_to_logits(p_priv=0.1, p_y=0.9, p_none=0.0, channel_y_idx=IDX_BELIEF)
        
        # 2. Asocial/None (Baseline 1): Priv=0.1, Belief=0.0, None=0.9
        logits_asocial = probs_to_logits(p_priv=0.1, p_y=0.0, p_none=0.9, channel_y_idx=IDX_BELIEF)
        
        # 3. High Private (Baseline 2): Priv=0.9, Belief=0.1, None=0.0
        logits_high_priv = probs_to_logits(p_priv=0.9, p_y=0.1, p_none=0.0, channel_y_idx=IDX_BELIEF)
        
        # 4. Average (Baseline 3): Priv=0.33, Belief=0.33, None=0.33
        logits_average = probs_to_logits(p_priv=0.33, p_y=0.33, p_none=0.33, channel_y_idx=IDX_BELIEF)
        
        # 5. Social + None (Baseline 4): Priv=0.1, Belief=0.45, None=0.45
        logits_social_none = probs_to_logits(p_priv=0.1, p_y=0.45, p_none=0.45, channel_y_idx=IDX_BELIEF)
        
        scenarios = [logits_social, logits_asocial, logits_high_priv, logits_average, logits_social_none]
        num_scenarios = len(scenarios)
        num_actual_sims = replicates * num_scenarios
        
        # Padding for CPU workers
        device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"
        padding = 0
        if device == "cpu":
            num_workers = multiprocessing.cpu_count()
            remainder = num_actual_sims % num_workers
            if remainder != 0:
                padding = num_workers - remainder
        
        total_pop_padded = num_actual_sims + padding
        trial_cfg.n_envs = total_pop_padded
        
        # Initialize Evaluator
        evaluator = VmasEvaluator(trial_cfg, device)
        env = evaluator.init_env(env_transform=GenePersistenceTransform)
        
        # 4. Construct Populations
        all_islands = []
        
        for logits in scenarios:
            for _ in range(replicates):
                agents = [SimpleAgent(logits) for _ in range(n_agents)]
                all_islands.append(agents)
            
        # Add Padding
        if padding > 0:
            dummy_template = all_islands[-1]
            for _ in range(padding):
                all_islands.append([SimpleAgent(list(a)) for a in dummy_template])
        
        # 5. Run Evaluation
        with torch.no_grad():
            full_fitness_tensor = evaluator.evaluate(env, all_islands)
            
        env.close()
            
        # 6. Calculate Objective
        # full_fitness_tensor: [total_pop, n_agents]
        # We need mean fitness per environment
        env_scores = full_fitness_tensor.mean(dim=1).cpu().numpy()
        
        # Extract scores for each scenario
        # Order matches 'scenarios' list
        # 0: Social, 1: Asocial, 2: HighPriv, 3: Average, 4: SocNone
        
        mean_scores = []
        start_idx = 0
        for i in range(num_scenarios):
            end_idx = start_idx + replicates
            segment_scores = env_scores[start_idx:end_idx]
            mean_scores.append(float(np.mean(segment_scores)))
            start_idx = end_idx
            
        mean_social = mean_scores[0]
        mean_asocial = mean_scores[1]
        mean_high_priv = mean_scores[2]
        mean_average = mean_scores[3]
        mean_social_none = mean_scores[4]
        
        # Calculate Baseline Max (Best of the non-target strategies)
        baseline_max = max(mean_asocial, mean_high_priv, mean_average, mean_social_none)
        
        # Objective: Margin + Bonus for high absolute performance
        margin = mean_social - baseline_max
        score = margin + 0.1 * mean_social
        
        # Log components for analysis
        trial.set_user_attr("mean_social", mean_social)
        trial.set_user_attr("mean_asocial", mean_asocial)
        trial.set_user_attr("mean_high_priv", mean_high_priv)
        trial.set_user_attr("mean_average", mean_average)
        trial.set_user_attr("mean_social_none", mean_social_none)
        trial.set_user_attr("margin", margin)
        
        print(f"Trial {trial.number}: Soc={mean_social:.2f}, Asoc={mean_asocial:.2f}, HPriv={mean_high_priv:.2f}, Avg={mean_average:.2f}, SocNone={mean_social_none:.2f}, Margin={margin:.2f}, Score={score:.4f}")
        return score

    # Create Study - Using TPE Sampler explicitly
    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    study = optuna.create_study(
        study_name=study_name, 
        direction="maximize", 
        storage=storage_name, 
        load_if_exists=True,
        sampler=sampler
    )
    
    print(f"To monitor the optimization, run:")
    print(f"  uv run optuna-dashboard {storage_name}")
    print(f"Starting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    
    print("Best params:")
    print(study.best_params)
    print("Best value:", study.best_value)
    
    # Save best params to yaml
    best_params_path = "best_env_params.yaml"
    with open(best_params_path, 'w') as f:
        OmegaConf.save(DictConfig(study.best_params), f)
    print(f"Saved best params to {best_params_path}")

if __name__ == "__main__":
    run_optimization()
