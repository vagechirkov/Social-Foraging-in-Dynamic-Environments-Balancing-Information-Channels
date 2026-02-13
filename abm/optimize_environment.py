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
    study_name = "env_optimization_v1_2"
    n_trials = cfg.n_trials

    # Define the objective function
    def objective(trial):
        # 1. Sample Parameters
        # Target Behavior
        target_speed = trial.suggest_float("target_speed", 0.05, 0.9)
        target_persistence = trial.suggest_float("target_persistence", 0.0, 50.0)
        relocation_interval = trial.suggest_int("relocation_interval", 20, 200)
        target_movement_pattern = trial.suggest_categorical("target_movement_pattern", ["crw", "periodically_relocate", "levy"])
        
        # Sensing & Physics
        dist_noise_scale_priv = trial.suggest_float("dist_noise_scale_priv", 0.0, 5.0)
        bias_magnitude = trial.suggest_float("bias_magnitude", 0.0, 2.0)
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        base_noise = trial.suggest_float("base_noise", 0.0, 0.5)
        process_noise_scale = trial.suggest_float("process_noise_scale", 0.0, 0.1)
        
        # Structure
        # n_targets = trial.suggest_int("n_targets", 1, 10)
        # n_agents = trial.suggest_int("n_agents", 5, 30)
        
        # Use config values if not sampled
        n_targets = cfg.n_targets
        n_agents = cfg.n_agents

        # 2. Update Configuration
        trial_cfg = cfg.copy()
        
        # Apply overrides
        trial_cfg.target_speed = target_speed
        trial_cfg.target_persistence = target_persistence
        trial_cfg.relocation_interval = relocation_interval
        trial_cfg.target_movement_pattern = target_movement_pattern
        
        trial_cfg.dist_noise_scale_priv = dist_noise_scale_priv
        trial_cfg.bias_magnitude = bias_magnitude
        trial_cfg.momentum = momentum
        trial_cfg.base_noise = base_noise
        trial_cfg.process_noise_scale = process_noise_scale
        
        # trial_cfg.n_targets = n_targets
        # trial_cfg.n_agents = n_agents
        
        # 3. Setup Experiment
        replicates = cfg.replicates
        num_scenarios = 2 # Social vs Asocial
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
        # Scenario Social (Belief=0.9, Priv=0.1, None=0.0)
        logits_social = probs_to_logits(p_priv=0.1, p_y=0.9, p_none=0.0, channel_y_idx=IDX_BELIEF)
        
        # Scenario Asocial (Belief=0.0, Priv=0.1, None=0.9)
        logits_asocial = probs_to_logits(p_priv=0.1, p_y=0.0, p_none=0.9, channel_y_idx=IDX_BELIEF)
        
        all_islands = []
        
        # Add Social Agents
        for _ in range(replicates):
            agents = [SimpleAgent(logits_social) for _ in range(n_agents)]
            all_islands.append(agents)
            
        # Add Asocial Agents
        for _ in range(replicates):
            agents = [SimpleAgent(logits_asocial) for _ in range(n_agents)]
            all_islands.append(agents)
            
        # Add Padding
        if padding > 0:
            dummy_template = all_islands[-1]
            for _ in range(padding):
                all_islands.append([SimpleAgent(list(a)) for a in dummy_template])
        
        # 5. Run Evaluation
        with torch.no_grad():
            full_fitness_tensor = evaluator.evaluate(env, all_islands)
            
        # 6. Calculate Objective
        # full_fitness_tensor: [total_pop, n_agents]
        # We need mean fitness per environment
        env_scores = full_fitness_tensor.mean(dim=1).cpu().numpy()
        
        scores_social = env_scores[:replicates]
        scores_asocial = env_scores[replicates:replicates*2]
        
        mean_social = float(np.mean(scores_social))
        mean_asocial = float(np.mean(scores_asocial))
        
        # Return raw difference (Social - Asocial)
        diff = mean_social - mean_asocial
        
        # Composite score to optimize for both efficient tracking and difference
        # We weight the difference heavily (1.0) but also reward high absolute social performance (0.1)
        # to avoid "stupid solutions" where everyone fails equally or social is barely better than terrible.
        score = diff + 0.1 * mean_social
        
        # Log components for analysis
        trial.set_user_attr("mean_social", mean_social)
        trial.set_user_attr("mean_asocial", mean_asocial)
        trial.set_user_attr("diff", diff)
        
        print(f"Trial {trial.number}: Social={mean_social:.4f}, Asocial={mean_asocial:.4f}, Diff={diff:.4f}, Score={score:.4f}")
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
