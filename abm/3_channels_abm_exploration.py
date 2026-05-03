import multiprocessing
import random
import math
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from abm.utils import VmasEvaluator, GenePersistenceTransform, SimpleAgent, ExperimentLogger

# Indices in CHANNEL_NAMES = ["Priv", "Belief", "Heading", "Pos", "None", "Consensus"]
IDX_PRIV = 0
IDX_BELIEF = 1
IDX_NONE = 4
IDX_CONSENSUS = 5

def generate_simplex_grid(resolution: int):
    """
    Generates (p_priv, p_bel, p_none) tuples that sum to 1.0.
    """
    configs = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            p_priv = i / resolution
            p_bel = j / resolution
            p_none = 1.0 - p_priv - p_bel

            # Ensure summation errors don't occur
            if abs(p_priv + p_bel + p_none - 1.0) < 1e-5:
                configs.append((p_priv, p_bel, p_none))
    return configs

def probs_to_logits(p_priv, p_y, p_none, channel_y_idx=IDX_BELIEF):
    """
    Converts desired probabilities to logits.
    We assign -100 to 0-probability channels (effectively 0 after softmax).
    """
    logits = [-100.0] * 6 # Updated to 6 channels

    def get_logit(p):
        return math.log(p) if p > 1e-6 else -100.0

    logits[IDX_PRIV] = get_logit(p_priv)
    logits[channel_y_idx] = get_logit(p_y)
    logits[IDX_NONE] = get_logit(p_none)

    return logits


@hydra.main(version_base=None, config_path=".", config_name="3_channels_abm")
def run_exploration(cfg: DictConfig):

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"

    # Determine Channel Y
    channel_y_name = getattr(cfg, "channel_y_name", "Belief")
    print(f"Exploration Mode: Priv vs {channel_y_name} vs None")

    if channel_y_name == "Belief":
        channel_y_idx = IDX_BELIEF
    elif channel_y_name == "Consensus":
        channel_y_idx = IDX_CONSENSUS
    else:
        raise ValueError(f"Unknown channel_y_name: {channel_y_name}")


    # 1. Generate Configurations & Calculate Exact Requirements
    print(f"Generating simplex grid with resolution {cfg.resolution}...")
    simplex_points = generate_simplex_grid(cfg.resolution)
    num_configs = len(simplex_points)

    # Total actual simulations needed
    num_actual_sims = num_configs * cfg.replicates
    print(f"Grid Points: {num_configs} | Replicates: {cfg.replicates} | Total Actual Sims: {num_actual_sims}")

    # 2. Reconciliation: Calculate Padding for CPU Workers
    # VmasEvaluator requires total_pop to be divisible by num_workers on CPU.
    # We must pad our population so we don't lose the tail end of our grid search.
    padding = 0
    if device == "cpu":
        num_workers = multiprocessing.cpu_count()
        remainder = num_actual_sims % num_workers
        if remainder != 0:
            padding = num_workers - remainder
            print(f"CPU Mode: Padding population with {padding} dummy runs to match worker count {num_workers}.")

    total_pop_padded = num_actual_sims + padding

    cfg.n_envs = total_pop_padded

    # 3. Initialize Evaluator (now with correct padded size)
    evaluator = VmasEvaluator(cfg, device)
    env = evaluator.init_env(env_transform=GenePersistenceTransform)

    # 4. Prepare Population
    all_islands = []

    # A. Generate Actual Islands
    for (p_p, p_y, p_n) in simplex_points:
        genes = probs_to_logits(p_p, p_y, p_n, channel_y_idx)
        for _ in range(cfg.replicates):
            agents = [SimpleAgent(genes) for _ in range(cfg.n_agents)]
            all_islands.append(agents)

    # B. Generate Dummy Islands for Padding
    # We just copy the last island's configuration for the dummy slots
    # These results will be discarded later.
    if padding > 0 and len(all_islands) > 0:
        dummy_template = all_islands[-1]
        for _ in range(padding):
            # Deep copy structure to avoid reference issues
            dummy_agents = [SimpleAgent(list(a)) for a in dummy_template]
            all_islands.append(dummy_agents)

    print(f"Running evaluation with Population Size: {len(all_islands)} (Actual: {num_actual_sims} + Padding: {padding})")

    # Initialize Logger
    save_fig = getattr(cfg, "save_fig", False)
    logger = ExperimentLogger(cfg.use_wandb, cfg, save_fig_locally=save_fig)

    # 4. Evaluate
    env_switch = getattr(cfg, "env_switch", False) and cfg.n_targets == 2
    
    def process_fitness(full_fitness_tensor, max_steps_phase):
        # We only care about the first 'num_actual_sims' rows
        fitness_tensor = full_fitness_tensor[:num_actual_sims]
        env_scores = fitness_tensor.mean(dim=1).cpu().numpy()
        results_map = {} # Key: (p_p, p_b, p_n), Value: list of scores

        for idx, score in enumerate(env_scores):
            config_idx = idx // cfg.replicates
            config = simplex_points[config_idx]

            if config not in results_map:
                results_map[config] = []
            results_map[config].append(score)

        # Calculate average fitness for each unique config
        plot_data = []
        for config, scores in results_map.items():
            avg_score = np.mean(scores) / max_steps_phase
            p_p, p_y, p_n = config
            
            plot_data.append({
                'priv': p_p,
                'bel': p_y, # Represents channel_y
                'none': p_n,
                'score': avg_score
            })
        return plot_data

    print("Starting simulation...")

    # Focal condition: closest grid point to (p_priv≈0.4, p_bel≈0.6)
    FOCAL_PRIV, FOCAL_BEL = 0.4, 0.6
    focal_config_idx = min(
        range(len(simplex_points)),
        key=lambda i: abs(simplex_points[i][0] - FOCAL_PRIV) + abs(simplex_points[i][1] - FOCAL_BEL)
    )
    focal_config = simplex_points[focal_config_idx]
    focal_flat_indices = list(range(focal_config_idx * cfg.replicates,
                                    (focal_config_idx + 1) * cfg.replicates))
    focal_label = f"priv{focal_config[0]:.2f}_bel{focal_config[1]:.2f}"
    print(f"Focal condition: priv={focal_config[0]:.2f}, bel={focal_config[1]:.2f}, "
          f"none={focal_config[2]:.2f} | flat env indices {focal_flat_indices[0]}–{focal_flat_indices[-1]}")

    # 4. Evaluate
    env_switch = getattr(cfg, "env_switch", False) and cfg.n_targets == 2
    switch_time = getattr(cfg, "switch_time", 500)
    
    def process_fitness(full_fitness_tensor, max_steps_total):
        # We only care about the first 'num_actual_sims' rows
        fitness_tensor = full_fitness_tensor[:num_actual_sims]
        env_scores = fitness_tensor.mean(dim=1).cpu().numpy()
        results_map = {} # Key: (p_p, p_b, p_n), Value: list of scores

        for idx, score in enumerate(env_scores):
            config_idx = idx // cfg.replicates
            config = simplex_points[config_idx]

            if config not in results_map:
                results_map[config] = []
            results_map[config].append(score)

        # Calculate average fitness for each unique config
        plot_data = []
        for config, scores in results_map.items():
            avg_score = np.mean(scores) / max_steps_total
            p_p, p_y, p_n = config
            
            plot_data.append({
                'priv': p_p,
                'bel': p_y, # Represents channel_y
                'none': p_n,
                'score': avg_score
            })
        return plot_data

    if env_switch and switch_time < cfg.max_steps:
        print(f"Environment Switch enabled. Running single simulation and reconstructing results (Switch Time: {switch_time}).")
        with torch.no_grad():
            full_fitness_tensor, info = evaluator.evaluate(
                env, all_islands, max_steps=cfg.max_steps,
                return_info=True, focal_indices=focal_flat_indices,
                return_rewards_all=True
            )
            
            rewards_all = info["rewards_all"] # [TotalPop, Steps, Agents, 1]
            
            # Split and sum rewards over the time dimension
            # Shape: [TotalPop, Agents]
            fitness_1 = rewards_all[:, :switch_time, :, :].sum(dim=1).squeeze(-1)
            fitness_2 = rewards_all[:, switch_time:, :, :].sum(dim=1).squeeze(-1)
            
            plot_data_before = process_fitness(fitness_1, switch_time)
            plot_data_after = process_fitness(fitness_2, cfg.max_steps - switch_time)
            
            # Calculate Performance Delta
            plot_data_delta = []
            for b, a in zip(plot_data_before, plot_data_after):
                plot_data_delta.append({
                    'priv': b['priv'],
                    'bel': b['bel'],
                    'none': b['none'],
                    'score': a['score'] - b['score']
                })

            # Log focal timeseries
            focal_rewards = info.get("focal_reward_timeseries", [])
            focal_uncertainties = info.get("focal_uncertainty_timeseries", [])
            
            logger.log_focal_timeseries(
                focal_rewards,
                focal_uncertainties,
                phase="full_run", focal_label=focal_label
            )

            # Log focal performance delta
            if focal_rewards and switch_time < len(focal_rewards):
                before_focal = focal_rewards[:switch_time]
                after_focal = focal_rewards[switch_time:]
                if before_focal and after_focal:
                    avg_before = sum(before_focal) / len(before_focal)
                    avg_after = sum(after_focal) / len(after_focal)
                    delta_focal = avg_after - avg_before
                    if logger.use_wandb:
                        import wandb
                        wandb.log({f"focal/{focal_label}/performance_delta": delta_focal})
                    print(f"Focal Performance Delta: {delta_focal:.4f}")
            logger.log_faceted_ternary_plot(plot_data_before, plot_data_after, cfg.resolution, vmin=0.3, vmax=0.7)
            logger.log_ternary_plot(
                plot_data_delta, 
                cfg.resolution, 
                midpoint=0.0, 
                vmin=-0.4, 
                vmax=0.4, 
                key="ternary_delta_landscape"
            )
    else:
        print(f"Starting simulation (Internal Switch: {env_switch}, Switch Time: {switch_time})...")
        with torch.no_grad():
            full_fitness_tensor, info = evaluator.evaluate(
                env, all_islands, max_steps=cfg.max_steps,
                return_info=True, focal_indices=focal_flat_indices
            )
            plot_data = process_fitness(full_fitness_tensor, cfg.max_steps)
            
            # Log full timeseries
            logger.log_focal_timeseries(
                info.get("focal_reward_timeseries", []),
                info.get("focal_uncertainty_timeseries", []),
                phase="full_run", focal_label=focal_label
            )

        logger.log_ternary_plot(plot_data, cfg.resolution, vmin=0.3, vmax=0.8)

    env.close()
    logger.finish()

if __name__ == "__main__":
    run_exploration()