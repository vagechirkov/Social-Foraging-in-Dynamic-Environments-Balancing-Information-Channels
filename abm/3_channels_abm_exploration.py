import multiprocessing
import random
import math
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from utils import VmasEvaluator, GenePersistenceTransform, SimpleAgent, ExperimentLogger

# Indices in CHANNEL_NAMES = ["Priv", "Belief", "Heading", "Pos", "None"]
IDX_PRIV = 0
IDX_BELIEF = 1
IDX_NONE = 4

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

def probs_to_logits(p_priv, p_bel, p_none):
    """
    Converts desired probabilities to logits.
    We assign -100 to 0-probability channels (effectively 0 after softmax).
    """
    logits = [-100.0] * 5

    def get_logit(p):
        return math.log(p) if p > 1e-6 else -100.0

    logits[IDX_PRIV] = get_logit(p_priv)
    logits[IDX_BELIEF] = get_logit(p_bel)
    logits[IDX_NONE] = get_logit(p_none)

    return logits

@hydra.main(version_base=None, config_path=".", config_name="3_channels_abm")
def run_exploration(cfg: DictConfig):

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"

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
    for (p_p, p_b, p_n) in simplex_points:
        genes = probs_to_logits(p_p, p_b, p_n)
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
    logger = ExperimentLogger(cfg.use_wandb, cfg, save_fig_locally=False)

    # 4. Evaluate
    print("Starting simulation...")
    with torch.no_grad():
        # full_fitness_tensor shape: [total_pop_padded, n_agents]
        full_fitness_tensor = evaluator.evaluate(env, all_islands)

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
    max_possible_score = cfg.max_steps * 1.75 if cfg.targets_quality == "HT" else cfg.max_steps
    for config, scores in results_map.items():
        avg_score = np.mean(scores) / max_possible_score
        plot_data.append({
            'priv': config[0],
            'bel': config[1],
            'none': config[2],
            'score': avg_score
        })

    logger.log_ternary_plot(plot_data, cfg.resolution)
    logger.finish()

if __name__ == "__main__":
    run_exploration()