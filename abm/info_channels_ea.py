import argparse
import random
import time
from typing import Any, List

import numpy as np
import torch
import wandb
from deap import base, creator, tools

from abm.utils import ExperimentLogger, GenePersistenceTransform, VmasEvaluator

# Global DEAP Setup
# Must be defined at module level for multiprocessing pickling compatibility
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class GeneticAlgorithm:
    """Handles all Evolutionary Computation logic (DEAP)."""

    def __init__(self, n_channels: int, sigma_init: float = 0.2, mutation_prob: float = 0.2):
        self.n_channels = n_channels
        self.sigma_init = sigma_init
        self.mutation_prob = mutation_prob
        self.toolbox = base.Toolbox()
        self._setup_toolbox()

    def _setup_toolbox(self):
        """Registers DEAP genetic operators."""
        self.toolbox.register("attr_float", random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_float, n=self.n_channels)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        # Mutation sigma is dynamic, so we register the base function here
        # but call it with specific arguments in the loop if needed,
        # or re-register it dynamically.

    def create_population(self, pop_size: int, n_agents: int) -> List[List[Any]]:
        """Creates a list of islands (populations)."""
        return [self.toolbox.population(n=n_agents) for _ in range(pop_size)]

    def evolve_population(self, islands: List[List[Any]], generation: int) -> List[List[Any]]:
        """Applies Selection, Cloning, and Annealed Mutation to all islands."""
        # Anneal mutation sigma: 0.2 * (0.999^gen)
        current_sigma = self.sigma_init * (0.999 ** generation)

        # Re-register mutation with new sigma
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=current_sigma, indpb=0.2)

        next_gen_islands = []
        for island in islands:
            # 1. Save Elites (e.g., Top 20%)
            n_elites = int(len(island) * 0.2)
            elites = tools.selBest(island, k=int(len(island) * 0.2))
            elites = list(map(self.toolbox.clone, elites)) # Clone to protect from mutation

            # 2. Select Parents for the rest
            offspring = self.toolbox.select(island, len(island) - n_elites)
            offspring = list(map(self.toolbox.clone, offspring))

            # 3. Mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            next_gen_islands.append(elites + offspring)

        return next_gen_islands


def parse_args():
    parser = argparse.ArgumentParser(description="Evolutionary Optimization for VMAS Agents")

    # Experiment
    parser.add_argument("--project_name", type=str, default="info_channels_ea",
                        help="WandB project name")
    parser.add_argument("--run_name", type=str, default='omega_ea', help="WandB run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--use_gpu", action="store_true", help="Enable WandB logging")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="Frequency of heatmap logging (generations)")

    # GA
    parser.add_argument("--ngen", type=int, default=50, help="Generations")
    parser.add_argument("--pop_size", type=int, default=100,
                        help="Total population size (will be divided across workers)")
    parser.add_argument("--top_k", type=int, default=10, help="Top islands to visualize")

    # Env
    parser.add_argument("--dim", type=int, default=5, help="Environment dimension")
    parser.add_argument("--n_agents", type=int, default=10, help="Agents per env")
    parser.add_argument("--episode_len", type=int, default=1000, help="Simulation steps")
    parser.add_argument("--target_speed", type=float, default=0.05, help="Agent speed")
    parser.add_argument("--costs", nargs=4, type=float, default=[1.0, 0.5, 0.25, 0.1],
                        help="Costs: [Priv, Belief, Heading, Pos]")
    parser.add_argument("--dist_noise_scale_priv", type=float, default=2.0, help="Private noise dist scale")
    parser.add_argument("--dist_noise_scale_soc", type=float, default=2.0, help="Social noise dist scale")

    return parser.parse_args()

def run_experiment():
    args = parse_args()

    # Global Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"

    # Initialize Modules
    logger = ExperimentLogger(args.use_wandb, args, save_fig=True)
    evaluator = VmasEvaluator(args, device)
    ga = GeneticAlgorithm(n_channels=5)

    # Calculate total population size
    total_pop_size = evaluator.total_pop
    if args.use_wandb:
        wandb.config.update({"total_pop_size": total_pop_size, "device": device})

    # Initialize Population
    print(f"Creating {total_pop_size} islands with {args.n_agents} agents...")
    islands = ga.create_population(pop_size=total_pop_size, n_agents=args.n_agents)

    # --- Evolution Loop ---
    start_time = time.time()
    env = evaluator.init_env(env_transform=GenePersistenceTransform)
    for gen in range(args.ngen):
        gen_step = gen + 1
        gen_start = time.time()

        # 1. Evaluate
        with torch.no_grad():
            fitness_tensor = evaluator.evaluate(env, islands)
        gen_duration = time.time() - gen_start

        # 2. Log
        steps_per_sec = (total_pop_size * args.episode_len) / gen_duration
        logger.log_metrics_ga(gen_step, fitness_tensor, steps_per_sec, islands)

        if gen % args.log_freq == 0 or gen_step == args.ngen:
            logger.log_heatmap(islands, fitness_tensor, gen_step)
            logger.log_parallel_plot(islands, fitness_tensor, gen_step)

        # 3. Evolve
        if gen_step < args.ngen:
            islands = ga.evolve_population(islands, gen)

    total_time = time.time() - start_time
    print(f"Evolution Complete. Total time: {total_time:.2f}s")
    logger.finish()

if __name__ == "__main__":
    run_experiment()