import random
import time
from typing import Any, List

import hydra
import numpy as np
import torch
import wandb
from deap import base, creator, tools
from omegaconf import DictConfig

from abm.utils import ExperimentLogger, GenePersistenceTransform, VmasEvaluator

# Global DEAP Setup
# Must be defined at module level for multiprocessing pickling compatibility
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class GeneticAlgorithm:
    """Handles all Evolutionary Computation logic (DEAP)."""

    def __init__(self, n_channels: int, frozen_indices: List[int] = None, sigma_init: float = 0.2, mutation_prob: float = 0.2):
        self.n_channels = n_channels
        self.frozen_indices = set(frozen_indices) if frozen_indices else set()
        self.sigma_init = sigma_init
        self.mutation_prob = mutation_prob
        self.toolbox = base.Toolbox()
        self._setup_toolbox()

    def _create_individual(self):
        """Creates an individual with frozen indices initialized to 0."""
        genotype = []
        for i in range(self.n_channels):
            if i in self.frozen_indices:
                genotype.append(-100)  # very low probability
            else:
                genotype.append(random.uniform(-1, 1))
        return creator.Individual(genotype)

    def _setup_toolbox(self):
        """Registers DEAP genetic operators."""
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        # Mutation sigma is dynamic, so we register the base function here
        # but call it with specific arguments in the loop if needed,
        # or re-register it dynamically.

    def _mutate(self, individual, mu, sigma, indpb):
        """Custom Gaussian mutation that respects frozen indices."""
        for i in range(len(individual)):
            if i in self.frozen_indices:
                continue
            if random.random() < indpb:
                individual[i] += random.gauss(mu, sigma)
        return individual,

    def create_population(self, pop_size: int, n_agents: int) -> List[List[Any]]:
        """Creates a list of islands (populations)."""
        return [self.toolbox.population(n=n_agents) for _ in range(pop_size)]

    def evolve_population(self, islands: List[List[Any]], generation: int) -> List[List[Any]]:
        """Applies Selection, Cloning, and Annealed Mutation to all islands."""
        # Anneal mutation sigma: 0.2 * (0.999^gen)
        current_sigma = self.sigma_init * (0.999 ** generation)

        # Re-register mutation with new sigma
        self.toolbox.register("mutate", self._mutate, mu=0, sigma=current_sigma, indpb=0.2)

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


@hydra.main(version_base=None, config_path=".", config_name="ea_evaluation")
def run_experiment(cfg: DictConfig):
    # Global Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"

    # Define frozen indices here (indices 2 and 3: Heading and Position)
    frozen_indices = [2, 3]

    # Initialize Modules
    evaluator = VmasEvaluator(cfg, device)
    cfg.n_envs = evaluator.total_pop
    logger = ExperimentLogger(cfg.use_wandb, cfg, save_fig_locally=True, frozen_indices=frozen_indices)
    ga = GeneticAlgorithm(n_channels=5, frozen_indices=frozen_indices)

    # Calculate total population size
    total_pop_size = evaluator.total_pop
    if cfg.use_wandb:
        wandb.config.update({"total_pop_size": total_pop_size, "device": device})

    # Initialize Population
    print(f"Creating {total_pop_size} islands with {cfg.n_agents} agents...")
    islands = ga.create_population(pop_size=total_pop_size, n_agents=cfg.n_agents)

    # --- Evolution Loop ---
    start_time = time.time()
    env = evaluator.init_env(env_transform=GenePersistenceTransform)
    for gen in range(cfg.ngen):
        gen_step = gen + 1
        gen_start = time.time()

        # 1. Evaluate
        with torch.no_grad():
            fitness_tensor = evaluator.evaluate(env, islands)
        gen_duration = time.time() - gen_start

        # 2. Log
        steps_per_sec = (total_pop_size * cfg.max_steps * cfg.n_agents) / gen_duration
        logger.log_metrics_ga(gen_step, fitness_tensor, steps_per_sec, islands)

        if gen % cfg.log_freq == 0 or gen_step == cfg.ngen:
            logger.log_heatmap(islands, fitness_tensor, gen_step)
            logger.log_parallel_plot(islands, fitness_tensor, gen_step)
            logger.log_ternary_density(islands, fitness_tensor, gen_step)

        # 3. Evolve
        if gen_step < cfg.ngen:
            islands = ga.evolve_population(islands, gen)

    total_time = time.time() - start_time
    print(f"Evolution Complete. Total time: {total_time:.2f}s")
    logger.finish()

if __name__ == "__main__":
    run_experiment()