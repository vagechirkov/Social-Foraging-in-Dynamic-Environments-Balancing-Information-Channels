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

    def __init__(self, n_channels: int, frozen_indices: List[int] = None, 
                 cxpb: float = 0.5, mutpb: float = 0.2, sigma: float = 0.1):
        self.n_channels = n_channels
        self.frozen_indices = set(frozen_indices) if frozen_indices else set()
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.sigma = sigma
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

    def _apply_frozen_constraints(self, individual):
        """Resets frozen indices to -100."""
        for i in self.frozen_indices:
            if i < len(individual):
                individual[i] = -100
        return individual

    def _setup_toolbox(self):
        """Registers DEAP genetic operators."""
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Standard Operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=self.sigma, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def create_population(self, pop_size: int, n_agents: int) -> List[List[Any]]:
        """Creates a list of islands (populations)."""
        return [self.toolbox.population(n=n_agents) for _ in range(pop_size)]

    def evolve_population(self, islands: List[List[Any]], generation: int) -> List[List[Any]]:
        """Applies Standard GA Pipeline with Elitism to all islands."""
        next_gen_islands = []
        for island in islands:
            # 1. Select offspring (Tournament)
            offspring = self.toolbox.select(island, len(island))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # 2. Identify the Elite (Best parent) BEFORE modification
            # We must clone it to ensure it isn't mutated later
            best_ind = tools.selBest(island, 1)[0]
            elite = self.toolbox.clone(best_ind)

            # 3. Apply Crossover and Mutation to offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # 4. Apply Constraints (Frozen Indices)
            for ind in offspring:
                 self._apply_frozen_constraints(ind)

            # 5. Re-inject Elitism
            # Replace the first offspring with the preserved elite parent
            # This ensures the max fitness of the population never decreases
            offspring[0] = elite
            
            next_gen_islands.append(offspring)

        return next_gen_islands


@hydra.main(version_base=None, config_path=".", config_name="ea_evaluation")
def run_experiment(cfg: DictConfig):
    # Global Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"

    # Define frozen indices here (indices 2: Heading, 3: Position, 5: Consensus)
    frozen_indices = [2, 3, 5]

    replicates = cfg.evolution.replicates
    generations = cfg.evolution.generations
    switch_interval = cfg.evolution.switch_interval
    
    work_cfg = cfg.copy()
    
    # Define the sequence of environments
    env_keys = ["baseline", "high_cost", "noisy_private", "fast_target"]
    
    def apply_env_category(target_cfg, category_name):
        """Applies parameters from a category to the config."""
        if category_name not in cfg.environments.categories:
            print(f"Warning: Category {category_name} not found in environments.categories")
            return
            
        params = cfg.environments.categories[category_name]
        print(f"--> Switching Environment to: {category_name}")
        for k, v in params.items():
            print(f"    {k}: {v}")
            # Update the specific parameter in the config
            # We use OmegaConf.update to ensure type safety if needed, or just set attribute
            if k in target_cfg:
                 target_cfg[k] = v
            else:
                 # If using open_dict, we can add new keys, but simpler to expect keys to exist
                 # params should ideally match root keys
                 pass
        
        # Manually sync some nested/derived params if needed (e.g. noise scales)
        # For now, we assume the YAML keys match the arguments VmasEvaluator/Scenario expect
        
    # Init Logger
    work_cfg.project_name = cfg.project_name # "dynamic_evolution_v1"
    work_cfg.run_name = f"{cfg.environment.mode}_{cfg.run_name}"
    
    # Total Agents = Replicates * N_Agents
    # Each replicate is an "island" in the islands list
    n_agents_per_island = cfg.n_agents
    
    # Update n_envs for VmasEvaluator (Total Population)
    total_agents = replicates * n_agents_per_island
    
    # Important: VmasEvaluator treats n_envs as the NUMBER OF ENVIRONMENTS (islands)
    # The config parameter `n_envs` in `ea_evaluation.yaml` was used as total pop size in previous script
    # Let's align: VmasEvaluator uses cfg.n_envs to determine batch size
    work_cfg.n_envs = replicates 
    
    evaluator = VmasEvaluator(work_cfg, device)
    
    # Re-initialize logger with correct config
    logger = ExperimentLogger(work_cfg.use_wandb, work_cfg, save_fig_locally=False, frozen_indices=frozen_indices)

    ga = GeneticAlgorithm(n_channels=6, frozen_indices=frozen_indices)
    
    if work_cfg.use_wandb:
        wandb.config.update({
            "total_agents": total_agents, 
            "replicates": replicates,
            "mode": cfg.environment.mode,
            "generations": generations
        })

    # Initialize Population
    print(f"Creating {replicates} islands with {n_agents_per_island} agents each...")
    islands = ga.create_population(pop_size=replicates, n_agents=n_agents_per_island)

    # Evolution Loop
    start_time = time.time()
    
    # Initial Environment Setup
    current_env_category = cfg.environment.static_category if cfg.environment.mode == "static" else env_keys[0]
    apply_env_category(work_cfg, current_env_category)

    env = evaluator.init_env(env_transform=GenePersistenceTransform)

    last_time = time.time()
    last_gen_step = 0
    n_env_steps = replicates * work_cfg.max_steps

    for gen in range(generations):
        gen_step = gen + 1
        
        # A. Dynamic Environment Switching
        if cfg.environment.mode == "dynamic":
             # Switch every 'switch_interval'
             # 0-250: Env 0
             # 250-500: Env 1 ...
             
             # Calculate stage
             stage_idx = (gen // switch_interval) % len(env_keys)
             new_category = env_keys[stage_idx]
             
             if new_category != current_env_category:
                 current_env_category = new_category
                 apply_env_category(work_cfg, current_env_category)
                 
                 # Re-create environment with new parameters
                 del env
                 evaluator = VmasEvaluator(work_cfg, device) # Re-init evaluator to pick up new work_cfg
                 env = evaluator.init_env(env_transform=GenePersistenceTransform)
                 print(f"--> Environment re-initialized for {new_category}")
                 
             if work_cfg.use_wandb:
                 wandb.log({"env_category_idx": stage_idx, "env_category": current_env_category}, step=gen)

        
        gen_start = time.time()

        # 1. Evaluate
        with torch.no_grad():
            # fitness_tensor: [replicates, n_agents] (if evaluate returns [n_envs, n_agents])
            # Check evaluate return shape in utils.py
            fitness_tensor = evaluator.evaluate(env, islands)
            
        # Assign fitness back to individuals
        cpu_fitness = fitness_tensor.cpu().numpy()
        for i, island in enumerate(islands):
            for j, individual in enumerate(island):
                 if i < len(cpu_fitness) and j < len(cpu_fitness[i]):
                    individual.fitness.values = (cpu_fitness[i, j],)
            
        gen_duration = time.time() - gen_start
        max_possible_score = work_cfg.max_steps * 1.75 if work_cfg.targets_quality == "HT" else work_cfg.max_steps
        fitness_tensor = fitness_tensor / max_possible_score

        # 2. Log Metrics
        # Scalar Logging (High Frequency)
        if gen_step % cfg.log_freq == 0:
            env_steps_per_sec = ((gen_step - last_gen_step) * n_env_steps) / (time.time() - last_time)
            last_time = time.time()
            last_gen_step = gen_step
            
            # Log full table snapshot at switch intervals or end of run
            is_snapshot = (gen_step % switch_interval == 0) or (gen_step == generations)
            
            logger.log_metrics_ga(gen, fitness_tensor, env_steps_per_sec, islands, log_table=is_snapshot)

        # Plot Logging (Lower Frequency)
        plot_freq = getattr(cfg, "plot_freq", 50)
        if gen_step % plot_freq == 0 or gen_step == generations:
            logger.log_heatmap(islands, fitness_tensor, gen)
            # logger.log_parallel_plot(islands, fitness_tensor, gen)
            logger.log_ternary_density(islands, fitness_tensor, gen)

        # 3. Evolve
        if gen_step < generations:
            islands = ga.evolve_population(islands, gen)

    total_time = time.time() - start_time
    print(f"Evolution Complete. Total time: {total_time:.2f}s")
    logger.finish()

if __name__ == "__main__":
    run_experiment()