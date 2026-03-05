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
                 cxpb: float = 0.5, mutpb: float = 0.2, sigma: float = 0.1,
                 indpb: float = 0.2, tournament_size: int = 3, elitism_count: int = 1,
                 selection: str = "individual-local", multi_level_selection: bool = False):
        self.n_channels = n_channels
        self.frozen_indices = set(frozen_indices) if frozen_indices else set()
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.sigma = sigma
        self.indpb = indpb
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.selection = selection
        self.multi_level_selection = multi_level_selection
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
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=self.sigma, indpb=self.indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

    def create_population(self, pop_size: int, n_agents: int) -> List[List[Any]]:
        """Creates a list of islands (populations)."""
        return [self.toolbox.population(n=n_agents) for _ in range(pop_size)]

    def evolve_population(self, islands: List[List[Any]], generation: int) -> List[List[Any]]:
        """Evolves population either locally tracking islands or globally acting on pooled islands."""
        
        if self.multi_level_selection:
            # Score groups
            island_fitnesses = []
            for island in islands:
                # Average fitness of the island
                fit = sum(ind.fitness.values[0] for ind in island) / len(island)
                island_fitnesses.append(fit)
                
            # Sort islands by fitness descending
            sorted_indices = np.argsort(island_fitnesses)[::-1]
            num_islands = len(islands)
            num_survivors = max(1, num_islands // 2)
            
            surviving_indices = sorted_indices[:num_survivors]
            surviving_islands = [islands[i] for i in surviving_indices]
            
            if self.selection == "individual-global":
                # Path A: The Homogeneous Swarm
                # Dump survivors into a global pool -> Select and mate globally -> Shuffle into new random groups.
                return self._evolve_global(surviving_islands, generation, total_target_islands=num_islands)
            elif self.selection == "individual-local":
                # Path B: The Co-adapted Team
                # Clone the winning groups to replace the culled ones -> Run selection strictly within each island.
                new_islands = []
                # First add surviving islands
                for island in surviving_islands:
                    new_islands.append([self.toolbox.clone(ind) for ind in island])
                # Fill the rest to replace culled ones
                while len(new_islands) < num_islands:
                    idx = (len(new_islands) - num_survivors) % num_survivors
                    new_islands.append([self.toolbox.clone(ind) for ind in surviving_islands[idx]])
                
                return self._evolve_local(new_islands, generation)
            else:
                raise ValueError(f"Unknown selection method: {self.selection}")
                
        else:
            if self.selection == "individual-global":
                return self._evolve_global(islands, generation, total_target_islands=len(islands))
            elif self.selection == "individual-local":
                return self._evolve_local(islands, generation)
            else:
                raise ValueError(f"Unknown selection method: {self.selection}")

    def _evolve_local(self, islands: List[List[Any]], generation: int) -> List[List[Any]]:
        """Applies Standard GA Pipeline with Elitism to all islands independently."""
        next_gen_islands = []
        for island in islands:
            # 1. Select offspring (Tournament)
            offspring = self.toolbox.select(island, len(island))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # 2. Identify the Elite (Best parent) BEFORE modification
            # We must clone it to ensure it isn't mutated later
            best_inds = tools.selBest(island, self.elitism_count)
            elites = [self.toolbox.clone(ind) for ind in best_inds]

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
            for i in range(len(elites)):
                offspring[i] = elites[i]
            
            next_gen_islands.append(offspring)

        return next_gen_islands

    def _evolve_global(self, islands: List[List[Any]], generation: int, total_target_islands: int = None) -> List[List[Any]]:
        """Applies GA Pipeline globally by pooling all source islands, then redistributing."""
        num_source_islands = len(islands)
        if num_source_islands == 0:
            return []
        
        num_target_islands = total_target_islands if total_target_islands is not None else num_source_islands
        
        # We assume all islands have the same capacity
        island_size = len(islands[0])
        total_size = num_target_islands * island_size
        
        # Pool all individuals
        global_pop = []
        for island in islands:
            global_pop.extend(island)
            
        # 1. Select offspring (Tournament) globally
        offspring = self.toolbox.select(global_pop, total_size)
        offspring = list(map(self.toolbox.clone, offspring))
        
        # 2. Identify the Elites globally BEFORE modification
        total_elitism = self.elitism_count * num_target_islands
        best_inds = tools.selBest(global_pop, min(total_elitism, len(global_pop)))
        elites = [self.toolbox.clone(ind) for ind in best_inds]
        
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
        # Replace the first `total_elitism` offspring with the preserved elite parents
        for i in range(len(elites)):
            if i < len(offspring):
                offspring[i] = elites[i]
                
        # Shuffle offspring so elites and families are randomly distributed across islands
        random.shuffle(offspring)
        
        # 6. Redistribute offspring back to islands
        next_gen_islands = []
        for i in range(num_target_islands):
            start_idx = i * island_size
            end_idx = start_idx + island_size
            next_gen_islands.append(offspring[start_idx:end_idx])
            
        return next_gen_islands


@hydra.main(version_base=None, config_path=".", config_name="ea_evaluation")
def run_experiment(cfg: DictConfig):
    # Global Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Global Optimizations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"

    # Define frozen indices here (indices 2: Heading, 3: Position, 5: Consensus)
    frozen_indices = [2, 3, 5]

    replicates = cfg.evolution.replicates
    generations = cfg.evolution.generations
    switch_interval = cfg.evolution.switch_interval

    run_size = 200 if replicates >= 200 else replicates
    num_runs = max(1, replicates // run_size)
    if replicates >= 200:
        replicates = num_runs * run_size
        print(f"Adjusted total replicates to {replicates} to form {num_runs} independent runs of {run_size} islands.")

    work_cfg = cfg.copy()
    
    env_keys = sorted(list(cfg.environments.categories.keys()))
    print(f"Loaded Environment Categories: {env_keys}")
    
    environment_sequence = "default"
    if cfg.environment.mode == "dynamic":
        if cfg.environment.static_category == "random":
            # Generate a random sequence big enough to cover the max number of switches
            num_switches = (generations // switch_interval) + 2
            # Use all available environments for random sequences
            available_envs = list(cfg.environments.categories.keys())
            env_keys = [random.choice(available_envs) for _ in range(num_switches)]
            environment_sequence = "random"
            print(f"Dynamic Mode: Using random environment sequence")
        elif "-" in cfg.environment.static_category:
            seq_keys = cfg.environment.static_category.split("-")
            valid_keys = [k for k in seq_keys if k in cfg.environments.categories]
            if len(valid_keys) == len(seq_keys):
                print(f"Dynamic Mode: Oscillating between sequence {valid_keys}")
                env_keys = valid_keys
                environment_sequence = "-".join(valid_keys)
            else:
                 print(f"Warning: Invalid sequence specified {cfg.environment.static_category}. Fallback to full cycle.")
                 environment_sequence = "-".join(env_keys)
        else:
             environment_sequence = "-".join(env_keys)
    
    def apply_env_category(target_cfg, category_name):
        """Applies parameters from a category to the config."""
        if category_name not in cfg.environments.categories:
            print(f"Warning: Category {category_name} not found in environments.categories")
            return
            
        params = cfg.environments.categories[category_name]
        print(f"--> Switching Environment to: {category_name}")
        for k, v in params.items():
            print(f"    {k}: {v}")
            if k in target_cfg:
                 target_cfg[k] = v
        
    # Init Logger
    work_cfg.project_name = cfg.project_name
    if cfg.environment.mode == "dynamic":
        if cfg.environment.static_category == "random":
            work_cfg.run_name = f"{cfg.environment.mode}_random_{cfg.run_name}"
        elif "-" in cfg.environment.static_category:
            work_cfg.run_name = f"{cfg.environment.mode}_{cfg.environment.static_category}_{cfg.run_name}"
        else:
            work_cfg.run_name = f"{cfg.environment.mode}_{cfg.run_name}"
    else:
         work_cfg.run_name = f"{cfg.environment.mode}_{cfg.run_name}"
    
    n_agents_per_island = cfg.n_agents
    total_agents = replicates * n_agents_per_island
    
    work_cfg.n_envs = replicates 
    
    evaluator = VmasEvaluator(work_cfg, device)
    
    # Re-initialize logger with correct config
    logger = ExperimentLogger(work_cfg.use_wandb, work_cfg, save_fig_locally=False, frozen_indices=frozen_indices)

    ga = GeneticAlgorithm(
        n_channels=6,
        frozen_indices=frozen_indices,
        cxpb=cfg.evolution.crossover_prob,
        mutpb=cfg.evolution.mutation_prob,
        sigma=cfg.evolution.sigma,
        indpb=cfg.evolution.indpb,
        tournament_size=cfg.evolution.tournament_size,
        elitism_count=cfg.evolution.elitism_count,
        selection=cfg.evolution.get("selection", "individual-local"),
        multi_level_selection=cfg.evolution.get("multi_level_selection", False)
    )
    
    if work_cfg.use_wandb:
        wandb.config.update({
            "total_agents": total_agents, 
            "replicates": replicates,
            "num_runs": num_runs,
            "run_size": run_size,
            "mode": cfg.environment.mode,
            "generations": generations,
            "elitism": cfg.evolution.elitism_count,
            "tournament_size": cfg.evolution.tournament_size,
            "mutation_prob": cfg.evolution.mutation_prob,
            "crossover_prob": cfg.evolution.crossover_prob,
            "sigma": cfg.evolution.sigma,
            "indpb": cfg.evolution.indpb,
            "selection": cfg.evolution.get("selection", "individual-local"),
            "multi_level_selection": cfg.evolution.get("multi_level_selection", False),
            "environment_sequence": environment_sequence,
            "actual_random_sequence": "-".join(env_keys) if cfg.environment.mode == "dynamic" and cfg.environment.static_category == "random" else None
        })

    # Initialize Population
    print(f"Creating {num_runs} runs containing {run_size} islands with {n_agents_per_island} agents each...")
    islands = []
    
    python_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()

    # Seed each run independently
    for run_id in range(num_runs):
        run_seed = cfg.seed + run_id * 1000
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        
        islands.extend(ga.create_population(pop_size=run_size, n_agents=n_agents_per_island))

    # Restore deterministic RNG sequence
    random.setstate(python_rng_state)
    np.random.set_state(np_rng_state)
    torch.set_rng_state(torch_rng_state)

    # Evolution Loop
    start_time = time.time()
    
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
            stage_idx = (gen // switch_interval) % len(env_keys)
            new_category = env_keys[stage_idx]
            
            if new_category != current_env_category:
                current_env_category = new_category
                apply_env_category(work_cfg, current_env_category)
                del env
                evaluator = VmasEvaluator(work_cfg, device)
                env = evaluator.init_env(env_transform=GenePersistenceTransform)
                print(f"--> Gen {gen}: Environment re-initialized for {new_category}")
                
            if work_cfg.use_wandb:
                wandb.log({"env_category_idx": stage_idx, "env_category": current_env_category}, step=gen)
        
        gen_start = time.time()

        # 1. Evaluate
        with torch.inference_mode():
            eval_result = evaluator.evaluate(env, islands, return_info=True)
            if isinstance(eval_result, tuple):
                fitness_tensor, extra_metrics = eval_result
            else:
                fitness_tensor = eval_result
                extra_metrics = {}
            
        # Assign fitness back to individuals
        flat_population = [ind for island in islands for ind in island]
        flat_fitness = fitness_tensor.cpu().numpy().flatten()
        for ind, fit in zip(flat_population, flat_fitness):
            ind.fitness.values = (float(fit),)
        gen_duration = time.time() - gen_start
        max_possible_score = work_cfg.max_steps * 1.75 if getattr(work_cfg, "targets_quality", None) == "HT" else work_cfg.max_steps
        fitness_tensor = fitness_tensor / max_possible_score

        # 2. Log Metrics
        if gen_step % cfg.log_freq == 0:
            env_steps_per_sec = ((gen_step - last_gen_step) * n_env_steps) / (time.time() - last_time)
            last_time = time.time()
            last_gen_step = gen_step
            
            is_snapshot = (gen_step % switch_interval == 0) or (gen_step == generations)
            
            # Global Logging
            logger.log_metrics_ga(gen, fitness_tensor, env_steps_per_sec, islands, log_table=is_snapshot, extra_metrics=extra_metrics)
            if num_runs > 1:
                for r in range(num_runs):
                    start_idx = r * run_size
                    end_idx = min((r + 1) * run_size, replicates)
                    if start_idx < end_idx:
                        run_ft = fitness_tensor[start_idx:end_idx]
                        run_is = islands[start_idx:end_idx]
                        logger.log_metrics_ga(gen, run_ft, env_steps_per_sec, run_is, log_table=is_snapshot, prefix=f"run_{r}/")

        # Plot Logging
        plot_freq = getattr(cfg, "plot_freq", 50)
        if gen_step % plot_freq == 0 or gen_step == generations:
            logger.log_heatmap(islands, fitness_tensor, gen)
            logger.log_ternary_density(islands, fitness_tensor, gen)
            if num_runs > 1:
                for r in range(num_runs):
                    start_idx = r * run_size
                    end_idx = min((r + 1) * run_size, replicates)
                    if start_idx < end_idx:
                        run_ft = fitness_tensor[start_idx:end_idx]
                        run_is = islands[start_idx:end_idx]
                        logger.log_heatmap(run_is, run_ft, gen, prefix=f"run_{r}/")
                        logger.log_ternary_density(run_is, run_ft, gen, prefix=f"run_{r}/")

        # 3. Evolve independently
        if gen_step < generations:
            if num_runs > 1:
                next_islands = []
                for r in range(num_runs):
                    start_idx = r * run_size
                    end_idx = min((r + 1) * run_size, replicates)
                    if start_idx < end_idx:
                        chunk = islands[start_idx:end_idx]
                        next_chunk = ga.evolve_population(chunk, gen)
                        next_islands.extend(next_chunk)
                islands = next_islands
            else:
                islands = ga.evolve_population(islands, gen)

    total_time = time.time() - start_time
    print(f"Evolution Complete. Total time: {total_time:.2f}s")
    logger.finish()

if __name__ == "__main__":
    run_experiment()