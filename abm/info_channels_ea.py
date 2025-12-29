import argparse
import multiprocessing
import random
import time
import uuid
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from deap import base, creator, tools
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import ParallelEnv, TransformedEnv, VmasEnv
from torchrl.envs.transforms import Transform

from model import Scenario

CHANNEL_NAMES = ["Priv", "Belief", "Heading", "Pos", "None"]
N_CHANNELS = len(CHANNEL_NAMES)

# Global DEAP Setup
# Must be defined at module level for multiprocessing pickling compatibility
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class GenePersistenceTransform(Transform):
    """
    Ensures that 'genes' found in the input tensordict are copied
    to the output tensordict at every step.
    """
    def _step(self, tensordict, next_tensordict):
        if "genes" in tensordict.keys():
            next_tensordict["genes"] = tensordict["genes"]
        return next_tensordict

    def _reset(self, tensordict, tensordict_reset):
        if "genes" in tensordict.keys():
            tensordict_reset["genes"] = tensordict["genes"]
        return tensordict_reset

    def transform_input_spec(self, input_spec):
        return input_spec

    def transform_output_spec(self, output_spec):
        return output_spec


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


class VmasEvaluator:
    """Handles Environment Creation and Fitness Evaluation."""

    def __init__(self, args, device: str):
        self.args = args
        self.device = device
        self.num_workers = self._get_num_workers()
        # Calculate batch sizes based on total population
        if self.num_workers > 1 and self.device == "cpu":
            self.sub_batch_size = args.pop_size // self.num_workers
            self.total_pop = self.sub_batch_size * self.num_workers
        else:
            self.sub_batch_size = args.pop_size
            self.total_pop = args.pop_size

        if self.total_pop != args.pop_size:
            print(f"Warning: Total population adjusted from {args.pop_size} to "
                  f"{self.total_pop} to match worker count.")

        print(f"Initializing Env on {self.device} | Workers: {self.num_workers} | Total Pop: {self.total_pop}")

    @staticmethod
    def _get_num_workers() -> int:
        available_cpus = multiprocessing.cpu_count()
        return max(1, available_cpus - 2)

    def _make_env_fn(self, num_envs: int):
        """Factory for a single VmasEnv batch."""
        return VmasEnv(
            scenario=Scenario(),
            num_envs=num_envs,
            device=self.device,
            continuous_actions=True,
            max_steps=self.args.episode_len,
            x_dim=self.args.dim,
            y_dim=self.args.dim,
            n_agents=self.args.n_agents,
            n_targets=3,
            targets_quality='HT',
            agent_radius=0.05,
            max_speed=self.args.target_speed,
            cost_priv=self.args.costs[0],
            cost_belief=self.args.costs[1],
            cost_heading=self.args.costs[2],
            cost_pos=self.args.costs[3],
            min_dist_between_entities=0.001,
        )

    def _init_env(self):
        """Initializes either a simple VmasEnv or a ParallelEnv."""
        if self.num_workers > 1 and self.device == "cpu":
            base_env = ParallelEnv(
                num_workers=self.num_workers,
                create_env_fn=lambda: self._make_env_fn(self.sub_batch_size)
            )
        else:
            base_env = self._make_env_fn(self.total_pop)

        # 2. Wrap the Main Env (keeps genes alive in the rollout loop)
        return TransformedEnv(base_env, GenePersistenceTransform())

    def evaluate(self, env, islands: List[List[Any]]) -> torch.Tensor:
        # Prepare Genes
        flat_population = [ind for island in islands for ind in island]
        population_genes = torch.tensor(flat_population, dtype=torch.float32, device=self.device)

        total_pop = len(islands)

        # Reshape Genes & Match Environment Batch Size
        # (This logic ensures [Workers, SubBatch] structure is correct)
        if isinstance(env.base_env, ParallelEnv): # Check base_env because of TransformedEnv wrapper
            reshaped_genes = population_genes.view(
                self.num_workers,
                self.sub_batch_size,
                self.args.n_agents,
                N_CHANNELS
            )
            batch_size_shape = [self.num_workers, self.sub_batch_size]
        else:
            reshaped_genes = population_genes.view(
                total_pop,
                self.args.n_agents,
                N_CHANNELS
            )
            batch_size_shape = [total_pop]

        # Create Initial TensorDict
        initial_tensordict = TensorDict(
            {"genes": reshaped_genes},
            batch_size=batch_size_shape,
            device=self.device
        )

        # Policy
        def policy_func(genes_logits):
            dist = torch.distributions.Categorical(logits=genes_logits)
            return dist.sample().unsqueeze(-1).float()

        policy = TensorDictModule(policy_func, in_keys=["genes"], out_keys=[env.action_key])

        # Rollout
        with torch.no_grad():
            env.reset(initial_tensordict)
            rollouts = env.rollout(
                max_steps=self.args.episode_len,
                policy=policy,
                tensordict=initial_tensordict,
                auto_reset=True
            )

        # Compute Rewards
        rewards_all = rollouts["next", "agents", "reward"]

        time_dim_idx = -3 # Assuming [..., Time, Agents, 1]
        total_rewards = rewards_all.sum(dim=time_dim_idx).squeeze(-1)

        # Flatten batch dimensions ([6, 10] -> [60])
        if len(total_rewards.shape) > 2:
            total_rewards = total_rewards.flatten(0, 1)

        # Assign Fitness
        flat_rewards = total_rewards.cpu().numpy().flatten()
        for ind, reward in zip(flat_population, flat_rewards):
            ind.fitness.values = (float(reward),)

        return total_rewards

class ExperimentLogger:
    """Handles visualization and WandB logging."""

    def __init__(self, use_wandb: bool, args, save_fig: bool = False):
        self.use_wandb = use_wandb
        self.args = args
        self.save_fig = save_fig
        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        wandb.login()
        wandb.init(
            project=self.args.project_name,
            name=self.args.run_name + "-" + str(uuid.uuid4())[:8],
            config=vars(self.args)
        )

    def log_metrics(self, gen: int, fitness_tensor: torch.Tensor, steps_per_sec: float, islands: List[List[Any]]):
        # 1. Global Statistics (All Agents)
        global_mean = fitness_tensor.mean().item()
        global_median = fitness_tensor.median().item()
        global_max = fitness_tensor.max().item()

        # 2. Per-Environment Statistics
        # fitness_tensor shape: [POP_SIZE, N_AGENTS]
        env_means = fitness_tensor.mean(dim=1)
        env_medians = fitness_tensor.median(dim=1).values

        # 3. Top-K Environments Statistics
        top_k = self.args.top_k
        top_k_indices = torch.argsort(env_means, descending=True)[:top_k]

        # Get all agent rewards for the top k islands to compute aggregated stats
        top_k_agents_fitness = fitness_tensor[top_k_indices]
        top_k_mean = top_k_agents_fitness.mean().item()
        top_k_median = top_k_agents_fitness.median().item()

        # 4. Channel Probabilities (Global vs Top-K)
        # Flatten all genes to calculate global probabilities
        all_genes = [ind for island in islands for ind in island]
        all_logits = torch.tensor(all_genes, dtype=torch.float32, device='cpu')
        # Softmax over channel dimension to get probabilities
        all_probs = torch.softmax(all_logits, dim=-1)
        global_probs_mean = all_probs.mean(dim=0)

        # Get Top-K genes
        top_k_indices_list = top_k_indices.cpu().numpy()
        top_k_genes = [ind for i in top_k_indices_list for ind in islands[i]]
        top_k_logits = torch.tensor(top_k_genes, dtype=torch.float32, device='cpu')
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        top_k_probs_mean = top_k_probs.mean(dim=0)

        print(f"Gen {gen:4d} | Global Mean: {global_mean:8.4f} | "
              f"Global Max: {global_max:8.4f} | "
              f"Top {top_k} Mean: {top_k_mean:8.4f} | "
              f"Speed: {steps_per_sec:.0f} env-steps/s")

        if self.use_wandb:
            metrics = {
                "gen": gen,

                # Global
                "global_mean": global_mean,
                "global_median": global_median,
                "global_max": global_max,

                # Top K
                f"top_{top_k}_mean": top_k_mean,
                f"top_{top_k}_median": top_k_median,

                # Independent Env Distributions
                "env_means_hist": wandb.Histogram(env_means.cpu().numpy()),
                "env_medians_hist": wandb.Histogram(env_medians.cpu().numpy()),

                # evaluation speed
                "steps_per_sec": steps_per_sec
            }

            # Log Channel Probabilities
            for i, name in enumerate(CHANNEL_NAMES):
                metrics[f"prob_global/{name}"] = global_probs_mean[i].item()
                metrics[f"prob_top_{top_k}/{name}"] = top_k_probs_mean[i].item()

            wandb.log(metrics)

    def log_heatmap(self, islands, fitness_tensor, gen):
        """Generates and logs the strategy heatmap."""
        fig = self._generate_heatmap_fig(islands, fitness_tensor)
        if self.use_wandb:
            wandb.log({"strategy_heatmap": wandb.Image(fig)}, step=gen)
        if self.save_fig:
            plt.savefig(f"heatmap_gen_{gen:04d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def log_parallel_plot(self, islands, fitness_tensor, gen):
        """Generates and logs the parallel coordinate plot for Top K islands."""
        fig = self._generate_parallel_plot_fig(islands, fitness_tensor)
        if self.use_wandb:
            wandb.log({"parallel_plot": wandb.Image(fig)}, step=gen)
        if self.save_fig:
            plt.savefig(f"parallel_plot_{gen:04d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _generate_heatmap_fig(self, islands, fitness_tensor):
        # 1. Select Top K
        island_means = fitness_tensor.mean(dim=1)
        top_k = self.args.top_k
        top_k_indices = torch.argsort(island_means, descending=True)[:top_k].cpu().numpy()
        top_islands = [islands[i] for i in top_k_indices]
        # Extract the means for these islands (to use as labels)
        top_k_means = island_means[top_k_indices]

        # Extract logits -> probs
        all_genes = [ind for island in top_islands for ind in island]
        logits_tensor = torch.tensor(all_genes, dtype=torch.float32, device='cpu')
        logits_tensor = logits_tensor.view(top_k, self.args.n_agents, N_CHANNELS)
        probs_tensor = torch.softmax(logits_tensor, dim=2)

        # 2. Sort Agents
        sort_order_indices = [3, 2, 1, 4, 0] # Reverse priority: Pos, Heading, Belief, None, Priv
        sorted_agents_tensor = probs_tensor.clone()

        for channel_idx in sort_order_indices:
            values = sorted_agents_tensor[:, :, channel_idx]
            values_rounded = torch.round(values * 10) / 10
            sort_idx = torch.argsort(values_rounded, dim=1, descending=True, stable=True)
            idx_expanded = sort_idx.unsqueeze(-1).expand(-1, -1, N_CHANNELS)
            sorted_agents_tensor = torch.gather(sorted_agents_tensor, 1, idx_expanded)

        # 3. Sort Islands (by private infor usage)
        # island_priv_sum = sorted_agents_tensor[:, :, 0].sum(dim=1)
        # island_sort_idx = torch.argsort(island_priv_sum, descending=True)
        # final_tensor = sorted_agents_tensor[island_sort_idx]
        # Sort the means to match the visual order of islands in the heatmap
        # sorted_means = top_k_means[island_sort_idx]

        # 3. Sort Islands (by fitness)
        # We now keep the islands sorted by their fitness (as determined in Step 1)
        # instead of re-sorting them by private channel usage.
        final_tensor = sorted_agents_tensor
        sorted_means = top_k_means

        # 4. Plot
        heatmap_data = final_tensor.view(-1, N_CHANNELS).numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')

        fig.colorbar(im, label="Selection Probability")
        ax.set_xticks(np.arange(N_CHANNELS))
        ax.set_xticklabels(CHANNEL_NAMES)
        ax.set_xlabel("Information Channel")

        # Set Y-axis labels to be the fitness scores
        n_agents = self.args.n_agents
        # Calculate centers for each island block
        y_ticks = np.arange(top_k) * n_agents + (n_agents / 2) - 0.5
        y_labels = [f"{int(mean)}" for mean in sorted_means.cpu().numpy()]

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel("Avg Group Reward")

        ax.set_title(f"Top {top_k} Groups")

        # Overlay lines
        for i in range(heatmap_data.shape[0]):
            ax.axhline(y=i - 0.5, color='black', linewidth=0.2, alpha=0.1)
        for i in range(1, top_k):
            ax.axhline(y=i * self.args.n_agents - 0.5, color='white', linewidth=1.5, alpha=0.8)

        plt.tight_layout()
        return fig

    def _generate_parallel_plot_fig(self, islands, fitness_tensor):
        # 1. Select Top K
        island_means = fitness_tensor.mean(dim=1)
        top_k = self.args.top_k
        top_k_indices = torch.argsort(island_means, descending=True)[:top_k].cpu().numpy()
        top_islands = [islands[i] for i in top_k_indices]

        # Gather genes from Top K islands
        all_genes = [ind for island in top_islands for ind in island]
        final_tensor = torch.tensor(all_genes, dtype=torch.float32, device='cpu')

        # Calculate Probabilities
        final_probs = torch.softmax(final_tensor.view(-1, N_CHANNELS), dim=1).numpy()

        fig, ax = plt.subplots(figsize=(12, 6))

        # Generate colors for each island using a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_islands)))

        for i, island in enumerate(top_islands):
            # Gather genes from this specific island
            island_genes = [ind for ind in island]
            island_tensor = torch.tensor(island_genes, dtype=torch.float32, device='cpu')

            # Calculate Probabilities [N_AGENTS, N_CHANNELS]
            island_probs = torch.softmax(island_tensor.view(-1, N_CHANNELS), dim=1).numpy()

            # Plot lines for this island with a specific color
            # We use a distinct color for the group and add a label for the legend
            ax.plot(CHANNEL_NAMES, island_probs.T, color=colors[i], alpha=0.2, linewidth=1)

            # Add a dummy line with the same color for the legend (to avoid multiple entries)
            ax.plot([], [], color=colors[i], label=f"Rank {i+1}")

        ax.set_title(f"Top {top_k} Groups")
        ax.set_ylabel("Selection Probability")
        ax.set_xlabel("Information Channel")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Group Rank", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        return fig

    def finish(self):
        if self.use_wandb:
            wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Evolutionary Optimization for VMAS Agents")

    # Experiment
    parser.add_argument("--project_name", type=str, default="info_channels_ea",
                        help="WandB project name")
    parser.add_argument("--run_name", type=str, default='omega_ea', help="WandB run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
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

    return parser.parse_args()

def run_experiment():
    args = parse_args()

    # Global Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # Initialize Modules
    logger = ExperimentLogger(args.use_wandb, args, save_fig=True)
    evaluator = VmasEvaluator(args, device)
    ga = GeneticAlgorithm(n_channels=N_CHANNELS)

    # Calculate total population size
    total_pop_size = evaluator.total_pop
    if args.use_wandb:
        wandb.config.update({"total_pop_size": total_pop_size, "device": device})

    # Initialize Population
    print(f"Creating {total_pop_size} islands with {args.n_agents} agents...")
    islands = ga.create_population(pop_size=total_pop_size, n_agents=args.n_agents)

    # --- Evolution Loop ---
    start_time = time.time()
    for gen in range(args.ngen):
        gen_step = gen + 1
        gen_start = time.time()

        # 1. Evaluate
        env = evaluator._init_env()
        fitness_tensor = evaluator.evaluate(env, islands)
        gen_duration = time.time() - gen_start

        # 2. Log
        steps_per_sec = (total_pop_size * args.episode_len) / gen_duration
        logger.log_metrics(gen_step, fitness_tensor, steps_per_sec, islands)

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