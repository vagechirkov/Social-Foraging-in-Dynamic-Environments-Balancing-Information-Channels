import multiprocessing
import uuid
from typing import Any, List, Dict

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import ParallelEnv, TransformedEnv, VmasEnv
from torchrl.envs.transforms import Transform

from abm.model import Scenario

CHANNEL_NAMES = ["Priv", "Belief", "Heading", "Pos", "None"]
N_CHANNELS = len(CHANNEL_NAMES)


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


class SimpleAgent(list):
    """
    A simple wrapper that mimics the structure of a DEAP individual
    (list of genes + fitness attribute) for use in non-evolutionary scripts.
    """
    def __init__(self, genes):
        super().__init__(genes)
        class Fitness:
            def __init__(self):
                self.values = (0.0,)
        self.fitness = Fitness()


class VmasEvaluator:
    """Handles Environment Creation and Fitness Evaluation."""

    def __init__(self, cfg, device: str):
        self.cfg = cfg
        self.device = device
        self.num_workers = self.get_num_workers()

        # Handle DictConfig (Hydra) vs Namespace (argparse) access for pop_size
        n_envs = cfg.n_envs

        # Calculate batch sizes based on total population
        if self.num_workers > 1 and self.device == "cpu":
            self.sub_batch_size = n_envs // self.num_workers
            self.total_pop = self.sub_batch_size * self.num_workers
        else:
            self.sub_batch_size = n_envs
            self.total_pop = n_envs

        if self.total_pop != n_envs:
            print(f"Warning: Total population adjusted from {n_envs} to "
                  f"{self.total_pop} to match worker count.")

        print(f"Initializing Env on {self.device} | Workers: {self.num_workers} | Total Pop: {self.total_pop}")

    @staticmethod
    def get_num_workers() -> int:
        available_cpus = multiprocessing.cpu_count()
        return max(1, available_cpus)

    def _make_env_fn(self, num_envs: int):
        """Factory for a single VmasEnv batch."""
        return VmasEnv(
            scenario=Scenario(),
            num_envs=num_envs,
            device=self.device,
            **self.cfg
        )

    def init_env(self, env_transform=None):
        """Initializes either a simple VmasEnv or a ParallelEnv."""
        if self.num_workers > 1 and self.device == "cpu":
            base_env = ParallelEnv(
                num_workers=self.num_workers,
                create_env_fn=lambda: self._make_env_fn(self.sub_batch_size)
            )
        else:
            base_env = self._make_env_fn(self.total_pop)

        if env_transform is not None:
            # Wrap the Main Env
            return TransformedEnv(base_env, env_transform())
        else:
            return base_env

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
                self.cfg.n_agents,
                N_CHANNELS
            )
            batch_size_shape = [self.num_workers, self.sub_batch_size]
        else:
            reshaped_genes = population_genes.view(
                total_pop,
                self.cfg.n_agents,
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
                max_steps=self.cfg.max_steps,
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

    def __init__(self, use_wandb: bool, cfg, save_fig: bool = False):
        self.use_wandb = use_wandb
        self.cfg = cfg
        self.save_fig = save_fig
        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        wandb.login()
        # Handle Hydra DictConfig or Argparse Namespace
        if isinstance(self.cfg, DictConfig):
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        else:
            config_dict = vars(self.cfg)

        wandb.init(
            project=self.cfg.project_name,
            name=self.cfg.run_name + "-" + str(uuid.uuid4())[:8],
            config=config_dict
        )

    def log_metrics_ga(self, gen: int, fitness_tensor: torch.Tensor, steps_per_sec: float, islands: List[List[Any]]):
        # 1. Global Statistics (All Agents)
        global_mean = fitness_tensor.mean().item()
        global_median = fitness_tensor.median().item()
        global_max = fitness_tensor.max().item()

        # 2. Per-Environment Statistics
        # fitness_tensor shape: [POP_SIZE, N_AGENTS]
        env_means = fitness_tensor.mean(dim=1)
        env_medians = fitness_tensor.median(dim=1).values

        # 3. Top-K Environments Statistics
        top_k = self.cfg.top_k
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

    def log_ternary_plot(self, plot_data: List[Dict[str, float]], resolution: int):
        """
        Generates and logs a ternary plot (Private vs Belief vs None).
        Expected plot_data structure: [{'priv': 0.1, 'bel': 0.8, 'none': 0.1, 'score': 10.5}, ...]
        """
        fig = self._generate_ternary_plot_fig(plot_data, resolution)

        if self.use_wandb:
            wandb.log({"ternary_landscape": wandb.Image(fig)})

        if self.save_fig:
            fname = "ternary_exploration.png"
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"Ternary plot saved to {fname}")

        plt.close(fig)

    def _generate_heatmap_fig(self, islands, fitness_tensor):
        # 1. Select Top K
        island_means = fitness_tensor.mean(dim=1)
        top_k = self.cfg.top_k
        top_k_indices = torch.argsort(island_means, descending=True)[:top_k].cpu().numpy()
        top_islands = [islands[i] for i in top_k_indices]
        # Extract the means for these islands (to use as labels)
        top_k_means = island_means[top_k_indices]

        # Extract logits -> probs
        all_genes = [ind for island in top_islands for ind in island]
        logits_tensor = torch.tensor(all_genes, dtype=torch.float32, device='cpu')
        logits_tensor = logits_tensor.view(top_k, self.cfg.n_agents, N_CHANNELS)
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
        n_agents = self.cfg.n_agents
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
            ax.axhline(y=i * self.cfg.n_agents - 0.5, color='white', linewidth=1.5, alpha=0.8)

        plt.tight_layout()
        return fig

    def _generate_parallel_plot_fig(self, islands, fitness_tensor):
        # 1. Select Top K
        island_means = fitness_tensor.mean(dim=1)
        top_k = self.cfg.top_k
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

    def _generate_ternary_plot_fig(self, data: List[Dict[str, float]], resolution: int):
        """
        Internal method to generate the matplotlib figure for the ternary plot using mpltern.
        """
        import mpltern
        # Extract components
        priv = [item['priv'] for item in data]
        bel = [item['bel'] for item in data]
        none = [item['none'] for item in data]
        scores = [item['score'] for item in data]

        fig = plt.figure(figsize=(10, 8))
        # Projection 'ternary' is provided by mpltern
        ax = fig.add_subplot(projection='ternary')

        # Order for mpltern is typically Top, Left, Right
        # We map:
        # Top = Belief
        # Left = Private
        # Right = None

        # tricontourf(t, l, r, values)
        cs = ax.tripcolor(bel, priv, none, scores, vmin=0, vmax=1, cmap="viridis",
                          shading='gouraud', rasterized=True)  # shading='flat'
        # ax.grid(alpha=0.2)
        ax.grid(axis='t', color='w')
        ax.grid(axis='l', color='w', linestyle='--')
        ax.grid(axis='r', color='w', linestyle=':')

        cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
        fig.colorbar(cs, label="Average Fitness", ticks=np.linspace(0, 1, 11), cax=cax)

        ax.set_tlabel("Belief")
        ax.set_llabel("Private")
        ax.set_rlabel("None")

        # ax.set_title(f"Channel Strategy Fitness Landscape\n(Grid Resolution: {resolution})")
        plt.tight_layout()
        return fig

    def finish(self):
        if self.use_wandb:
            wandb.finish()