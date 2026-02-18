import multiprocessing
import uuid
from typing import Any, List, Dict, Optional

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import ParallelEnv, TransformedEnv, VmasEnv
from torchrl.envs.transforms import Transform

from .model import Scenario
from .agent import ForagingAgent, TargetAgent

STATE_COLOR_MAP = {
    0: (0, 0.619, 0.451),  # private #009E73
    1: (0.337, 0.706, 0.914),  # social #56B4E9
    2: (0.337, 0.706, 0.914),  # social #56B4E9
    3: (0.337, 0.706, 0.914),  # social #56B4E9
    4: (0.902, 0.624, 0),  # none #E69F00
    5: (0.337, 0.706, 0.914), # social #56B4E9
}


def get_gaussian_density(mean, cov, x_range, y_range, res=50):
    x = np.linspace(x_range[0], x_range[1], res)
    y = np.linspace(y_range[0], y_range[1], res)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    try:
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_factor = 1.0 / (2 * np.pi * np.sqrt(max(det_cov, 1e-12)))
        diff = pos - mean
        exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=-1)
        return norm_factor * np.exp(exponent)
    except:
        return np.zeros((res, res))


def render_env_frame(env, ax):
    """
    Renders the current state of the environment onto the provided matplotlib axis.
    """
    if hasattr(env, '_env'):
         raw_env = env._env
    else:
         raw_env = env
         
    scenario = raw_env.scenario
    ax.clear()
    ax.set_xlim(-scenario.x_dim, scenario.x_dim)
    ax.set_ylim(-scenario.y_dim, scenario.y_dim)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

    # 1. Background Heatmap - Agent 0 Beliefs
    agent_0 = next((a for a in raw_env.agents if a.name == "agent_0"), None)
    if agent_0 and hasattr(agent_0, 'belief_target_pos'):
        res = 60
        total_density = np.zeros((res, res))
        means = agent_0.belief_target_pos[0].detach().cpu().numpy()
        covs = agent_0.belief_target_covariance[0].detach().cpu().numpy()
        for k in range(len(means)):
            if not np.allclose(covs[k], 0):
                density = get_gaussian_density(means[k], covs[k], 
                                               [-scenario.x_dim, scenario.x_dim], 
                                               [-scenario.y_dim, scenario.y_dim], res=res)
                total_density += density
        if np.max(total_density) > 0:
            ax.imshow(total_density, extent=[-scenario.x_dim, scenario.x_dim, -scenario.y_dim, scenario.y_dim],
                      origin='lower', alpha=0.4, cmap='Blues', interpolation='bilinear', zorder=1)

    # 2. Plot Agents
    agents_x, agents_y, colors, sizes = [], [], [], []
    for agent in raw_env.world.agents:
        if "agent" in agent.name and isinstance(agent, ForagingAgent):
            pos = agent.state.pos[0].detach().cpu().numpy()
            agents_x.append(pos[0])
            agents_y.append(pos[1])
            sizes.append(150 if agent.name == "agent_0" else 50)
            color = (0, 0, 1) # Default blue
            if hasattr(agent, 'action') and agent.action.u is not None:
                # Use environment index 0 for current categorical state
                idx = int(agent.action.u[0, 0].item())
                color = STATE_COLOR_MAP.get(idx, (0, 0, 1))
            colors.append(color)
            
    if agents_x:
        ax.scatter(agents_x, agents_y, c=colors, s=sizes, edgecolors='black', alpha=0.9, zorder=15)

    # 3. Plot Targets
    for agent in raw_env.world.agents:
        if "target" in agent.name or isinstance(agent, TargetAgent):
            pos = agent.state.pos[0].detach().cpu().numpy()
            quality = getattr(agent, 'quality', 1.0)
            if torch.is_tensor(quality):
                quality = quality[0].item()
            ax.scatter(pos[0], pos[1], color='red', marker='*', s=200 * quality, edgecolors='black', zorder=10)
    
    return ax


CHANNEL_NAMES = ["Priv", "Belief", "Heading", "Pos", "None", "Consensus"]
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
            if n_envs < self.num_workers:
                self.num_workers = n_envs
                self.sub_batch_size = 1
                print(f"reduced workers to {self.num_workers} as n_envs < cpu_count")
            else:
                # Use ceiling division to ensure we can cover all n_envs
                self.sub_batch_size = (n_envs + self.num_workers - 1) // self.num_workers
            
            self.total_pop = self.sub_batch_size * self.num_workers
        else:
            self.sub_batch_size = n_envs
            self.total_pop = n_envs

        if self.total_pop != n_envs:
            print(f"Note: Evaluator capacity adjusted from {n_envs} to "
                  f"{self.total_pop} to fit parallel batching (padding will be used).")

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

        n_requested_islands = len(islands)
        
        # Check if padding is needed
        if n_requested_islands < self.total_pop:
            # Pad with zeros to match total_pop.
            # These padded agents will run but their results will be discarded.
            n_pad = self.total_pop - n_requested_islands
            pad_genes = torch.zeros(n_pad * self.cfg.n_agents, N_CHANNELS, device=self.device)
            population_genes = torch.cat([population_genes, pad_genes], dim=0)
        
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
                self.total_pop,
                self.cfg.n_agents,
                N_CHANNELS
            )
            batch_size_shape = [self.total_pop]

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
        # rollouts["next", "agents", "reward"] has shape [Batch..., Steps, Agents, 1]
        rewards_all = rollouts["next", "agents", "reward"]

        # Sum rewards over the time dimension (Steps)
        # The time dimension is typically the one after the batch dimensions.
        # If batch_size_shape is [W, SB], then rewards_all is [W, SB, Steps, Agents, 1]
        # If batch_size_shape is [TP], then rewards_all is [TP, Steps, Agents, 1]
        time_dim_idx = len(batch_size_shape)
        total_rewards = rewards_all.sum(dim=time_dim_idx) # Shape: [Batch..., Agents, 1]

        # Flatten batch dimensions and remove the last '1' dimension
        # Resulting shape should be [TotalPop, Agents]
        if isinstance(env.base_env, ParallelEnv):
            total_rewards = total_rewards.view(self.total_pop, self.cfg.n_agents)
        else:
            total_rewards = total_rewards.view(self.total_pop, self.cfg.n_agents)
        
        # Remove Padding if applied
        if n_requested_islands < self.total_pop:
            total_rewards = total_rewards[:n_requested_islands]

        # Assign Fitness
        # Flatten total_rewards to match flat_population for assignment
        flat_rewards = total_rewards.cpu().numpy().flatten()
        for ind, reward in zip(flat_population, flat_rewards):
            ind.fitness.values = (float(reward),)

        return total_rewards


class ExperimentLogger:
    """Handles visualization and WandB logging."""

    def __init__(self, use_wandb: bool, cfg, save_fig_locally: bool = False,
                 frozen_indices: Optional[List[int]] = None):
        self.use_wandb = use_wandb
        self.cfg = cfg
        self.save_fig = save_fig_locally
        self.frozen_indices = frozen_indices if frozen_indices is not None else []
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

    def log_metrics_ga(self, gen: int, fitness_tensor: torch.Tensor, steps_per_sec: float, islands: List[List[Any]], log_table: bool = False):
        # 1. Global Statistics (All Agents)
        global_mean = fitness_tensor.mean().item()
        global_median = fitness_tensor.median().item()
        global_max = fitness_tensor.max().item()

        # 2. Per-Environment Statistics
        env_means = fitness_tensor.mean(dim=1)
        
        # 3. Top-K Environments
        top_k = self.cfg.top_k
        top_k_indices = torch.argsort(env_means, descending=True)[:top_k]

        # Get all agent rewards for the top k islands
        top_k_agents_fitness = fitness_tensor[top_k_indices]
        
        print(f"Gen {gen:4d} | Global Mean: {global_mean:8.4f} | "
              f"Global Max: {global_max:8.4f} | "
              f"Speed: {steps_per_sec:.0f} env-steps/s")

        if self.use_wandb:
            metrics = {
                "gen": gen,
                
                # Scalars
                "global_mean": global_mean,
                "global_max": global_max,
                "steps_per_sec": steps_per_sec,
                
                # Fitness Histograms
                "hist/fitness_global": wandb.Histogram(fitness_tensor.cpu().numpy().flatten()),
                "hist/fitness_top_k": wandb.Histogram(top_k_agents_fitness.cpu().numpy().flatten()),
                
                # Env Mean Histogram
                "hist/env_means": wandb.Histogram(env_means.cpu().numpy()),
            }

            # 4. Channel Probabilities (Histograms)
            # Calculate probabilities for ALL agents
            all_genes = [ind for island in islands for ind in island]
            all_logits = torch.tensor(all_genes, dtype=torch.float32, device='cpu')
            all_probs = torch.softmax(all_logits, dim=-1) # [TotalAgents, N_CHANNELS]
            
            # Calculate probabilities for Top-K agents
            top_k_indices_list = top_k_indices.cpu().numpy()
            top_k_genes = [ind for i in top_k_indices_list for ind in islands[i]]
            top_k_logits = torch.tensor(top_k_genes, dtype=torch.float32, device='cpu')
            top_k_probs = torch.softmax(top_k_logits, dim=-1) # [TopK*Agents, N_CHANNELS]

            # Log distributions for active channels only
            active_indices = [i for i in range(N_CHANNELS) if i not in self.frozen_indices]
            
            for i in active_indices:
                name = CHANNEL_NAMES[i]
                # Global Histogram for this channel
                metrics[f"hist/prob_global_{name}"] = wandb.Histogram(all_probs[:, i].numpy())
                # Top-K Histogram for this channel
                metrics[f"hist/prob_top_k_{name}"] = wandb.Histogram(top_k_probs[:, i].numpy())
   
            # 5. Best Individual Scalars
            best_ind_idx = torch.argmax(fitness_tensor.view(-1))
            best_probs = all_probs[best_ind_idx]
            metrics["best/fitness"] = global_max
            for i in active_indices:
               metrics[f"best/prob_{CHANNEL_NAMES[i]}"] = best_probs[i].item()

            # 6. Average Probabilities (Scalars)
            avg_probs = all_probs.mean(dim=0)
            for i in active_indices:
                metrics[f"avg/prob_{CHANNEL_NAMES[i]}"] = avg_probs[i].item()

            # 7. Periodic Table Logging (Snapshot)
            if log_table:
                columns = ["gen", "fitness"] + [f"prob_{CHANNEL_NAMES[i]}" for i in active_indices] 
                tbl = wandb.Table(columns=columns)
                
                # Log a sample of Top-K agents (e.g., up to 100 to avoid huge tables)
                # Randomly sample or take top N
                n_sample = min(len(top_k_probs), 100)
                # Sort top_k_probs by fitness if we had individual fitness readily aligned...
                # For now just take the first n_sample of the Top-K block
                for i in range(n_sample):
                    row = [gen, top_k_agents_fitness.view(-1)[i].item()]
                    for ch_idx in active_indices:
                        row.append(top_k_probs[i, ch_idx].item())
                    tbl.add_data(*row)
                
                metrics["population_snapshot"] = tbl

            wandb.log(metrics, step=gen)

    def log_heatmap(self, islands, fitness_tensor, gen):
        """Generates and logs the strategy heatmap."""
        fig = self._generate_heatmap_fig(islands, fitness_tensor)
        if self.use_wandb:
            wandb.log({"strategy_heatmap": wandb.Image(fig)}, step=gen)
        if self.save_fig:
            plt.savefig(f"heatmap_gen_{gen:04d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def log_ternary_density(self, islands, fitness_tensor, gen):
        """
        Generates ternary density plots (heatmaps) of individual strategies
        for both the Top K groups and the entire population.
        Requires exactly 3 active (non-frozen) channels.
        """
        active_indices = [i for i in range(N_CHANNELS) if i not in self.frozen_indices]

        if len(active_indices) != 3:
            return

        # 1. Plot All Agents
        all_agents = [ind for island in islands for ind in island]
        fig_all = self._generate_ternary_density_fig(all_agents, len(islands), active_indices,
                                                     "All Agents Strategy Density")

        # 2. Plot Top K Agents
        island_means = fitness_tensor.mean(dim=1)
        top_k = self.cfg.top_k
        top_k_indices = torch.argsort(island_means, descending=True)[:top_k].cpu().numpy()
        top_islands = [islands[i] for i in top_k_indices]
        top_k_agents = [ind for island in top_islands for ind in island]

        fig_top = self._generate_ternary_density_fig(top_k_agents, len(top_islands), active_indices,
                                                     f"Top {len(top_islands)} Groups Strategy Density")

        if self.use_wandb:
            wandb.log({
                "ternary_density_all": wandb.Image(fig_all),
                "ternary_density_top_k": wandb.Image(fig_top)
            }, step=gen)

        if self.save_fig:
            fig_all.savefig(f"ternary_all_gen_{gen:04d}.png", dpi=150, bbox_inches='tight')
            fig_top.savefig(f"ternary_top_k_gen_{gen:04d}.png", dpi=150, bbox_inches='tight')

        plt.close(fig_all)
        plt.close(fig_top)

    def log_parallel_plot(self, islands, fitness_tensor, gen):
        """Generates and logs the parallel coordinate plot for Top K islands."""
        fig = self._generate_parallel_plot_fig(islands, fitness_tensor)
        if self.use_wandb:
            wandb.log({"parallel_plot": wandb.Image(fig)}, step=gen)
        if self.save_fig:
            plt.savefig(f"parallel_plot_{gen:04d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def log_ternary_plot(self, plot_data: List[Dict[str, float]], resolution: int, midpoint: Optional[float] = None):
        """
        Generates and logs a ternary plot (Private vs Belief vs None).
        Expected plot_data structure: [{'priv': 0.1, 'bel': 0.8, 'none': 0.1, 'score': 10.5}, ...]
        """
        fig = self._generate_ternary_plot_fig(plot_data, resolution, midpoint)

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
        
        # Use actual length of top_islands, which might be less than cfg.top_k if replicates < top_k
        actual_k = len(top_islands)
        logits_tensor = logits_tensor.view(actual_k, self.cfg.n_agents, N_CHANNELS)
        probs_tensor = torch.softmax(logits_tensor, dim=2)

        # 2. Sort Agents
        # Sort order: Pos(3), Heading(2), Belief(1), None(4), Priv(0), Consensus(5)
        sort_order_indices = [3, 2, 1, 4, 0, 5] 

        # Filter out frozen indices from sorting logic to avoid noise
        active_sort_indices = [idx for idx in sort_order_indices if idx not in self.frozen_indices]

        sorted_agents_tensor = probs_tensor.clone()

        for channel_idx in active_sort_indices:
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
        active_indices = [i for i in range(N_CHANNELS) if i not in self.frozen_indices]
        filtered_tensor = final_tensor[:, :, active_indices]
        filtered_names = [CHANNEL_NAMES[i] for i in active_indices]

        heatmap_data = filtered_tensor.view(-1, len(active_indices)).numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')

        fig.colorbar(im, label="Fitness")
        ax.set_xticks(np.arange(len(active_indices)))
        ax.set_xticklabels(filtered_names)
        ax.set_xlabel("Information Channel")

        # Set Y-axis labels to be the fitness scores
        n_agents = self.cfg.n_agents
        # Calculate centers for each island block
        actual_k = len(sorted_means)
        y_ticks = np.arange(actual_k) * n_agents + (n_agents / 2) - 0.5
        y_labels = [f"{mean:.2f}" for mean in sorted_means.cpu().numpy()]

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel("Avg Group Reward")

        ax.set_title(f"Top {actual_k} Groups")

        # Overlay lines
        for i in range(heatmap_data.shape[0]):
            ax.axhline(y=i - 0.5, color='black', linewidth=0.2, alpha=0.1)
        for i in range(1, actual_k):
            ax.axhline(y=i * self.cfg.n_agents - 0.5, color='white', linewidth=1.5, alpha=0.8)

        plt.tight_layout()
        return fig

    def _generate_parallel_plot_fig(self, islands, fitness_tensor):
        # 1. Select Top K
        island_means = fitness_tensor.mean(dim=1)
        top_k = self.cfg.top_k
        top_k_indices = torch.argsort(island_means, descending=True)[:top_k].cpu().numpy()
        top_islands = [islands[i] for i in top_k_indices]

        # Determine active channels for plotting
        active_indices = [i for i in range(N_CHANNELS) if i not in self.frozen_indices]
        filtered_names = [CHANNEL_NAMES[i] for i in active_indices]

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

            # Filter out frozen columns
            filtered_probs = island_probs[:, active_indices]

            # Plot lines for this island with a specific color
            # We use a distinct color for the group and add a label for the legend
            ax.plot(filtered_names, filtered_probs.T, color=colors[i], alpha=0.2, linewidth=1)

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

    def _generate_ternary_density_fig(self, agents, n_islands, active_indices, title):
        """Generates a ternary density plot (hexbin) for a list of individual agents."""
        import mpltern

        # Extract Genes -> Logits -> Probs
        logits = torch.tensor(agents, dtype=torch.float32, device='cpu')

        # Softmax over all channels first to get true probabilities relative to everything
        logits_tensor = logits.view(n_islands, self.cfg.n_agents, N_CHANNELS)
        probs_tensor = torch.softmax(logits_tensor, dim=2)

        # Extract active
        probs_active = probs_tensor[:, :, active_indices]
        vals = probs_active.view(n_islands * self.cfg.n_agents, 3).numpy()

        # Order for mpltern is typically Top, Left, Right
        # We assign:
        # Top   = Index 1 in active list
        # Left  = Index 0
        # Right = Index 2
        t = vals[:, 1]
        l = vals[:, 0]
        r = vals[:, 2]

        labels = [CHANNEL_NAMES[i] for i in active_indices]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='ternary')

        # Hexbin density
        # gridsize determines resolution of hexagons
        hb = ax.hexbin(t, l, r, gridsize=10)
        # ax.scatter(t, l, r)

        ax.set_tlabel(labels[1])
        ax.set_llabel(labels[0])
        ax.set_rlabel(labels[2])

        ax.grid(axis='t', color='gray', alpha=0.5)
        ax.grid(axis='l', color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='r', color='gray', linestyle=':', alpha=0.5)

        cb = fig.colorbar(hb, ax=ax, shrink=0.8)
        cb.set_label('Agent Count')

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def _generate_ternary_plot_fig(self, data: List[Dict[str, float]], resolution: int, midpoint: Optional[float] = None):
        """
        Internal method to generate the matplotlib figure for the ternary plot using mpltern.
        """
        import mpltern
        from matplotlib.colors import TwoSlopeNorm
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
        #  shading='gouraud', rasterized=True
        
        cmap = "RdBu_r" if midpoint is not None else "viridis"
        norm = TwoSlopeNorm(vmin=0, vcenter=midpoint, vmax=1) if midpoint is not None else None
        
        if norm is not None:
            cs = ax.tripcolor(bel, priv, none, scores, cmap=cmap, norm=norm, shading='flat')
        else:
            cs = ax.tripcolor(bel, priv, none, scores, cmap=cmap, shading='flat')
        # ax.grid(alpha=0.2)
        ax.grid(axis='t', color='w')
        ax.grid(axis='l', color='w', linestyle='--')
        ax.grid(axis='r', color='w', linestyle=':')

        cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
        fig.colorbar(cs, label="Average Fitness", cax=cax)  # , ticks=np.linspace(0, 1, 11)

        ax.set_tlabel("Belief")
        ax.set_llabel("Private")
        ax.set_rlabel("None")

        # ax.set_title(f"Channel Strategy Fitness Landscape\n(Grid Resolution: {resolution})")
        plt.tight_layout()
        return fig

    def finish(self):
        if self.use_wandb:
            wandb.finish()