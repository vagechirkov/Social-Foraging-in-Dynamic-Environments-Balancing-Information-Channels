import hydra
import torch
import wandb
import numpy as np
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import VmasEnv

from abm.utils import VmasEvaluator, GenePersistenceTransform, ExperimentLogger
from abm.utils import N_GENES, CHANNEL_NAMES, step_tensordict


@hydra.main(version_base=None, config_path=".", config_name="ssga_2_targets")
def run_ssga(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    evaluator = VmasEvaluator(cfg, device)
    env = evaluator.init_env(env_transform=GenePersistenceTransform)
    
    # Must run on VmasEnv to directly modify Scenario
    assert isinstance(env.base_env, VmasEnv), "SSGA requires direct VmasEnv (not ParallelEnv). Ensure GPU is used or CPU workers=1."
    
    logger = ExperimentLogger(cfg.use_wandb, cfg, save_fig_locally=False)
    
    # Initialize Population
    n_islands = evaluator.total_pop
    n_agents = cfg.n_agents
    
    # Population initialization
    if cfg.init_strategy == "uniform":
        channel_logits = torch.empty(n_islands, n_agents, 3, device=device).uniform_(-2, 2)
    elif cfg.init_strategy == "middle":
        # All probabilities around 0.33 -> logits around 0
        channel_logits = torch.empty(n_islands, n_agents, 3, device=device).normal_(0, 0.5)
    elif cfg.init_strategy == "all_private":
        # Private (index 0) around 0.9, others low. log(0.9/0.05) approx 2.9
        channel_logits = torch.zeros(n_islands, n_agents, 3, device=device).normal_(0, 0.5)
        channel_logits[..., 0] += 3.0
    else:
        raise ValueError(f"Unknown init_strategy: {cfg.init_strategy}")
    # EVE (3) initialized around 0 (sigmoid(0) = 0.5 -> scaled to ~0.1 process noise)
    eve_logits = torch.empty(n_islands, n_agents, 1, device=device).normal_(0, 0.5)
    
    population_genes = torch.cat([channel_logits, eve_logits], dim=-1)
    
    batch_size_shape = [n_islands]
    
    initial_tensordict = TensorDict(
        {"genes": population_genes},
        batch_size=batch_size_shape,
        device=device
    )
    
    def policy_func(genes_logits):
        channel_logits = genes_logits[..., :3]
        dist = torch.distributions.Categorical(logits=channel_logits)
        action_choice = dist.sample().unsqueeze(-1).float()
        
        eve_raw = genes_logits[..., 3:4]
        eve = torch.sigmoid(eve_raw) * (cfg.eve_max - cfg.eve_min) + cfg.eve_min
        
        return torch.cat([action_choice, eve], dim=-1)

    policy = TensorDictModule(policy_func, in_keys=["genes"], out_keys=[env.action_key])
    
    env.reset(initial_tensordict)
    last_td = initial_tensordict
    
    eval_interval = cfg.ssga.eval_interval
    cull_fraction = cfg.ssga.cull_fraction
    n_cull = max(1, int(n_agents * cull_fraction))
    
    total_ticks = 0
    max_ticks = cfg.max_ticks
    
    print(f"Starting SSGA Simulation for {max_ticks} ticks (eval interval: {eval_interval})...")
    
    while total_ticks < max_ticks:
        with torch.inference_mode():
            rollouts = env.rollout(
                max_steps=eval_interval,
                policy=policy,
                tensordict=last_td,
                auto_reset=False,
                auto_cast_to_device=True,
            )
            last_td = step_tensordict(rollouts[..., -1])
            
            # 1. Calculate fitness for this interval
            # [Islands, Steps, Agents, 1]
            rewards = rollouts["next", "agents", "reward"]
            interval_fitness = rewards.mean(dim=1).squeeze(-1) # [Islands, Agents]
            
            # Phenotypes for analysis
            genes = last_td["genes"] # [Islands, Agents, 4]
            # Probabilities for analysis [Islands, Agents, 3]
            probs = torch.softmax(genes[..., :3], dim=-1)
            # Raw EVE for analysis [Islands, Agents, 1]
            eve_val = torch.sigmoid(genes[..., 3:4]) * (cfg.eve_max - cfg.eve_min) + cfg.eve_min
            phenotypes = torch.cat([probs, eve_val], dim=-1) # [Islands, Agents, 4]

            # Logging Per-Tick Avg Fitness and Genes
            if cfg.use_wandb:
                step_avg_fit = rewards.mean(dim=(0, 2, 3)).cpu().tolist()
                avg_probs = probs.mean(dim=(0, 1)).cpu().tolist()
                avg_eve = eve_val.mean().item()
                
                for step_idx in range(eval_interval):
                    wandb.log({
                        "tick/avg_fitness": step_avg_fit[step_idx],
                        "tick/avg_prob_priv": avg_probs[0],
                        "tick/avg_prob_soc": avg_probs[1],
                        "tick/avg_prob_none": avg_probs[2],
                        "tick/avg_eve": avg_eve
                    }, step=total_ticks + step_idx)
            
            # 2. Price Equation Components
            
            w_bar = interval_fitness.mean(dim=1, keepdim=True) # [Islands, 1]
            w_rel = interval_fitness / (w_bar + 1e-8) # [Islands, Agents]
            
            z_bar = phenotypes.mean(dim=1, keepdim=True) # [Islands, 1, 4]
            dz = phenotypes - z_bar # [Islands, Agents, 4]
            dw_rel = w_rel - w_rel.mean(dim=1, keepdim=True) # [Islands, Agents]
            
            # Covariance: E[(w_rel - 1)(z - z_bar)]
            covs = (dw_rel.unsqueeze(-1) * dz).mean(dim=1) # [Islands, 4]
            global_covs = covs.mean(dim=0) # [4]
            
            if cfg.use_wandb and (total_ticks % (eval_interval * cfg.log_freq) == 0):
                logger.log_ssga_metrics(
                    interval_fitness, 
                    probs, 
                    eve_val, 
                    genes, 
                    global_covs, 
                    total_ticks + eval_interval - 1, 
                    n_islands
                )
            
            # 3. Vectorized SSGA Culling & Cloning
            # Sort fitness
            sorted_fit, sort_indices = torch.sort(interval_fitness, dim=1, descending=True)
            
            # Identify bottom n_cull
            bottom_indices = sort_indices[:, -n_cull:] # [Islands, n_cull]
            # Identify top n_cull to clone
            top_indices = sort_indices[:, :n_cull] # [Islands, n_cull]
            
            # Create newborn mask
            newborn_mask = torch.zeros(n_islands, n_agents, dtype=torch.bool, device=device)
            newborn_mask.scatter_(1, bottom_indices, True)
            
            # Extract top genes
            # [Islands, n_cull, 1] expanded to [Islands, n_cull, 4]
            top_genes = torch.gather(genes, 1, top_indices.unsqueeze(-1).expand(-1, -1, N_GENES))
            
            # Mutate
            mutation_mask = (torch.rand_like(top_genes) < cfg.ssga.mutation_prob)
            mutation_noise = torch.randn_like(top_genes) * cfg.ssga.mutation_sigma
            cloned_genes = top_genes + (mutation_noise * mutation_mask)
            
            # Scatter cloned genes into bottom indices
            genes.scatter_(1, bottom_indices.unsqueeze(-1).expand(-1, -1, N_GENES), cloned_genes)
            last_td["genes"] = genes
            
            # 4. Partial Reinitialization in model.py
            env.base_env.scenario.reinitialize_agents(newborn_mask)
            
            total_ticks += eval_interval

    print("Simulation Complete.")
    env.close()
    logger.finish()

if __name__ == "__main__":
    run_ssga()
