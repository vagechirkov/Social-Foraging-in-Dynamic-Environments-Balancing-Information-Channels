import argparse
import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensordict import TensorDict

# Ensure we can import from abm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abm.model import Scenario
from abm.agent import ForagingAgent, TargetAgent
from torchrl.envs import VmasEnv
from abm.utils import render_env_frame

def run_simulation_and_save_video(config, category_name, category_params, seed, p_private, p_social, p_none, output_file):
    print(f"Generating video for category: {category_name}...")
    
    # 1. Parameter Setup
    # Base params from app.py logic
    base_params = {
        'x_dim': 2, 'y_dim': 2,
        'target_speed': 0.1, # Default
        'n_agents': 30, # Default in yaml
        'n_targets': 1,
        'targets_quality': 'HM',
        'is_interactive': False,
        'initialization_box_ratio': 1.0,
        'visualize_semidims': True,
        'min_dist_between_entities': 0.1,
        'agent_radius': 0.01, 
        'max_speed': 0.05,
        'dist_noise_scale_priv': 0.5,
        'dist_noise_scale_soc': 0.0,
        'social_trans_scale': 0.01,
        'belief_selectivity_threshold': 0.1,
        'process_noise_scale': 0.05, 
        'cost_priv': 0.0,
        'cost_belief': 0.0,
        'base_noise': 0.1,
        'cost_consensus': 0.0,
        'consensus_selectivity_threshold': 0.1,
        'target_persistence': 20, # Default
        'target_movement_pattern': 'crw',
        'relocation_interval': 1000,
        'process_noise_scale_het_ratio': 0.0,
        'process_noise_scale_het_scale': 1.0,
        'bias_magnitude': 0.0,
        'channel_y_name': "Belief",
        'spot_radius': 0.5
    }

    # Load from config parameters if they exist as root keys
    for k in base_params.keys():
        if k in config:
            base_params[k] = config[k]
        
    # Override with category params
    for k, v in category_params.items():
        base_params[k] = v

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize Environment
    # VmasEnv expects 'num_envs'
    env = VmasEnv(scenario=Scenario(), num_envs=1, device="cpu", **base_params)
    env.reset()

    # Probability normalization
    total_p = p_none + p_private + p_social
    p_consensus = 0.0 # Assuming consensus is not used in this request or default 0
    if total_p > 0:
        p_n = p_none / total_p
        p_p = p_private / total_p
        p_b = p_social / total_p # Mapping social to belief
        p_c = 0.0
    else:
        p_n, p_p, p_b, p_c = 1.0, 0.0, 0.0, 0.0

    # Simulation settings
    max_steps = config.get('max_steps', 1000)
    if 'max_steps_override' in config and config['max_steps_override'] is not None:
        max_steps = config['max_steps_override']

    
    # Matplotlib setup
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Data for updating plot
    # We will use a list to store frames for animation
    # Or creating a function for FuncAnimation

    def update(frame):
        # Step simulation
        probs = torch.zeros(6)
        probs[0] = p_p
        probs[1] = p_b
        probs[4] = p_n
        probs[5] = p_c
        
        # Sample actions
        n_agents = base_params['n_agents']
        n_targets = base_params['n_targets']
        
        foraging_actions = torch.distributions.Categorical(probs=probs).sample((n_agents,))
        target_actions = torch.zeros(n_targets, dtype=torch.long)
        all_actions = torch.cat([foraging_actions, target_actions])
        
        td = TensorDict({"agents": TensorDict({"action": all_actions.unsqueeze(0)}, 
                        batch_size=[1, n_agents + n_targets])}, batch_size=[1])
        env.step(td)

        # Render
        render_env_frame(env, ax)
        ax.set_title(f"{category_name} - Step {frame}")
        
        return ax.artists

    ani = animation.FuncAnimation(fig, update, frames=max_steps, blit=False)
    
    # Save video
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(output_file, writer=writer)
    plt.close(fig)
    print(f"Saved video to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate environment videos.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--p_private", type=float, required=True, help="Probability of private channel")
    parser.add_argument("--p_social", type=float, required=True, help="Probability of social channel")
    parser.add_argument("--p_none", type=float, required=True, help="Probability of no channel")
    parser.add_argument("--config", type=str, default="abm/ea_evaluation.yaml", help="Path to configuration file")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps for debugging")
    parser.add_argument("--output_dir", type=str, default="videos", help="Directory to save videos")
    parser.add_argument("--category", type=str, default=None, help="Specific category to generate video for")
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
        
    if args.max_steps is not None:
        config['max_steps_override'] = args.max_steps

    categories = config.get('environments', {}).get('categories', {})
    if not categories:
        print("No environment categories found in configuration.")
        sys.exit(1)

    if args.category:
        if args.category not in categories:
            print(f"Error: Category '{args.category}' not found in configuration.")
            print(f"Available categories: {list(categories.keys())}")
            sys.exit(1)
        # Filter to only the requested category
        categories = {args.category: categories[args.category]}

    for category_name, category_params in categories.items():
        output_file = os.path.join(args.output_dir, f"{category_name}_seed{args.seed}.mp4")
        run_simulation_and_save_video(config, category_name, category_params, args.seed, 
                                      args.p_private, args.p_social, args.p_none, output_file)

if __name__ == "__main__":
    main()
