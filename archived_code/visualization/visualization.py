import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from tensordict import TensorDict

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from abm.model import Scenario
from torchrl.envs import VmasEnv
from abm.utils import STATE_COLOR_MAP, get_gaussian_density
from abm.agent import ForagingAgent, TargetAgent

def create_agent_marker():
    # A pentagon pointing to the right (along positive x-axis)
    # Vertices: bottom-left, bottom-right, tip, top-right, top-left
    verts = [
        (-0.5, -0.4),
        (0.2, -0.4),
        (0.6, 0.0),
        (0.2, 0.4),
        (-0.5, 0.4),
        (-0.5, -0.4), # close path
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    return Path(verts, codes)

def render_frame(env, ax_main, heat_axs, frame_idx):
    if hasattr(env, '_env'):
         raw_env = env._env
    else:
         raw_env = env
         
    scenario = raw_env.scenario
    ax_main.clear()
    ax_main.set_xlim(-scenario.x_dim, scenario.x_dim)
    ax_main.set_ylim(-scenario.y_dim, scenario.y_dim)
    ax_main.set_aspect('equal')
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_title(f"Environment - Step {frame_idx}", fontsize=24)

    # Plot Target
    for agent in raw_env.world.agents:
        if "target" in agent.name or isinstance(agent, TargetAgent):
            pos = agent.state.pos[0].detach().cpu().numpy()
            ax_main.scatter(pos[0], pos[1], color='blue', s=800, edgecolors='black', zorder=10) # Larger blue dot

    # Plot Agents
    agent_idx = 0
    agent_marker_path = create_agent_marker()
    
    for agent in raw_env.world.agents:
        if "agent" in agent.name and isinstance(agent, ForagingAgent):
            pos = agent.state.pos[0].detach().cpu().numpy()
            vel = agent.state.vel[0].detach().cpu().numpy()
            
            # Calculate angle from velocity, default to 0 if stationary
            angle = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel) > 1e-4 else 0.0
            
            color = (0, 0, 1) # Default blue
            if hasattr(agent, 'action') and agent.action.u is not None:
                idx = int(agent.action.u[0, 0].item())
                # Handle old indices mapping if they happen to be used
                if idx == 4: idx = 2
                if idx == 5: idx = 1
                color = STATE_COLOR_MAP.get(idx, (0.5, 0.5, 0.5))
                
            # Create a transformed path for this agent
            # Scale needs to be adjusted based on environment bounds
            scale = scenario.x_dim * 0.1
            t = plt.matplotlib.transforms.Affine2D().scale(scale).rotate(angle).translate(pos[0], pos[1])
            patch = PathPatch(agent_marker_path, transform=t + ax_main.transData, 
                              facecolor=color, edgecolor='black', alpha=0.9, zorder=15)
            ax_main.add_patch(patch)
            
            # Plot Belief Heatmap for this agent
            if agent_idx < 4:
                ax = heat_axs[agent_idx]
                ax.clear()
                ax.set_xlim(-scenario.x_dim, scenario.x_dim)
                ax.set_ylim(-scenario.y_dim, scenario.y_dim)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Agent {agent_idx} Belief", fontsize=20)
                
                res = 60
                total_density = np.zeros((res, res))
                means = agent.belief_target_pos[0].detach().cpu().numpy()
                covs = agent.belief_target_covariance[0].detach().cpu().numpy()
                for k in range(len(means)):
                    if not np.allclose(covs[k], 0):
                        density = get_gaussian_density(means[k], covs[k], 
                                                       [-scenario.x_dim, scenario.x_dim], 
                                                       [-scenario.y_dim, scenario.y_dim], res=res)
                        total_density += density
                if np.max(total_density) > 0:
                    ax.imshow(total_density, extent=[-scenario.x_dim, scenario.x_dim, -scenario.y_dim, scenario.y_dim],
                              origin='lower', alpha=0.8, cmap='Blues', interpolation='bilinear', zorder=1)
                
                # Plot the corresponding agent position
                ax.scatter(pos[0], pos[1], color='red', s=150, marker='x', zorder=5)
                
            agent_idx += 1

def main():
    n_agents = 4
    n_targets = 1
    
    # Base parameters matching standard simulation setup
    params = {
        'x_dim': 2, 'y_dim': 2, 
        'target_speed': 0.5,
        'n_agents': n_agents, 
        'n_targets': n_targets, 
        'target_quality': 'HT',
        'is_interactive': False, 
        'visualize_semidims': True, 
        'min_dist_between_entities': 0.1,
        'agent_radius': 0.01, 
        'max_speed': 0.05,
        'dist_noise_scale_priv': 0.05,
        'dist_noise_scale_soc': 0,
        'social_trans_scale': 0.01,
        'belief_selectivity_threshold': 0.25,
        'process_noise_scale': 0.08, 
        'cost_priv': 0.0,
        'cost_belief': 0.0,
        'base_noise': 0.1,
        'cost_consensus': 0.0,
        'consensus_selectivity_threshold': 0.1,
        'target_persistence': 25, 
        'target_movement_pattern': 'crw', 
        'relocation_interval': 250, 
        'process_noise_scale_het_ratio': 0, 
        'process_noise_scale_het_scale': 10,
        'bias_magnitude': 0,
        'decision_making': 'greedy',
        'n_private_samples': 1,
        'p_spatial_explore': 0.01,
        'channel_y_name': "Belief",
        'spot_radius': 1.0,
    }
    
    # Setup environment
    env = VmasEnv(scenario=Scenario(), num_envs=1, device="cpu", **params)
    env.reset()
    
    raw_env = env._env if hasattr(env, '_env') else env
    
    # Initialize the target closer to one corner (top-right) but not in the corner
    for t in raw_env.scenario.target_agents:
        t.state.pos[:, 0] = params['x_dim'] * 0.7
        t.state.pos[:, 1] = params['y_dim'] * 0.7
        
    # Initialize agents with broad random belief
    for agent in raw_env.scenario.foraging_agents:
        agent.belief_target_covariance.copy_(torch.eye(2).reshape(1, 1, 2, 2).repeat(1, n_targets, 1, 1) * 2.0)
    
    # Channel qualities requested: 0.4, 0.3, 0.3
    # Mapped to: Private, Social(Belief), None
    p_p = 0.2
    p_b = 0.6
    p_n = 0.2
    
    probs = torch.zeros(3)
    probs[0] = p_p
    probs[1] = p_b
    probs[2] = p_n

    # Layout setup: 2 rows, 3 columns. 
    # Left column (span 2 rows, 2 columns) for Environment.
    # Right column (2x2 grid in remaining space) for the 4 heatmaps.
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 3)
    
    ax_main = fig.add_subplot(gs[:, 0:2])
    ax_h1 = fig.add_subplot(gs[0, 2])
    ax_h2 = fig.add_subplot(gs[1, 2])
    # The prompt asked for "heatmaps a two by two matrix on the second column".
    # A 2x2 matrix would mean we need 2 columns on the right. So total 4 columns.
    
    plt.close(fig) # close the temporary figure
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    ax_main = fig.add_subplot(gs[:, 0:2])
    ax_h1 = fig.add_subplot(gs[0, 2])
    ax_h2 = fig.add_subplot(gs[0, 3])
    ax_h3 = fig.add_subplot(gs[1, 2])
    ax_h4 = fig.add_subplot(gs[1, 3])
    
    heat_axs = [ax_h1, ax_h2, ax_h3, ax_h4]
    
    # Add dummy titles so tight_layout reserves space
    ax_main.set_title("Environment - Step 000", fontsize=24)
    for i, ax in enumerate(heat_axs):
        ax.set_title(f"Agent {i} Belief", fontsize=20)
        
    # Tight layout
    plt.tight_layout(pad=2.0)
    
    max_steps = 150
    
    def update(frame):
        # Sample for all FORAGING agents
        foraging_actions = torch.distributions.Categorical(probs=probs).sample((n_agents,))
        
        # Determine how action is passed based on action_size in env.
        # usually it is a 2D tensor: [channel_idx, process_noise_scale]
        foraging_actions_2d = torch.stack([
            foraging_actions.float(), 
            torch.full((n_agents,), 0.05)
        ], dim=-1)
        
        # Dummies for TARGET agents
        target_actions_2d = torch.zeros((n_targets, 2))
        all_actions = torch.cat([foraging_actions_2d, target_actions_2d], dim=0)
        
        td = TensorDict({"agents": TensorDict({"action": all_actions.unsqueeze(0)}, 
                        batch_size=[1, n_agents + n_targets])}, batch_size=[1])
        env.step(td)
        
        render_frame(env, ax_main, heat_axs, frame)
        return []

    print("Generating video...")
    ani = animation.FuncAnimation(fig, update, frames=max_steps, blit=False)
    
    output_file = "visualization.mp4"
    writer = animation.FFMpegWriter(fps=5, bitrate=3000)
    ani.save(output_file, writer=writer)
    plt.close(fig)
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    main()
