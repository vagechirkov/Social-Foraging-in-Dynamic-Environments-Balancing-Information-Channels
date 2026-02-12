import streamlit as st
import numpy as np
import time
import os
import sys
import torch
import io
import matplotlib.pyplot as plt
from PIL import Image
from tensordict import TensorDict

# Add the project root to sys.path to allow importing the 'abm' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Color map for agents based on state
STATE_COLOR_MAP = {
    0: (0, 0.619, 0.451),  # private #009E73
    1: (0.337, 0.706, 0.914),  # social #56B4E9
    2: (0.337, 0.706, 0.914),  # social #56B4E9
    3: (0.337, 0.706, 0.914),  # social #56B4E9
    4: (0.902, 0.624, 0),  # none #E69F00
    5: (0.337, 0.706, 0.914), # social #56B4E9
}

st.set_page_config(page_title="ABM Simulation", layout="wide")
# st.title("Social Foraging ABM - Fast Heatmap")

# Lazy Import
try:
    from torchrl.envs import VmasEnv
    from abm.model import Scenario
    from abm.agent import ForagingAgent, TargetAgent
except ImportError as e:
    st.error(f"Failed to import VMAS/Scenario/Agents: {e}")
    st.stop()

# --- Simulation Settings ---
# st.sidebar.header("Simulation Settings")
n_agents = st.sidebar.slider("Number of Agents", 1, 50, 5)
n_targets = st.sidebar.slider("Number of Targets", 1, 10, 3)
fps = st.sidebar.slider("FPS Limit", 1, 260, 100)
plot_size = st.sidebar.slider("Plot Size", 6, 20, 8)

st.sidebar.header("Movement Distribution")
p_none = st.sidebar.slider("P(None)", 0.0, 1.0, 0.5)
p_private = st.sidebar.slider("P(Private)", 0.0, 1.0, 0.25)
p_belief = st.sidebar.slider("P(Belief)", 0.0, 1.0, 0.25)
p_consensus = st.sidebar.slider("P(Consensus)", 0.0, 1.0, 0.0)

st.sidebar.header("Target Behavior")
target_pattern = st.sidebar.selectbox("Target Movement Pattern", ["crw", "periodically_relocate", "levy"], index=2)
if target_pattern == "periodically_relocate":
    relocation_interval = st.sidebar.slider("Relocation Interval", 50, 1000, 250)
    persistence = 1 
    t_speed = 0.1 # Default for relocation
elif target_pattern == "crw":
    persistence = st.sidebar.slider("Persistence (Degrees)", 1, 90, 20)
    t_speed = st.sidebar.slider("Target Speed", 0.01, 1.0, 0.5)
    relocation_interval = 250
if target_pattern == "levy":
    relocation_interval = st.sidebar.slider("Relocation Interval", 50, 1000, 250)
    persistence = st.sidebar.slider("Persistence (Degrees)", 1, 90, 20)
    t_speed = st.sidebar.slider("Target Speed", 0.01, 1.0, 0.5)

# --- Session State ---
if 'env' not in st.session_state:
    st.session_state.env = None
if 'step' not in st.session_state:
    st.session_state.step = 0

def reset_simulation():
    # IMPORTANT: is_interactive must be False so the scenario 
    # doesn't override our actions in Scenario.process_action
    params = {
        'x_dim': 2, 'y_dim': 2, 
        'target_speed': t_speed,
        'n_agents': n_agents, 
        'n_targets': n_targets, 
        'targets_quality': 'HT',
        'is_interactive': False, 
        'initialization_box_ratio': 0.5,
        'viewer_zoom': 1.05, 
        'viewer_size': (400, 400),
        'visualize_semidims': True, 
        'min_dist_between_entities': 0.1,
        'agent_radius': 0.01, 
        'max_speed': 0.05,
        'dist_noise_scale_priv': 0.1,
        'dist_noise_scale_soc': 0,
        'social_trans_scale': 0.01,
        'belief_selectivity_threshold': 100.,
        'process_noise_scale': 0.1, 
        'cost_priv': 0.0,
        'cost_belief': 0.0,
        'base_noise': 0.1,
        'cost_consensus': 0.0,
        'consensus_selectivity_threshold': 0.1,
        'target_persistence': persistence, 
        'target_movement_pattern': target_pattern, 
        'relocation_interval': relocation_interval, 
        'process_noise_scale_het_ratio': 0, 
        'process_noise_scale_het_scale': 10
    }
    env = VmasEnv(scenario=Scenario(), num_envs=1, device="cpu", **params)
    env.reset()
    st.session_state.env = env
    st.session_state.step = 0

col1, col2 = st.columns(2)
if col2.button("Initialize / Reset"): reset_simulation()
run_simulation = col2.toggle("Run Simulation", value=False)
placeholder = col1.empty()

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

def render_heatmap(env, fig, ax):
    raw_env = env._env
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

    # 3. Plot Agents
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

    # 2. Plot Targets
    for agent in raw_env.world.agents:
        if "target" in agent.name or isinstance(agent, TargetAgent):
            pos = agent.state.pos[0].detach().cpu().numpy()
            quality = getattr(agent, 'quality', 1.0)
            if torch.is_tensor(quality):
                quality = quality[0].item()
            ax.scatter(pos[0], pos[1], color='red', marker='*', s=200 * quality, edgecolors='black', zorder=10)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=80)
    buf.seek(0)
    return Image.open(buf)


if st.session_state.env is not None:
    env = st.session_state.env
    fig, ax = plt.subplots(figsize=(plot_size, plot_size))
    
    if run_simulation:
        # Probabilities normalization
        total_p = p_none + p_private + p_belief + p_consensus
        if total_p > 0:
            p_n, p_p, p_b, p_c = p_none/total_p, p_private/total_p, p_belief/total_p, p_consensus/total_p
        else:
            p_n, p_p, p_b, p_c = 1.0, 0.0, 0.0, 0.0

        while True:
            # Action Mapping
            # 0: Priv, 4: None, 5: Consensus
            probs = torch.zeros(6)
            probs[0] = p_p
            probs[1] = p_b
            probs[4] = p_n
            probs[5] = p_c
            
            # Sample for all FORAGING agents
            foraging_actions = torch.distributions.Categorical(probs=probs).sample((n_agents,))
            # Dummies for TARGET agents (they are also in the environment's agents list)
            target_actions = torch.zeros(n_targets, dtype=torch.long)
            all_actions = torch.cat([foraging_actions, target_actions])
            
            td = TensorDict({"agents": TensorDict({"action": all_actions.unsqueeze(0)}, 
                            batch_size=[1, n_agents + n_targets])}, batch_size=[1])
            env.step(td)
            st.session_state.step += 1
            
            img = render_heatmap(env, fig, ax)
            placeholder.image(img, caption=f"Step {st.session_state.step}")
            time.sleep(1/fps)
    else:
        img = render_heatmap(env, fig, ax)
        placeholder.image(img, caption=f"Paused - Step {st.session_state.step}")
    plt.close(fig)
else:
    st.info("Initialize to start.")
