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

from abm.utils import render_env_frame, STATE_COLOR_MAP

# Lazy Import
try:
    from torchrl.envs import VmasEnv
    from abm.model import Scenario
    from abm.agent import ForagingAgent, TargetAgent
except ImportError as e:
    st.error(f"Failed to import VMAS/Scenario/Agents: {e}")
    st.stop()

st.set_page_config(page_title="ABM Simulation", layout="wide")
# st.title("Social Foraging ABM - Fast Heatmap")

# --- Simulation Settings ---
# st.sidebar.header("Simulation Settings")
n_agents = st.sidebar.slider("Number of Agents", 1, 50, 10)
n_targets = st.sidebar.slider("Number of Targets", 1, 10, 2)
spot_radius = st.sidebar.slider("Spotlight Radius", 0.1, 2.0, 0.5)
target_circle_radius = st.sidebar.slider("Target Circle Radius", 0.0, 2.0, 0.4)

p_none = st.sidebar.slider("P(None)", 0.0, 1.0, 0.0)
p_private = st.sidebar.slider("P(Private)", 0.0, 1.0, 0.1)
p_belief = st.sidebar.slider("P(Belief)", 0.0, 1.0, 0.9)
# p_consensus = st.sidebar.slider("P(Consensus)", 0.0, 1.0, 0.0)
p_consensus = 0.0

target_pattern = st.sidebar.selectbox("Target Movement Pattern", ["crw", "periodically_relocate", "levy"], index=0)
if target_pattern == "periodically_relocate":
    relocation_interval = st.sidebar.slider("Relocation Interval", 50, 1000, 250)
    persistence = 1 
    t_speed = 0.1 # Default for relocation
elif target_pattern == "crw":
    persistence = st.sidebar.slider("Persistence (Degrees)", 1, 90, 25)
    t_speed = st.sidebar.slider("Target Speed", 0.01, 2.0, 0.3)
    relocation_interval = 250
if target_pattern == "levy":
    relocation_interval = st.sidebar.slider("Relocation Interval", 50, 1000, 250)
    persistence = st.sidebar.slider("Persistence (Degrees)", 1, 90, 20)
    t_speed = st.sidebar.slider("Target Speed", 0.01, 2.0, 0.5)


decision_making = st.sidebar.selectbox("Decision Making", ["sum", "greedy", "thompson"], index=1)

# fps = st.sidebar.slider("FPS Limit", 1, 260, 100)
fps = 100
plot_size = st.sidebar.slider("Plot Size", 6, 20, 12)

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
        'target_quality': 'HT',
        # 'target_speeds': [0.3, 0.5],
        'target_qualities': [0.05, 1.0],
        'is_interactive': False, 
        'initialization_box_ratio': 1.0,
        'visualize_semidims': True, 
        'min_dist_between_entities': 0.1,
        'agent_radius': 0.01, 
        'max_speed': 0.05,
        'dist_noise_scale_priv': 0.5,
        'dist_noise_scale_soc': 0,
        'social_trans_scale': 0.01,
        'belief_selectivity_threshold': 0.1,
        'process_noise_scale': 0.05, 
        'cost_priv': 0.0,
        'cost_belief': 0.0,
        'base_noise': 0.1,
        'cost_consensus': 0.0,
        'consensus_selectivity_threshold': 0.1,
        'target_persistence': persistence, 
        'target_movement_pattern': target_pattern, 
        'relocation_interval': relocation_interval, 
        'process_noise_scale_het_ratio': 0, 
        'process_noise_scale_het_scale': 10,
        'bias_magnitude': 0,
        'decision_making': decision_making,
        'spot_radius': spot_radius,
        'channel_y_name': "Belief",
    }
    env = VmasEnv(scenario=Scenario(), num_envs=1, device="cpu", **params)
    env.reset()
    st.session_state.env = env
    st.session_state.step = 0

col1, col2 = st.columns(2)
if col2.button("Initialize / Reset"): reset_simulation()
run_simulation = col2.toggle("Run Simulation", value=False)
placeholder = col1.empty()

def render_heatmap(env, fig, ax, target_circle_radius):
    render_env_frame(env, ax)
    
    # Draw circle around targets
    if hasattr(env, '_env'):
         raw_env = env._env
    else:
         raw_env = env
    
    for agent in raw_env.world.agents:
        if "target" in agent.name or isinstance(agent, TargetAgent):
            pos = agent.state.pos[0].detach().cpu().numpy()
            circle = plt.Circle((pos[0], pos[1]), target_circle_radius, color='red', fill=False, linestyle='--', alpha=0.5)
            ax.add_patch(circle)

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
            
            img = render_heatmap(env, fig, ax, target_circle_radius)
            placeholder.image(img, caption=f"Step {st.session_state.step}")
            time.sleep(1/fps)
    else:
        img = render_heatmap(env, fig, ax, target_circle_radius)
        placeholder.image(img, caption=f"Paused - Step {st.session_state.step}")
    plt.close(fig)
else:
    st.info("Initialize to start.")
