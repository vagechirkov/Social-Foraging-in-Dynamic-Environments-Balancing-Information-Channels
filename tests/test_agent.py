import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from abm.agent import invert_2x2_matrix, ForagingAgent, AgentObservations, TargetAgent
from abm.model import Scenario
from vmas import make_env
from vmas.simulator.core import World, Sphere

# --- Fixtures ---

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def world(device):
    batch_dim = 2
    return World(batch_dim=batch_dim, device=device, x_semidim=10, y_semidim=10)

@pytest.fixture
def agent(device, world):
    batch_dim = 2
    n_targets = 2
    a = ForagingAgent(
        batch_dim=batch_dim,
        device=device,
        n_targets=n_targets,
        name="test_agent",
        shape=Sphere(radius=0.05)
    )
    world.add_agent(a)
    return a

@pytest.fixture
def other_agents(device, world):
    batch_dim = 2
    n_targets = 2
    agents = []
    for i in range(3):
        a = ForagingAgent(
            batch_dim=batch_dim,
            device=device,
            n_targets=n_targets,
            name=f"neighbor_{i}",
            shape=Sphere(radius=0.05)
        )
        world.add_agent(a)
        agents.append(a)
    
    # Set positions for reasonable distances
    agents[0].state.pos = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device=device)
    agents[1].state.pos = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=device)
    agents[2].state.pos = torch.tensor([[0.0, 1.0], [0.0, 1.0]], device=device)
    return agents

@pytest.fixture
def targets(device, world):
    batch_dim = 2
    t1 = TargetAgent(batch_dim=batch_dim, device=device, name="target_1", shape=Sphere(radius=0.05))
    t2 = TargetAgent(batch_dim=batch_dim, device=device, name="target_2", shape=Sphere(radius=0.05))
    world.add_agent(t1)
    world.add_agent(t2)
    t1.state.pos = torch.tensor([[10.0, 10.0], [-10.0, -10.0]], device=device)
    t2.state.pos = torch.tensor([[20.0, 0.0], [0.0, 20.0]], device=device)
    return [t1, t2]


# --- Visualization Helper ---

def plot_ellipse(ax, mean, cov, color, label, alpha=0.3):
    """Plots a 2D covariance ellipse."""
    m = mean.detach().cpu().numpy()
    c = cov.detach().cpu().numpy()
    
    # Eigen decomposition
    vals, vecs = np.linalg.eigh(c)
    # 2 standard deviations
    width, height = 2 * 2 * np.sqrt(np.maximum(vals, 1e-9))
    angle = np.degrees(np.atan2(vecs[1, 0], vecs[0, 0]))
    
    ell = patches.Ellipse(
        xy=m, width=width, height=height, angle=angle,
        edgecolor=color, facecolor=color, alpha=alpha, label=label, lw=2
    )
    ax.add_patch(ell)
    ax.plot(m[0], m[1], 'x', color=color)


# --- Utility Tests ---

def test_invert_2x2_matrix():
    # case 1: Identity
    mat = torch.eye(2).unsqueeze(0)
    inv = invert_2x2_matrix(mat)
    assert torch.allclose(inv, mat)

    # case 2: Scaling
    mat = torch.eye(2).unsqueeze(0) * 2
    inv = invert_2x2_matrix(mat)
    expected = torch.eye(2).unsqueeze(0) * 0.5
    assert torch.allclose(inv, expected)

    # case 3: Arbitrary invertible
    mat = torch.tensor([[[4.0, 7.0], [2.0, 6.0]]])
    inv = invert_2x2_matrix(mat)
    expected = torch.tensor([[[0.6, -0.7], [-0.2, 0.4]]])
    assert torch.allclose(inv, expected, atol=1e-5)

    # case 4: Singular (or close to)
    # The function adds epsilon if determinant is small.
    # [[1, 2], [2, 4]] -> det = 0
    mat = torch.tensor([[[1.0, 2.0], [2.0, 4.0]]])
    inv = invert_2x2_matrix(mat)
    # Determinant should be replaced by eps (1e-6)
    # Det = 0 -> 1e-6
    # Inv = [[4, -2], [-2, 1]] * 1e6
    expected_det_inv = 1.0 / 1e-6
    expected = torch.tensor([[[4.0, -2.0], [-2.0, 1.0]]]) * expected_det_inv
    assert torch.allclose(inv, expected)


# --- Core Logic Tests ---

def test_update_belief_manual_calculation(agent):
    """
    Verify Kalman Filter update manually for one step.
    """
    device = agent.device
    idx = 0 # Test first batch element
    target_idx = 0

    # 1. Setup Prior Belief (Sigma_t, mu_t)
    # Let's say we believe target is at [0, 0] with some uncertainty
    agent.belief_target_pos[:] = 0.0
    initial_cov_val = 2.0
    # Use direct assignment or clone() to avoid expansion errors
    agent.belief_target_covariance = torch.eye(2, device=device).expand_as(agent.belief_target_covariance).clone() * initial_cov_val

    # 2. Setup Observation (z_k, R_k)
    # Observation says target is at [1, 1] with lower uncertainty
    obs_pos = torch.tensor([1.0, 1.0], device=device)
    obs_cov_val = 1.0
    obs_cov = torch.eye(2, device=device) * obs_cov_val

    # Set observation in agent
    agent.obs_target_pos[idx, target_idx] = obs_pos
    agent.obs_target_covariance[idx, target_idx] = obs_cov
    
    # Enable update for this agent/target
    agent.obs_validity_mask[idx] = True
    # Mock action to ensure channel != 4 (None)
    agent.action.u = torch.zeros(agent.batch_dim, 1, device=device) # Channel 0

    # 3. Manual Calculation
    # Sigma_t = 2I
    # R_k = 1I
    # S = Sigma_t + R_k = 3I
    # K = Sigma_t @ S^-1 = 2I @ (1/3)I = (2/3)I
    
    # mu_t = [0, 0]
    # z_k = [1, 1]
    # y = z_k - mu_t = [1, 1]
    
    # correction = K @ y = (2/3) * [1, 1] = [0.666..., 0.666...]
    # mu_new = mu_t + correction = [0.666..., 0.666...]
    expected_mu = torch.tensor([2.0/3.0, 2.0/3.0], device=device)

    # Sigma_new = (I - K) @ Sigma_t = (I - 2/3 I) @ 2I = (1/3)I @ 2I = (2/3)I
    expected_cov = torch.eye(2, device=device) * (2.0/3.0)
    
    # Store priors for visualization
    prior_mu = agent.belief_target_pos[idx, target_idx].clone()
    prior_cov = agent.belief_target_covariance[idx, target_idx].clone()

    # 4. Run Update
    from abm.agent import update_belief
    update_belief(agent)

    # 5. Assertions
    updated_mu = agent.belief_target_pos[idx, target_idx]
    updated_cov = agent.belief_target_covariance[idx, target_idx]

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_ellipse(ax, prior_mu, prior_cov, 'red', 'Prior Belief')
    plot_ellipse(ax, obs_pos, obs_cov, 'green', 'Observation')
    plot_ellipse(ax, updated_mu, updated_cov, 'blue', 'Posterior (Updated)')
    
    ax.set_title("Kalman Filter Update: Spatial Verification")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    os.makedirs("tests/visuals", exist_ok=True)
    plt.savefig("tests/visuals/kf_update_verification.png")
    plt.close(fig)

    assert torch.allclose(updated_mu, expected_mu, atol=1e-5), \
        f"Expected mean {expected_mu}, got {updated_mu}"
    assert torch.allclose(updated_cov, expected_cov, atol=1e-5), \
        f"Expected cov {expected_cov}, got {updated_cov}"


def test_consensus_manual_calculation(agent, other_agents):
    """
    Verify Consensus (Federated Kalman Filter) manually.
    """
    from abm.agent import AgentObservations
    device = agent.device
    idx = 0
    target_idx = 0

    # Setup Agent (Receiver)
    agent.state.pos[:] = 0.0 # Origin
    # Set self belief to something differing to allow check of "selectivity" if needed
    # But `others_consensus` returns the aggregated observation, it doesn't do the update itself.
    # The update happens later in `update_belief`.
    # Here we check the return values of `others_consensus`.

    # Setup Neighbors (Senders)
    # Neighbor 1: at [0, 1] (dist 1). Belief: [10, 10], Cov: 1*I
    n1 = other_agents[0]
    n1.state.pos[idx] = torch.tensor([0.0, 1.0], device=device)
    n1.belief_target_pos[idx, target_idx] = torch.tensor([10.0, 10.0], device=device)
    n1.belief_target_covariance[idx, target_idx] = torch.eye(2, device=device) * 1.0
    n1.belief_target_qual_var[idx, target_idx] = 1.0

    # Neighbor 2: at [1, 0] (dist 1). Belief: [20, 20], Cov: 1*I
    n2 = other_agents[1]
    n2.state.pos[idx] = torch.tensor([1.0, 0.0], device=device)
    n2.belief_target_pos[idx, target_idx] = torch.tensor([20.0, 20.0], device=device)
    n2.belief_target_covariance[idx, target_idx] = torch.eye(2, device=device) * 1.0
    n2.belief_target_qual_var[idx, target_idx] = 1.0

    subset_neighbors = [n1, n2]

    # Transmission Noise Parameters
    agent.base_sigma_trans = 0.0
    agent.dist_noise_scale_soc = 1.0 # Sigma_trans = 0 + 1 * dist
    # Distances are 1.0 for both.
    # Sigma_trans = 1.0. Var_trans = 1.0.

    # Manual Calculation
    # Neighbor 1 "Message":
    # Cov_1_noisy = Cov_1 + Var_trans*I = 1*I + 1*I = 2*I
    # Prec_1 = (2*I)^-1 = 0.5*I
    # Weighted_Mu_1 = 0.5*I @ [10, 10] = [5, 5]

    # Neighbor 2 "Message":
    # Cov_2_noisy = Cov_2 + Var_trans*I = 2*I
    # Prec_2 = 0.5*I
    # Weighted_Mu_2 = 0.5*I @ [20, 20] = [10, 10]

    # Aggregation
    # Sum_Prec = 0.5*I + 0.5*I = 1.0*I
    # Agg_Cov = (Sum_Prec)^-1 = 1.0*I
    
    # Sum_Weighted_Mu = [5, 5] + [10, 10] = [15, 15]
    # Agg_Mu = Agg_Cov @ Sum_Weighted_Mu = 1*I @ [15, 15] = [15, 15]

    expected_mu = torch.tensor([15.0, 15.0], device=device)
    expected_cov = torch.eye(2, device=device)

    # Run Consensus
    indices = torch.tensor([idx], device=device)
    # mask is not used for consensus in the same way (it aggregates ALL),
    # but the signature requires it.
    mask = torch.zeros(agent.batch_dim, device=device) 
    
    mu_res, cov_res, qual_res, qual_var_res = AgentObservations.others_consensus(
        agent, subset_neighbors, mask, indices=indices
    )

    # Check Result for idx 0, target 0
    res_mu_0 = mu_res[0, target_idx]
    res_cov_0 = cov_res[0, target_idx]

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot noisy neighbor beliefs
    # In others_consensus: noise is added: Cov_noisy = Cov + Var_trans*I
    var_trans = 1.0
    plot_ellipse(ax, n1.belief_target_pos[0,0], n1.belief_target_covariance[0,0] + torch.eye(2)*var_trans, 
                 'orange', 'Neighbor 1 (Noisy Message)', alpha=0.1)
    plot_ellipse(ax, n2.belief_target_pos[0,0], n2.belief_target_covariance[0,0] + torch.eye(2)*var_trans, 
                 'purple', 'Neighbor 2 (Noisy Message)', alpha=0.1)
    
    # Plot result
    plot_ellipse(ax, res_mu_0, res_cov_0, 'blue', 'Consensus Result (Aggregated)', alpha=0.5)
    
    ax.set_title("Consensus Mechanism: Federated Kalman Filter Verification")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    os.makedirs("tests/visuals", exist_ok=True)
    plt.savefig("tests/visuals/consensus_verification.png")
    plt.close(fig)

    assert torch.allclose(res_mu_0, expected_mu, atol=1e-5), \
        f"Expected Mean {expected_mu}, Got {res_mu_0}"
    assert torch.allclose(res_cov_0, expected_cov, atol=1e-5), \
        f"Expected Cov {expected_cov}, Got {res_cov_0}"


# --- Observation Tests ---

def test_private_observation_shape(agent, targets):
    indices = torch.arange(agent.batch_dim)
    p, c, q, qv = AgentObservations.private(agent, targets, indices)
    
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, t in enumerate(targets):
        t_pos = t.state.pos[0].detach().cpu().numpy()
        obs_pos = p[0, i].detach().cpu().numpy()
        obs_cov = c[0, i]
        
        ax.plot(t_pos[0], t_pos[1], 'r*', markersize=12, label=f'True Target {i}')
        plot_ellipse(ax, p[0, i], obs_cov, 'green', f'Obs Target {i}', alpha=0.2)
    
    ax.plot(agent.state.pos[0, 0].item(), agent.state.pos[0, 1].item(), 'bo', label='Agent')
    ax.set_title("Private Sensing Verification")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.savefig("tests/visuals/private_sensing_verification.png")
    plt.close(fig)

    assert p.shape == (agent.batch_dim, len(targets), 2)
    assert c.shape == (agent.batch_dim, len(targets), 2, 2)
    assert q.shape == (agent.batch_dim, len(targets))
    assert qv.shape == (agent.batch_dim, len(targets))

def test_others_belief_shape(agent, other_agents):
    # Setup correct mock mask
    indices = torch.arange(agent.batch_dim)
    mask = torch.zeros(agent.batch_dim, dtype=torch.long) # Select neighbor 0

    p, c, q, qv = AgentObservations.others_belief(agent, other_agents, mask, indices)
    
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 8))
    neighbor = other_agents[0]
    
    # Plot neighbor's true belief for target 0
    plot_ellipse(ax, neighbor.belief_target_pos[0, 0], neighbor.belief_target_covariance[0, 0], 
                 'orange', 'Neighbor True Belief', alpha=0.3)
    
    # Plot what agent received (should be noisier)
    plot_ellipse(ax, p[0, 0], c[0, 0], 'blue', 'Agent Received Message', alpha=0.2)
    
    ax.plot(agent.state.pos[0, 0].item(), agent.state.pos[0, 1].item(), 'bo', label='Agent')
    ax.plot(neighbor.state.pos[0, 0].item(), neighbor.state.pos[0, 1].item(), 'go', label='Neighbor')
    
    ax.set_title("Belief Transfer Verification")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.savefig("tests/visuals/belief_transfer_verification.png")
    plt.close(fig)

    n_targets = agent.n_targets
    assert p.shape == (agent.batch_dim, n_targets, 2)
    assert c.shape == (agent.batch_dim, n_targets, 2, 2)

def test_others_heading_shape(agent, other_agents):
    indices = torch.arange(agent.batch_dim)
    mask = torch.zeros(agent.batch_dim, dtype=torch.long)
    
    # Needs valid velocity
    for a in other_agents:
        a.state.vel = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=a.device)

    p, c, q, qv = AgentObservations.others_heading(agent, other_agents, mask, indices)
    
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 8))
    neighbor = other_agents[0]
    
    # Plot agent and neighbor
    ax.plot(agent.state.pos[0, 0].item(), agent.state.pos[0, 1].item(), 'bo', label='Agent')
    ax.plot(neighbor.state.pos[0, 0].item(), neighbor.state.pos[0, 1].item(), 'go', label='Neighbor')
    
    # Plot neighbor's velocity vector
    vel = neighbor.state.vel[0].detach().cpu().numpy()
    ax.arrow(neighbor.state.pos[0, 0].item(), neighbor.state.pos[0, 1].item(), 
             vel[0]*0.5, vel[1]*0.5, head_width=0.05, head_length=0.1, fc='k', ec='k', label='Neighbor Velocity')
    
    # Plot inferred targets (heading heuristic projects along velocity)
    # Filter for non-huge variance (the chosen target)
    p_best = p[0, (c[0, :, 0, 0] < 1e5)]
    c_best = c[0, (c[0, :, 0, 0] < 1e5)]
    
    if p_best.numel() > 0:
        plot_ellipse(ax, p_best[0], c_best[0], 'orange', 'Inferred Target (Heading)', alpha=0.5)

    ax.set_title("Heading Heuristic Verification")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.savefig("tests/visuals/heading_heuristic_verification.png")
    plt.close(fig)

    n_targets = agent.n_targets
    assert p.shape == (agent.batch_dim, n_targets, 2)

def test_others_location_shape(agent, other_agents):
    indices = torch.arange(agent.batch_dim)
    mask = torch.zeros(agent.batch_dim, dtype=torch.long)

    p, c, q, qv = AgentObservations.others_location(agent, other_agents, mask, indices)
    
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 8))
    neighbor = other_agents[0]
    
    ax.plot(agent.state.pos[0, 0].item(), agent.state.pos[0, 1].item(), 'bo', label='Agent')
    ax.plot(neighbor.state.pos[0, 0].item(), neighbor.state.pos[0, 1].item(), 'go', label='Neighbor')
    
    # This heuristic assumes the target is EXACTLY where the neighbor is
    p_best = p[0, (c[0, :, 0, 0] < 1e5)]
    c_best = c[0, (c[0, :, 0, 0] < 1e5)]
    
    if p_best.numel() > 0:
        plot_ellipse(ax, p_best[0], c_best[0], 'purple', 'Inferred Target (Location)', alpha=0.5)

    ax.set_title("Positional Proxy Verification")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.savefig("tests/visuals/positional_proxy_verification.png")
    plt.close(fig)

    n_targets = agent.n_targets
    assert p.shape == (agent.batch_dim, n_targets, 2)

def test_target_levy_movement(device):
    """
    Verify the Lévy flight movement pattern: CRW + Periodic Relocation.
    """
    relocation_interval = 10
    scenario = Scenario()
    env = make_env(
        scenario=scenario,
        num_envs=1,
        device=device,
        seed=0,
        n_agents=1,
        n_targets=1,
        target_movement_pattern="levy",
        relocation_interval=relocation_interval,
        target_speed=1.0,
    )
    
    env.reset()
    target = [a for a in env.world.agents if "target" in a.name][0]
    target.time_since_last_relocation.fill_(0)
    
    positions = []
    relocation_steps = []
    
    total_steps = 30
    for i in range(total_steps):
        old_pos = target.state.pos.clone()
        env.step(torch.zeros(1, 1, 1)) 
        new_pos = target.state.pos.clone()
        dist = torch.norm(old_pos - new_pos).item()
        
        positions.append(new_pos[0].detach().cpu().numpy())
        
        # In our implementation, relocation happens when time_since >= interval
        # After step 10, time_since was 10, so it relocated.
        # So at step i=10 (the 11th step), it should have relocated.
        if i == relocation_interval or i == (2 * relocation_interval + 1):
            assert dist > 0.1, f"Should have relocated at step {i}, but dist was {dist}"
            relocation_steps.append(i)
        elif i > 0:
            assert dist < 0.1, f"Should be CRW at step {i}, but dist was {dist}"

    # --- Visualization ---
    positions = np.array(positions)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the full trajectory
    ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.3, label='Trajectory')
    
    # Plot segments (to highlight jumps)
    last_idx = 0
    for r_idx in relocation_steps:
        # Segment before jump
        ax.plot(positions[last_idx:r_idx, 0], positions[last_idx:r_idx, 1], 'g-', lw=2)
        # The jump itself
        ax.arrow(positions[r_idx-1, 0], positions[r_idx-1, 1], 
                 positions[r_idx, 0] - positions[r_idx-1, 0], 
                 positions[r_idx, 1] - positions[r_idx-1, 1],
                 color='red', linestyle='--', head_width=0.2, length_includes_head=True, label='Relocation Jump' if last_idx == 0 else "")
        last_idx = r_idx
    
    # Final segment
    ax.plot(positions[last_idx:, 0], positions[last_idx:, 1], 'g-', lw=2, label='CRW Segments')
    
    # Start and End
    ax.plot(positions[0, 0], positions[0, 1], 'ko', label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'kx', label='End')

    ax.set_title(f"Target Lévy Flight Trajectory (Relocation Interval: {relocation_interval})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    os.makedirs("tests/visuals", exist_ok=True)
    plt.savefig("tests/visuals/levy_flight_verification.png")
    plt.close(fig)

    print("Target Lévy flight verification successful! Plot saved to tests/visuals/levy_flight_verification.png")
