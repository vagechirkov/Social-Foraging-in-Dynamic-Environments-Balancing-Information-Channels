import torch
import pytest
from vmas.simulator.core import World
from abm.agent import ForagingAgent, AgentObservations

@pytest.fixture
def test_setup():
    device = "cpu"
    batch_dim = 2
    n_targets = 3
    
    world = World(batch_dim=batch_dim, device=device)

    agent = ForagingAgent(
        batch_dim=batch_dim,
        device=device,
        n_targets=n_targets,
        base_noise=0.1,
        dist_noise_scale_priv=1.0,
        dist_noise_scale_soc=1.0,
        process_noise_scale=0.02,
        bias_magnitude=0.0,
        social_trans_scale=1.0,
        social_pos_scale=1.0,
        social_heading_scale=1.0,
        belief_selectivity_threshold=1.0,
        consensus_selectivity_threshold=1.0,
        social_info_aggregation="average",
        momentum=0.9,
        max_belief_uncertainty=20.0,
        cost_priv=0.1,
        cost_belief=0.1,
        cost_heading=0.1,
        cost_pos=0.1,
        cost_consensus=0.1,
        spot_radius=0.5,
        decision_making="sum",
        x_dim=1.0,
        y_dim=1.0,
        name="test_agent",
        collide=False,
        shape=None,
        action_size=1,
        max_speed=0.05,
        color=None,
        dynamics=None
    )
    world.add_agent(agent)

    # Set agent position to (0,0)
    agent.state.pos = torch.zeros(batch_dim, 2)
    
    # Set beliefs to be very certain
    agent.belief_target_pos = torch.zeros(batch_dim, n_targets, 2)
    agent.belief_target_covariance = torch.eye(2, device=device).reshape(1, 1, 2, 2).repeat(batch_dim, n_targets, 1, 1) * 0.0001
    
    class MockTarget:
        def __init__(self, pos, quality):
            self.state = type('obj', (object,), {'pos': pos})
            self.quality = quality

    targets = [
        MockTarget(torch.tensor([[10.0, 10.0], [10.0, 10.0]]), torch.tensor([1.0, 1.0])),
        MockTarget(torch.tensor([[20.0, 20.0], [20.0, 20.0]]), torch.tensor([0.5, 0.5])),
        MockTarget(torch.tensor([[30.0, 30.0], [30.0, 30.0]]), torch.tensor([2.0, 2.0]))
    ]
    
    return agent, targets

def test_multi_spotlight_count_one(test_setup):
    agent, targets = test_setup
    agent.n_private_samples = 1
    batch_dim = agent.batch_dim
    n_targets = agent.n_targets
    
    agent.obs_target_mask = torch.zeros(batch_dim, n_targets, dtype=torch.bool)
    AgentObservations.private_spotlight(agent, targets)
    
    # Check that exactly one target is sampled per agent in batch
    assert (agent.obs_target_mask.sum(dim=1) == 1).all(), "Exactly 1 target should be sampled when n_private_samples=1"

def test_multi_spotlight_count_all(test_setup):
    agent, targets = test_setup
    agent.n_private_samples = 3
    batch_dim = agent.batch_dim
    n_targets = agent.n_targets
    
    agent.obs_target_mask = torch.zeros(batch_dim, n_targets, dtype=torch.bool)
    AgentObservations.private_spotlight(agent, targets)
    
    # Check that all targets are sampled
    assert (agent.obs_target_mask.sum(dim=1) == 3).all(), "All 3 targets should be sampled when n_private_samples=3"

def test_multi_spotlight_spatial_exploration(test_setup):
    agent, targets = test_setup
    agent.p_spatial_explore = 1.0  # Force random look
    agent.n_private_samples = 1
    
    # Place Target 0 within the random scan range [-1, 1]
    targets[0].state.pos = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    
    # Reset mask
    agent.obs_target_mask = torch.zeros(agent.batch_dim, agent.n_targets, dtype=torch.bool)
    
    # Use a radius large enough to ensure a hit when looking in [-1, 1]
    p, c, q, qv = AgentObservations.private_spotlight(agent, targets, spot_radius=3.0) 
    
    assert agent.obs_target_mask[:, 0].all(), "Target 0 should be discovered during random scan"
    
def test_multi_spotlight_vectorized(test_setup):
    agent, targets = test_setup
    agent.n_private_samples = 3 # Force all
    batch_dim = agent.batch_dim
    n_targets = agent.n_targets
    spot_radius = 0.5
    
    # Set beliefs to match target positions to test HITS
    target_pos_stack = torch.stack([t.state.pos[0] for t in targets], dim=0) # (n_targets, 2)
    agent.belief_target_pos = target_pos_stack.unsqueeze(0).expand(batch_dim, n_targets, 2)

    # Reset mask
    agent.obs_target_mask = torch.zeros(batch_dim, n_targets, dtype=torch.bool)
    
    p, c, q, qv = AgentObservations.private_spotlight(agent, targets, spot_radius=spot_radius)
    
    # Assertions
    assert agent.obs_target_mask.all(), "All targets should be in the mask after spotlight"
    
    # All should be hits now
    assert (q[:, 0] == 1.0).all(), "Target 0 should be a Hit with quality 1.0"
    
    # Check variances for hits (should be much less than HUGE_VAR=1e4)
    assert (c[:, 0, 0, 0] < 1000.0).all(), "Target 0 (Hit) should have reasonable variance"
    assert (c[:, 1, 0, 0] < 1000.0).all(), "Target 1 (Hit) should have reasonable variance"
