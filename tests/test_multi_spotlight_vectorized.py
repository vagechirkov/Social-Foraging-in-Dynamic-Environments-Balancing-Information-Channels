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
        MockTarget(torch.tensor([[0.1, 0.1], [0.1, 0.1]]), torch.tensor([1.0, 1.0])),
        MockTarget(torch.tensor([[2.0, 2.0], [2.0, 2.0]]), torch.tensor([0.5, 0.5])),
        MockTarget(torch.tensor([[0.0, 0.0], [0.0, 0.0]]), torch.tensor([2.0, 2.0]))
    ]
    
    return agent, targets

def test_multi_spotlight_vectorized(test_setup):
    agent, targets = test_setup
    batch_dim = agent.batch_dim
    n_targets = agent.n_targets
    spot_radius = 0.5
    
    # Reset mask
    agent.obs_target_mask = torch.zeros(batch_dim, n_targets, dtype=torch.bool)
    
    p, c, q, qv = AgentObservations.private_spotlight(agent, targets, spot_radius=spot_radius)
    
    # Assertions
    assert agent.obs_target_mask.all(), "All targets should be in the mask after spotlight"
    
    # Target 0: (0.1, 0.1) vs belief (0,0) with low variance -> should hit
    assert (q[:, 0] == 1.0).all(), "Target 0 should be a Hit with quality 1.0"
    
    # Target 1: (2.0, 2.0) vs belief (0,0) -> should miss
    assert (q[:, 1] == 0.0).all(), "Target 1 should be a Miss with quality 0.0"
    
    # Target 2: (0.0, 0.0) vs belief (0,0) -> should hit
    assert (q[:, 2] == 2.0).all(), "Target 2 should be a Hit with quality 2.0"
    
    # Check variances
    # For Misses, position variance should be HUGE_VAR (1e4)
    # out_cov is (batch, n_targets, 2, 2)
    assert (c[:, 1, 0, 0] == 1e4).all(), "Target 1 (Miss) should have HUGE_VAR for position"
    
    # For Hits, position variance should be var_spot
    # var_spot = (sqrt(1e-2) + beta * dist)**2
    # dist for Target 2 is 0, so var_spot = 1e-2 = 0.01
    assert torch.allclose(c[:, 2, 0, 0], torch.tensor(0.01)), "Target 2 (Hit) should have TINY_VAR"
