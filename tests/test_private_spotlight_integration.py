
import torch
import pytest
import math
from vmas.simulator.core import World
from abm.agent import ForagingAgent, TargetAgent, AgentObservations, update_belief

@pytest.fixture
def setup_agents():
    device = "cpu"
    batch_dim = 1
    n_targets = 2
    
    # Create World
    world = World(batch_dim, device, dt=1.0)

    # Mock Agent
    agent = ForagingAgent(
        name="agent_0",
        collide=False,
        shape=None,
        action_size=1,
        max_speed=1.0, 
        color=None,
        device=device, 
        dynamics=None,
        u_range=1,
        batch_dim=batch_dim,
        n_targets=n_targets,
        spot_radius=0.5
    )
    world.add_agent(agent)
    
    # Mock Targets
    t1 = TargetAgent(name="t1", collide=False, shape=None, color=None, 
                     render_action=False, max_speed=0, action_script=None, 
                     action_size=1, batch_dim=batch_dim, device=device, dynamics=None,
                     quality=1.0)
    world.add_agent(t1)

    t2 = TargetAgent(name="t2", collide=False, shape=None, color=None, 
                     render_action=False, max_speed=0, action_script=None, 
                     action_size=1, batch_dim=batch_dim, device=device, dynamics=None,
                     quality=0.5)
    world.add_agent(t2)
    
    # Set mock states
    # T1 at (5,5)
    t1.state.pos = torch.tensor([[5.0, 5.0]], device=device)
    t1.quality = torch.tensor([1.0], device=device)
    
    # T2 at (-5,-5)
    t2.state.pos = torch.tensor([[-5.0, -5.0]], device=device)
    t2.quality = torch.tensor([0.5], device=device)
    
    targets = [t1, t2]
    
    # Initialize Agent Beliefs
    # Agent knows roughly where T1 is, but with uncertainty
    agent.belief_target_pos[0, 0] = torch.tensor([4.5, 4.5]) # Close to T1
    agent.belief_target_covariance[0, 0] = torch.eye(2) * 1.0 # Moderate uncertainty
    agent.belief_target_qual[0, 0] = 0.5 # Unsure quality
    agent.belief_target_qual_var[0, 0] = 1.0 # High quality uncertainty
    
    # Agent knows roughly where T2 is
    agent.belief_target_pos[0, 1] = torch.tensor([-5.0, -5.0]) # Exact T2
    agent.belief_target_covariance[0, 1] = torch.eye(2) * 1.0
    agent.belief_target_qual[0, 1] = 0.2
    agent.belief_target_qual_var[0, 1] = 1.0
    
    # Set Action to utilize Private Channel (0)
    # [:, 0] is the channel selection in action vector u
    agent.action.u = torch.zeros(batch_dim, 1, device=device) 
    # obs_validity_mask must be true for update to happen
    agent.obs_validity_mask = torch.tensor([True], device=device)
    
    return agent, targets

def test_hit_scenario_update(setup_agents):
    agent, targets = setup_agents
    
    # 1. Force Selection of Target 0 (T1) -> Hit
    # Let's make T1 the only viable candidate for selection to be sure
    agent.belief_target_qual[0, 1] = 0.0 # T2 quality to 0
    
    # Run spotlight
    # Note: private_spotlight updates agent.obs_target_mask in-place
    # Use large radius to GUARANTEE Hit for this test logic
    out_pos, out_cov, out_q, out_q_var = AgentObservations.private_spotlight(agent, targets, spot_radius=10.0)
    
    # Assign to agent observation buffers
    agent.obs_target_pos = out_pos
    agent.obs_target_covariance = out_cov
    agent.obs_target_qual = out_q
    agent.obs_target_qual_var = out_q_var
    
    # Validate Observation BEFORE Update
    # Target 0 should be Selected (Mask=True)
    # Target 1 should NOT be Selected (Mask=False)
    assert agent.obs_target_mask[0, 0] == True, "Target 0 should be selected"
    assert agent.obs_target_mask[0, 1] == False, "Target 1 should not be selected"
    
    # Run Belief Update
    old_mean = agent.belief_target_pos[0, 0].clone()
    old_cov = agent.belief_target_covariance[0, 0].clone()
    
    # T2 should remain strictly unchanged
    old_t2_mean = agent.belief_target_pos[0, 1].clone()
    old_t2_cov = agent.belief_target_covariance[0, 1].clone()
    
    update_belief(agent)
    
    new_mean = agent.belief_target_pos[0, 0]
    new_cov = agent.belief_target_covariance[0, 0]
    
    # Validation
    # 1. Covariance should decrease (Information Gain)
    assert torch.det(new_cov) < torch.det(old_cov), "Covariance should decrease after Hit"
    
    # 2. Mean should move towards True Position (5,5)
    # Old prediction (4.5, 4.5), True (5,5). New should be closer to (5,5)
    err_old = torch.norm(old_mean - torch.tensor([5.0, 5.0]))
    err_new = torch.norm(new_mean - torch.tensor([5.0, 5.0]))
    assert err_new < err_old, "Mean should move closer to truth"

    # 3. T2 unchanged
    assert torch.allclose(agent.belief_target_pos[0, 1], old_t2_mean), "T2 mean should be unchanged"
    assert torch.allclose(agent.belief_target_covariance[0, 1], old_t2_cov), "T2 cov should be unchanged"

def test_miss_scenario_update(setup_agents):
    agent, targets = setup_agents
    
    # Setup MISS for Target 0
    # Agent thinks T1 is at (0,0), but it is at (5,5)
    agent.belief_target_pos[0, 0] = torch.tensor([0.0, 0.0])
    agent.belief_target_covariance[0, 0] = torch.eye(2) * 0.1 # Confident but wrong
    
    # Ensure T1 is selected
    agent.belief_target_qual[0, 1] = 0.0 
    
    # Run Spotlight (Radius 0.5) -> Check (0,0) -> Miss
    out_pos, out_cov, out_q, out_q_var = AgentObservations.private_spotlight(agent, targets, spot_radius=0.5)
    
    agent.obs_target_pos = out_pos
    agent.obs_target_covariance = out_cov
    agent.obs_target_qual = out_q
    agent.obs_target_qual_var = out_q_var
    
    # Validate Mask
    # New Logic: Miss -> Mask is False -> No Update
    assert agent.obs_target_mask[0, 0] == False, "Target 0 should NOT be selected on Miss"
    assert agent.obs_target_mask[0, 1] == False, "Target 1 not selected"
    
    # Run Update
    old_pos_belief_mean = agent.belief_target_pos[0, 0].clone()
    old_pos_belief_cov = agent.belief_target_covariance[0, 0].clone()
    old_qual_belief = agent.belief_target_qual[0, 0].clone()
    
    update_belief(agent)
    
    # Validation
    # 1. Position Belief should NOT change (No update)
    assert torch.allclose(agent.belief_target_pos[0, 0], old_pos_belief_mean), "Position mean should not change on Miss"
    assert torch.allclose(agent.belief_target_covariance[0, 0], old_pos_belief_cov), "Position cov should not change on Miss"
    
    # 2. Quality Belief should NOT change (No update)
    assert torch.allclose(agent.belief_target_qual[0, 0], old_qual_belief), "Quality belief should not change on Miss"

def test_multi_target_isolation(setup_agents):
    agent, targets = setup_agents
    
    # T1 and T2 both viable.
    # We want to check that updating one does NOT touch the other.
    
    # Hardcode observation buffers: Say we observed T1 perfectly, T2 no info.
    dummy_pos = torch.zeros_like(agent.belief_target_pos) # (1, 2, 2)
    dummy_pos[0, 0] = torch.tensor([5.0, 5.0]) # True Pos T1
    
    dummy_cov = torch.zeros_like(agent.belief_target_covariance)
    HUGE = 1e6
    TINY = 1e-2
    dummy_cov[:, :] = torch.eye(2) * HUGE 
    dummy_cov[0, 0] = torch.eye(2) * TINY # Valid info for T1 only
    
    agent.obs_target_pos = dummy_pos
    agent.obs_target_covariance = dummy_cov
    
    # Set MASK: Only T1 is observed
    agent.obs_target_mask[:] = False
    agent.obs_target_mask[0, 0] = True
    
    # Backup T2 state
    old_t2_mean = agent.belief_target_pos[0, 1].clone()
    old_t2_cov = agent.belief_target_covariance[0, 1].clone()
    
    # Update
    update_belief(agent)
    
    # Check T1 updated
    # (Assuming prior was not perfect, it should change. We set it to 4.5, 4.5 in setup)
    assert not torch.allclose(agent.belief_target_pos[0, 0], torch.tensor([4.5, 4.5])), "T1 should update"
    
    # Check T2 UNCHANGED
    assert torch.allclose(agent.belief_target_pos[0, 1], old_t2_mean), "T2 Mean should be isolated"
    assert torch.allclose(agent.belief_target_covariance[0, 1], old_t2_cov), "T2 Cov should be isolated"
