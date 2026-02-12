
import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from abm.agent import ForagingAgent, AgentObservations, update_belief, add_process_noise_to_belief
from vmas.simulator.core import World, Sphere

@pytest.fixture
def device():
    return torch.device("cpu")

def test_bias_correction_consensus(device):
    """
    Verifies that two agents with opposing biases (+2, -2) converge to the true target 
    position (0) when using the consensus channel, provided they are close enough 
    and the selectivity threshold allows interaction.
    """
    batch_dim = 1
    n_targets = 1
    world = World(batch_dim=batch_dim, device=device, x_semidim=20, y_semidim=20)

    # Agent 1: Bias +2.0
    a1 = ForagingAgent(
        batch_dim=batch_dim,
        device=device,
        n_targets=n_targets,
        name="bias_plus",
        shape=Sphere(radius=0.5),
        bias_magnitude=0.0,
        base_noise=0.1,
        dist_noise_scale_priv=0.0,
        dist_noise_scale_soc=0.1,
        # Threshold should be high enough (default is now 100.0)
        social_info_aggregation="average"
    )
    a1.sensor_bias = torch.tensor([[2.0, 0.0]], device=device)
    world.add_agent(a1)

    # Agent 2: Bias -2.0
    a2 = ForagingAgent(
        batch_dim=batch_dim,
        device=device,
        n_targets=n_targets,
        name="bias_minus",
        shape=Sphere(radius=0.5),
        bias_magnitude=0.0,
        base_noise=0.1,
        dist_noise_scale_priv=0.0,
        dist_noise_scale_soc=0.1,
        social_info_aggregation="average"
    )
    a2.sensor_bias = torch.tensor([[-2.0, 0.0]], device=device)
    world.add_agent(a2)

    # Dummy Target at [0, 0]
    class DummyTarget:
        def __init__(self, pos):
            self.state = type('obj', (object,), {'pos': pos})
            self.quality = torch.tensor([1.0], device=device)
    
    target = DummyTarget(torch.tensor([[0.0, 0.0]], device=device))
    targets = [target]

    # Setup Agents
    a1.state.pos[:] = torch.tensor([10.0, 0.0])
    a2.state.pos[:] = torch.tensor([10.0, 1.0]) # Close proximity

    # Init Beliefs to 0 with high uncertainy
    for a in [a1, a2]:
        a.belief_target_pos[:] = 0.0
        a.belief_target_covariance[:] = torch.eye(2) * 10.0

    # Phase 1: Private Sensing (Bias Induction)
    # Run for 20 steps to let them drift to their biases
    for _ in range(20):
        a1.action.u = torch.zeros(batch_dim, 1, device=device)
        a2.action.u = torch.zeros(batch_dim, 1, device=device)
        
        # Observe
        p, c, q, qv = AgentObservations.private(a1, targets)
        a1.obs_target_pos = p
        a1.obs_target_covariance = c
        a1.obs_target_qual = q
        a1.obs_target_qual_var = qv
        a1.obs_validity_mask[:] = True
        
        p, c, q, qv = AgentObservations.private(a2, targets)
        a2.obs_target_pos = p
        a2.obs_target_covariance = c
        a2.obs_target_qual = q
        a2.obs_target_qual_var = qv
        a2.obs_validity_mask[:] = True
        
        update_belief(a1)
        update_belief(a2)
        add_process_noise_to_belief(a1, 0.1)
        add_process_noise_to_belief(a2, 0.1)

    # Assert biases are present
    bias1 = a1.belief_target_pos[0, 0, 0].item()
    bias2 = a2.belief_target_pos[0, 0, 0].item()
    # Expect roughly 2.0 and -2.0
    assert bias1 > 1.5, f"Agent 1 should be biased > 1.5, got {bias1}"
    assert bias2 < -1.5, f"Agent 2 should be biased < -1.5, got {bias2}"

    # Run for 50 steps
    history = []
    for _ in range(50):
        # A1 Listens to A2
        p, c, q, qv = AgentObservations.others_consensus(
            a1, [a2], torch.zeros(batch_dim, dtype=torch.long, device=device),
            targets=targets, aggregation_method="average"
        )
        gate_a1 = a1.obs_validity_mask[0].item()
        
        a1.obs_target_pos = p
        a1.obs_target_covariance = c
        a1.obs_validity_mask[:] = gate_a1

        # A2 Listens to A1
        p, c, q, qv = AgentObservations.others_consensus(
            a2, [a1], torch.zeros(batch_dim, dtype=torch.long, device=device),
            targets=targets, aggregation_method="average"
        )
        gate_a2 = a2.obs_validity_mask[0].item()
        
        a2.obs_target_pos = p
        a2.obs_target_covariance = c
        a2.obs_validity_mask[:] = gate_a2

        update_belief(a1)
        update_belief(a2)
        add_process_noise_to_belief(a1, 0.1)
        add_process_noise_to_belief(a2, 0.1)
        
        history.append((a1.belief_target_pos[0,0,0].item(), a2.belief_target_pos[0,0,0].item(), gate_a1, gate_a2))

    final_pos_a1 = history[-1][0]
    final_pos_a2 = history[-1][1]
    
    # Visualization (optional, for debug/artifacts)
    os.makedirs("tests/visuals", exist_ok=True)
    plt.figure()
    plt.plot([h[0] for h in history], label='A1 (+Bias)')
    plt.plot([h[1] for h in history], label='A2 (-Bias)')
    plt.axhline(0, color='k', linestyle='--', label='True Target')
    plt.legend()
    plt.title("Bias Correction Test")
    plt.savefig("tests/visuals/test_bias_correction.png")
    plt.close()

    # Check convergence
    # They should meet in the middle (0.0)
    # With symmetric noise, they should be very close to 0.0
    print(f"Final Positions: A1={final_pos_a1}, A2={final_pos_a2}")
    assert abs(final_pos_a1) < 0.2, f"A1 did not converge enough: {final_pos_a1}"
    assert abs(final_pos_a2) < 0.2, f"A2 did not converge enough: {final_pos_a2}"
    
    # Check that gates were open at least sometimes
    # Since we set default to 100.0, gates SHOULD be open most of the time
    gates_open = sum([h[2] and h[3] for h in history])
    assert gates_open > 5, "Gates did not open frequently enough to correct bias"
