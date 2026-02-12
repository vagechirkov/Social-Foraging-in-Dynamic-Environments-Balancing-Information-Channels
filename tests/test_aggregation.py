import torch
import sys
import os

# Add the project root to sys.path to import abm.agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abm.agent import ForagingAgent, AgentObservations, TargetAgent
from vmas.simulator.core import World

def test_consensus_aggregation():
    device = torch.device("cpu")
    batch_dim = 1
    n_targets = 1
    
    world = World(batch_dim=batch_dim, device=device, x_semidim=10.0, y_semidim=10.0)
    
    # Create target
    target = TargetAgent(batch_dim=batch_dim, device=device, name="target")
    world.add_agent(target)
    target.state.pos = torch.tensor([[5.0, 5.0]])
    targets = [target]
    
    # Create main agent
    agent = ForagingAgent(name="agent", batch_dim=batch_dim, device=device, n_targets=n_targets)
    world.add_agent(agent)
    agent.state.pos = torch.zeros(batch_dim, 2)
    # Agent belief is at (0,0)
    agent.belief_target_pos = torch.zeros(batch_dim, n_targets, 2)
    agent.belief_target_covariance = torch.eye(2).view(1, 1, 2, 2) * 1.0
    
    # Neighbor A: Close to target but very low certainty
    # Confidence: Low, Novelty: High
    neighbor_a = ForagingAgent(name="neighbor_a", batch_dim=batch_dim, device=device, n_targets=n_targets)
    world.add_agent(neighbor_a)
    neighbor_a.state.pos = torch.tensor([[1.0, 1.0]])
    neighbor_a.belief_target_pos = torch.tensor([[[4.8, 4.8]]]) # Close to (5,5)
    neighbor_a.belief_target_covariance = torch.eye(2).view(1, 1, 2, 2) * 10.0
    
    # Neighbor B: Far from target but very high certainty
    # Confidence: High, Novelty: High
    neighbor_b = ForagingAgent(name="neighbor_b", batch_dim=batch_dim, device=device, n_targets=n_targets)
    world.add_agent(neighbor_b)
    neighbor_b.state.pos = torch.tensor([[-1.0, -1.0]])
    neighbor_b.belief_target_pos = torch.tensor([[[2.0, 2.0]]]) # Far from (5,5)
    neighbor_b.belief_target_covariance = torch.eye(2).view(1, 1, 2, 2) * 0.01
    
    other_agents = [neighbor_a, neighbor_b]
    indices = torch.tensor([0])
    
    # Test "most_useful" (Confidence * Novelty)
    # Should pick Neighbor B because it's much more certain
    print("Testing 'most_useful' aggregation (Confidence * Novelty)...")
    mu_useful, _, _, _ = AgentObservations.others_consensus(
        agent, other_agents, None, targets=targets, indices=indices, aggregation_method="most_useful"
    )
    print(f"Selected Mu (most_useful): {mu_useful.numpy()}")
    
    # Test "oracle" (Closest to true target)
    # Should pick Neighbor A because it's closest to (5,5)
    print("\nTesting 'oracle' aggregation (Closest to true target)...")
    mu_oracle, _, _, _ = AgentObservations.others_consensus(
        agent, other_agents, None, targets=targets, indices=indices, aggregation_method="oracle"
    )
    print(f"Selected Mu (oracle): {mu_oracle.numpy()}")
    
    # Verification
    dist_useful_to_b = torch.norm(mu_useful - neighbor_b.belief_target_pos[0])
    dist_oracle_to_a = torch.norm(mu_oracle - neighbor_a.belief_target_pos[0])
    
    success = True
    if dist_useful_to_b < 0.1:
        print("SUCCESS: 'most_useful' selected Neighbor B (High Confidence).")
    else:
        print("FAILURE: 'most_useful' did not select Neighbor B.")
        success = False
        
    if dist_oracle_to_a < 0.1:
        print("SUCCESS: 'oracle' selected Neighbor A (Closest to true target).")
    else:
        print("FAILURE: 'oracle' did not select Neighbor A.")
        success = False
        
    if success:
        print("\nALL TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED!")

if __name__ == "__main__":
    test_consensus_aggregation()
