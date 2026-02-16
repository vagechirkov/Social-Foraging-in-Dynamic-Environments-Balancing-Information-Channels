import pytest
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
import os
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Mock modules if missing
try:
    import torchrl
except ImportError:
    sys.modules['torchrl'] = MagicMock()
    sys.modules['torchrl.envs'] = MagicMock()
try:
    import vmas
except ImportError:
    sys.modules['vmas'] = MagicMock()
try:
    import wandb
except ImportError:
    sys.modules['wandb'] = MagicMock()

from abm.info_channels_ea import run_experiment

@patch('abm.info_channels_ea.VmasEvaluator')
@patch('abm.info_channels_ea.GeneticAlgorithm')
@patch('abm.info_channels_ea.ExperimentLogger')
def test_environment_switching(MockLogger, MockGA, MockEvaluator):
    # Setup Mocks
    mock_evaluator_instance = MockEvaluator.return_value
    # minimal tensor output
    import torch
    mock_evaluator_instance.total_pop = 10
    mock_evaluator_instance.evaluate.return_value = torch.zeros(10, 5) 
    
    # Test Config
    cfg = OmegaConf.create({
        "seed": 42,
        "use_gpu": False,
        "n_agents": 5,
        "project_name": "test",
        "run_name": "test_run",
        "max_steps": 10,
        "log_freq": 1,
        "targets_quality": "HM",
        "use_wandb": False,
        "n_envs": 10,
        
        "environment": {
            "mode": "dynamic",
            "static_category": "baseline"
        },
        "evolution": {
            "generations": 10,
            "replicates": 2,
            "switch_interval": 2 
        },
        "environments": {
            "categories": {
                "baseline": {"target_speed": 0.1},
                "high_cost": {"target_speed": 1.0}
            }
        },
        # Add missing genetic algorithm params that might be needed
        "cxpb": 0.5,
        "mutpb": 0.2,
        "sigma": 0.1
    })
    
    # Run
    run_experiment(cfg)
    
    # Verification
    # We expect init_env to be called at start + (generations // switch_interval) times
    # 10 gens, switch every 2 -> 0, 2, 4, 6, 8.
    # Initial call: env=baseline (idx 0)
    # Gen 0: stage 0. env=baseline. No switch.
    # Gen 2: stage 1. env=high_cost. SWITCH.
    # Gen 4: stage 2 (wrap to 0). env=baseline. SWITCH.
    # ...
    
    # Total init_env calls should be >= 2 (initial + at least one switch)
    print(f"init_env call count: {mock_evaluator_instance.init_env.call_count}")
    
    assert mock_evaluator_instance.init_env.call_count >= 2

