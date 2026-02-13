import pytest
import os
import numpy as np
from unittest.mock import MagicMock
from abm.utils import ExperimentLogger

# Ensure the visuals directory exists
VISUALS_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(VISUALS_DIR, exist_ok=True)

def generate_ternary_data(strategy='balanced', resolution=20):
    """Generates synthetic data for a ternary plot."""
    data = []
    step = 1.0 / resolution
    for p in np.arange(0, 1.0 + step, step):
        for b in np.arange(0, 1.0 + step - p, step):
            n = 1.0 - p - b
            # Ensure they sum to roughly 1 (floating point issues)
            if abs(p + b + n - 1.0) > 1e-5:
                continue
            
            if strategy == 'balanced':
                # High score if balanced, low if one dominates
                score = 1.0 - (abs(p - 1/3) + abs(b - 1/3) + abs(n - 1/3))
            elif strategy == 'priv':
                score = p
            elif strategy == 'bel':
                score = b
            elif strategy == 'none':
                score = n
            else:
                score = 0.0

            data.append({
                'priv': p,
                'bel': b,
                'none': n,
                'score': score
            })
    return data

@pytest.mark.parametrize("strategy", ["balanced", "priv", "bel", "none"])
def test_ternary_plot_generation(strategy):
    """
    Test that the ternary plot generation works without errors and saves a file.
    """
    # 1. Generate Data
    data = generate_ternary_data(strategy=strategy)
    
    # 2. Setup Mock Config and Logger
    cfg = MagicMock()
    cfg.project_name = "test_project"
    cfg.run_name = "test_run"
    
    # We set save_fig_locally=True so it attempts to save
    logger = ExperimentLogger(use_wandb=False, cfg=cfg, save_fig_locally=True)
    
    # 3. Call plotting function
    # The default behavior of log_ternary_plot is to save to "ternary_exploration.png" in the CWD.
    # We want to intercept this or move it.
    # Since we can't easily change the save path in the method without changing code, 
    # we will let it save to CWD and then move it.
    
    try:
        logger.log_ternary_plot(data, resolution=20, midpoint=0.5)
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")
        
    # 4. Verify file exists and move to visuals
    expected_file = "ternary_exploration.png"
    assert os.path.exists(expected_file), "Plot file was not created"
    
    target_name = f"ternary_test_{strategy}.png"
    target_path = os.path.join(VISUALS_DIR, target_name)
    
    # Clean up previous run if exists
    if os.path.exists(target_path):
        os.remove(target_path)
        
    os.rename(expected_file, target_path)
    
    assert os.path.exists(target_path), "Failed to move plot file to visuals directory"

if __name__ == "__main__":
    # Allow running this file directly
    pytest.main([__file__])
