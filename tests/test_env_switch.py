"""
Tests for the internal periodic environment switch mechanism.

Verifies that:
1. Scenario tracks current_t correctly.
2. Target qualities are swapped periodically at switch_time intervals.
3. The switch happens internally during the rollout (no external intervention needed).
"""

import pytest
import torch
from omegaconf import OmegaConf

from abm.agent import TargetAgent
from abm.utils import VmasEvaluator, GenePersistenceTransform, SimpleAgent, N_CHANNELS


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def cfg():
    """Minimal OmegaConf config."""
    return OmegaConf.create({
        "n_agents": 2,
        "n_targets": 2,
        "n_envs": 2,
        "max_steps": 20,
        "use_gpu": False,
        "project_name": "test",
        "run_name": "test_internal_switch",
        "use_wandb": False,
        "top_k": 1,
        "resolution": 2,
        "seed": 42,
        "target_qualities": "[1.0, 0.1]",
        "targets_quality": "HM",
        "target_speed": 0.0,
        "cost_priv": 0.0,
        "cost_belief": 0.0,
        "cost_consensus": 0.0,
        "base_noise": 0.1,
        "dist_noise_scale_priv": 0.05,
        "process_noise_scale": 0.05,
        "process_noise_scale_het_ratio": 0,
        "process_noise_scale_het_scale": 1,
        "belief_selectivity_threshold": 0.25,
        "consensus_selectivity_threshold": 1.0,
        "social_trans_scale": 0.01,
        "x_dim": 2,
        "y_dim": 2,
        "target_persistence": 20,
        "target_movement_pattern": "crw",
        "relocation_interval": 1000,
        "channel_y_name": "Belief",
        "bias_magnitude": 0.0,
        "spot_radius": 0.25,
        "decision_making": "greedy",
        "p_spatial_explore": 0.0,
        "env_switch": True,
        "switch_time": 5,
    })


@pytest.fixture
def evaluator_and_env(cfg, device, monkeypatch):
    monkeypatch.setattr(VmasEvaluator, "get_num_workers", staticmethod(lambda: 1))
    evaluator = VmasEvaluator(cfg, device)
    env = evaluator.init_env(env_transform=GenePersistenceTransform)
    return evaluator, env


@pytest.fixture
def islands(cfg):
    n_envs = cfg.n_envs
    n_agents = cfg.n_agents
    genes = [0.0] * N_CHANNELS
    return [[SimpleAgent(genes) for _ in range(n_agents)] for _ in range(n_envs)]


def _get_base_vmas_env(env):
    curr = env
    while hasattr(curr, "base_env"):
        curr = curr.base_env
    return curr


def _get_targets(env):
    base = _get_base_vmas_env(env)
    return [a for a in base._env.world.agents if isinstance(a, TargetAgent)]


class TestInternalPeriodicSwitch:

    def test_scenario_tracks_time(self, evaluator_and_env, islands):
        evaluator, env = evaluator_and_env
        base = _get_base_vmas_env(env)
        scenario = base._env.scenario

        assert scenario.current_t == 0
        
        evaluator.evaluate(env, islands, max_steps=3, reset=True)
        assert scenario.current_t == 3

    def test_periodic_swap_logic(self, evaluator_and_env, islands):
        """
        Verify that qualities swap every 'switch_time' steps.
        With switch_time=5, max_steps=12:
        t=0: [1.0, 0.1]
        t=5: swap -> [0.1, 1.0]
        t=10: swap -> [1.0, 0.1]
        """
        evaluator, env = evaluator_and_env
        base = _get_base_vmas_env(env)
        scenario = base._env.scenario
        targets = _get_targets(env)
        
        # Initial state
        q0_orig = targets[0].quality[0].item()
        q1_orig = targets[1].quality[0].item()
        assert q0_orig == pytest.approx(1.0)
        assert q1_orig == pytest.approx(0.1)

        # Step 1-4: No swap
        evaluator.evaluate(env, islands, max_steps=4, reset=True)
        assert targets[0].quality[0].item() == pytest.approx(q0_orig)
        assert targets[1].quality[0].item() == pytest.approx(q1_orig)

        # Step 5: First swap
        evaluator.evaluate(env, islands, max_steps=1, reset=False)
        assert targets[0].quality[0].item() == pytest.approx(q1_orig)
        assert targets[1].quality[0].item() == pytest.approx(q0_orig)

        # Step 6-9: Still swapped
        evaluator.evaluate(env, islands, max_steps=4, reset=False)
        assert targets[0].quality[0].item() == pytest.approx(q1_orig)
        assert targets[1].quality[0].item() == pytest.approx(q0_orig)

        # Step 10: Second swap (back to original)
        evaluator.evaluate(env, islands, max_steps=1, reset=False)
        assert targets[0].quality[0].item() == pytest.approx(q0_orig)
        assert targets[1].quality[0].item() == pytest.approx(q1_orig)
