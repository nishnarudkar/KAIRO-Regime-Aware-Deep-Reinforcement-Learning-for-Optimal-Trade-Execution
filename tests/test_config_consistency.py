"""
The YAML files in config/ document the parameters the code uses. They are not
loaded at runtime, so this test fails if they drift from the code constants.
"""

from pathlib import Path

import pytest
import yaml

from src.environment.rewards import DEFAULT_REWARD_SCALE
from src.evaluation import protocol as P
from src.execution.impact_models import DEFAULT_ETA, DEFAULT_GAMMA

CONFIG = Path(__file__).resolve().parents[1] / "config"


def _load(name):
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))


def test_dqn_yaml_matches_code():
    cfg = _load("dqn.yaml")["dqn"]
    for key in ("learning_rate", "buffer_size", "batch_size", "gamma", "train_freq",
                "target_update_interval", "exploration_fraction",
                "exploration_initial_eps", "exploration_final_eps"):
        assert cfg[key] == pytest.approx(P.DQN_HPARAMS[key]), key
    assert cfg["net_arch"] == [128, 128]


def test_ppo_yaml_matches_code():
    cfg = _load("ppo.yaml")
    for key in ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range"):
        assert cfg[key] == pytest.approx(P.PPO_HPARAMS[key]), key
    assert cfg["policy_kwargs"]["net_arch"] == [128, 128]


def test_environment_yaml_matches_code():
    cfg = _load("environment.yaml")
    assert cfg["impact_model"]["eta"] == pytest.approx(DEFAULT_ETA)
    assert cfg["impact_model"]["gamma"] == pytest.approx(DEFAULT_GAMMA)
    assert cfg["reward"]["reward_scale"] == pytest.approx(DEFAULT_REWARD_SCALE)
    assert cfg["environment"]["action_fractions"] == [0.0, 0.10, 0.25, 0.50]
    assert cfg["environment"]["max_steps"] == P.HORIZON_STEPS
    assert cfg["environment"]["initial_inventory"] == P.TARGET_INVENTORY
