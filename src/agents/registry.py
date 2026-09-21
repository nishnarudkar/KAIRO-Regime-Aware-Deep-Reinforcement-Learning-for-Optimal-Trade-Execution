"""
src/agents/registry.py — Trained-model registry

Trained agents are produced once by ``scripts/train_models.py`` and served from
disk. Nothing is ever trained inside an API request.

Layout of the models directory (``KAIRO_MODELS_DIR`` or ``<repo>/models``):

    registry.json        metadata + held-out evaluation summary of each model
    kairo_hmm.joblib     HMM fitted on the pooled training data
    dqn_base.zip         SB3 checkpoints, one per model id
    dqn_regime.zip
    ppo_base.zip
    ppo_regime.zip

A model is reported as ``trained`` only when its checkpoint exists (and, for
regime-aware models, the HMM exists too).
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# model_id -> spec. ``policy`` is the name used by the API and the UI.
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "dqn_base": dict(policy="DQN", name="DQN (no regime)", algo="dqn", kind="plain",
                     regime_aware=False, state_dim=7, algorithm_class="Value-Based / Off-Policy"),
    "dqn_regime": dict(policy="Regime-Aware DQN", name="Regime-Aware DQN", algo="dqn", kind="regime",
                       regime_aware=True, state_dim=12, algorithm_class="Value-Based / Off-Policy"),
    "ppo_base": dict(policy="PPO", name="PPO (no regime)", algo="ppo", kind="plain",
                     regime_aware=False, state_dim=7, algorithm_class="Policy-Gradient / On-Policy"),
    "ppo_regime": dict(policy="Regime-Aware PPO", name="Regime-Aware PPO", algo="ppo", kind="regime",
                       regime_aware=True, state_dim=12, algorithm_class="Policy-Gradient / On-Policy"),
}
POLICY_TO_MODEL = {spec["policy"]: mid for mid, spec in MODEL_SPECS.items()}
BASELINE_POLICIES = ("TWAP", "VWAP", "POV")
RL_POLICIES = tuple(POLICY_TO_MODEL)
HMM_FILENAME = "kairo_hmm.joblib"
REGISTRY_FILENAME = "registry.json"


class ModelUnavailable(RuntimeError):
    """Raised when a policy is requested whose trained checkpoint is not available."""


def models_dir() -> Path:
    return Path(os.environ.get("KAIRO_MODELS_DIR", REPO_ROOT / "models"))


def _checkpoint_path(model_id: str) -> Path:
    return models_dir() / f"{model_id}.zip"


def read_registry() -> Dict[str, Any]:
    path = models_dir() / REGISTRY_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_registry(data: Dict[str, Any]) -> None:
    models_dir().mkdir(parents=True, exist_ok=True)
    (models_dir() / REGISTRY_FILENAME).write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_trained(model_id: str) -> bool:
    spec = MODEL_SPECS[model_id]
    if not _checkpoint_path(model_id).exists():
        return False
    if spec["regime_aware"] and not (models_dir() / HMM_FILENAME).exists():
        return False
    return True


def describe_models() -> List[Dict[str, Any]]:
    """Metadata for every model, including real training / evaluation information."""
    reg = read_registry()
    out = []
    for mid, spec in MODEL_SPECS.items():
        meta = (reg.get("models") or {}).get(mid, {})
        trained = is_trained(mid)
        out.append({
            "model_id": mid, "name": spec["name"], "policy": spec["policy"],
            "algorithm_class": spec["algorithm_class"], "regime_aware": spec["regime_aware"],
            "state_dim": spec["state_dim"], "net_arch": [128, 128],
            "trained": trained, "status": "trained" if trained else "untrained",
            "timesteps": meta.get("timesteps") if trained else None,
            "trained_at": meta.get("trained_at") if trained else None,
            "evaluation": meta.get("evaluation") if trained else None,
        })
    return out


@dataclass
class LoadedPolicy:
    policy: str
    model_id: str
    agent: Any            # DQNAgent / PPOAgent wrapper (has .predict and .model)
    hmm: Any              # MarketHMM or None
    spec: Dict[str, Any]


_cache: Dict[str, LoadedPolicy] = {}
_lock = threading.Lock()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def load_policy(policy: str) -> LoadedPolicy:
    """Load (and cache) the trained agent for a policy name. Raises ModelUnavailable."""
    if policy not in POLICY_TO_MODEL:
        raise ModelUnavailable(f"'{policy}' is not a learned policy.")
    model_id = POLICY_TO_MODEL[policy]
    with _lock:
        if model_id in _cache:
            return _cache[model_id]
        if not is_trained(model_id):
            raise ModelUnavailable(
                f"Model '{policy}' has no trained checkpoint. "
                f"Run `python scripts/train_models.py` to train and save it."
            )
        spec = MODEL_SPECS[model_id]
        path = str(_checkpoint_path(model_id))
        if spec["algo"] == "dqn":
            from src.agents.dqn_agent import DQNAgent
            agent = DQNAgent.load(path, env=None)
        else:
            from src.agents.ppo_agent import PPOAgent
            agent = PPOAgent.load(path, env=None)
        hmm = None
        if spec["regime_aware"]:
            from src.regimes.hmm_model import MarketHMM
            hmm = MarketHMM.load(str(models_dir() / HMM_FILENAME))
        loaded = LoadedPolicy(policy=policy, model_id=model_id, agent=agent, hmm=hmm, spec=spec)
        _cache[model_id] = loaded
        return loaded


def load_hmm():
    """The pooled HMM used for regime-aware models and the regime endpoint, or None."""
    path = models_dir() / HMM_FILENAME
    if not path.exists():
        return None
    from src.regimes.hmm_model import MarketHMM
    return MarketHMM.load(str(path))
