"""Shared pytest configuration."""

import os

# Must be set before the API module is imported: an in-memory execution store and
# MLflow's file store (used by the training-related tests).
os.environ.setdefault("KAIRO_DB_PATH", ":memory:")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("KAIRO_RATE_LIMIT", "0")   # tests hammer the API; a dedicated test enables the limiter


import pytest


def _models_available() -> bool:
    from src.agents import registry
    return all(registry.is_trained(m) for m in registry.MODEL_SPECS)


requires_models = pytest.mark.skipif(
    not _models_available(),
    reason="trained checkpoints missing; run `python scripts/train_models.py`",
)
