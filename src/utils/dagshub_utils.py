"""
src/utils/dagshub_utils.py — DagsHub & Remote MLflow Integration Utility

Automatically configures MLflow remote experiment tracking when DagsHub credentials are present in environment variables (.env).
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional

# Reconfigure stdout/stderr for Windows console compatibility with emojis
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)


def setup_dagshub_mlflow(
    repo_name: str = "KAIRO-Regime-Aware-Deep-Reinforcement-Learning-for-Optimal-Trade-Execution",
    username: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[str]:
    """
    Initialize DagsHub MLflow remote experiment tracking if credentials exist.

    Args:
        repo_name: DagsHub repository name
        username: DagsHub username (defaults to DAGSHUB_USERNAME env var)
        token: DagsHub API token (defaults to DAGSHUB_TOKEN env var)

    Returns:
        DagsHub MLflow tracking URI string if configured, or None for local fallback.
    """
    from dotenv import load_dotenv
    load_dotenv()

    user = username or os.getenv("DAGSHUB_USERNAME")
    tok = token or os.getenv("DAGSHUB_TOKEN")

    if user and tok:
        try:
            import dagshub

            os.environ["DAGSHUB_USERNAME"] = user
            os.environ["DAGSHUB_TOKEN"] = tok
            os.environ["MLFLOW_TRACKING_USERNAME"] = user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = tok

            tracking_uri = f"https://dagshub.com/{user}/{repo_name}.mlflow"
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

            import mlflow
            mlflow.set_tracking_uri(tracking_uri)

            logger.info(f"DagsHub MLflow tracking successfully initialized → {tracking_uri}")
            return tracking_uri
        except Exception as err:
            logger.warning(f"DagsHub initialization failed: {err}. Falling back to local MLflow tracking.")
            return None
    else:
        logger.info("DagsHub credentials not found in environment. Using local MLflow tracking.")
        return None
