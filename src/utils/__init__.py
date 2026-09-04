"""
Utility modules (logging, configuration, seed management, MLflow tracking).
"""

import random
import numpy as np

def set_seed(seed: int = 42) -> None:
    """Set global random seeds for numpy and random for perfect reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
