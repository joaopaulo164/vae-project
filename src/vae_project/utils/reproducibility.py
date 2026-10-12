"""Reproducibility utilities: global seed configuration.

Sets seeds across all relevant libraries to maximize experiment reproducibility.
Note: absolute determinism across different hardware, CUDA versions, or
library versions cannot be guaranteed.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Configure seeds across Python, NumPy, and PyTorch.

    Sets random seeds for:
    - Python built-in random module
    - NumPy
    - PyTorch CPU operations
    - PyTorch CUDA operations (if available)

    Args:
        seed: The integer seed value.
        deterministic: If True, enables torch.use_deterministic_algorithms(True)
            which forces deterministic CUDA operations at the cost of performance.
            Some operations may raise RuntimeError if no deterministic
            implementation exists.

    Note:
        Reproducibility between different machines, OS, CUDA versions, or
        library versions is not guaranteed even with identical seeds.
    """
    logger.info("Setting global seed: %d (deterministic=%s)", seed, deterministic)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(
            "Deterministic mode enabled. Some CUDA operations may be slower "
            "or raise RuntimeError if no deterministic kernel exists."
        )
    else:
        # Allow cuDNN auto-tuning for performance
        torch.backends.cudnn.benchmark = True
        logger.debug("Deterministic mode disabled (default). cuDNN benchmark enabled.")
