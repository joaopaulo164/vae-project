"""Environment information collection for reproducibility metadata.

Gathers Python, library, OS, hardware, and Git information to enable
exact reproduction of experiments.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any

import numpy as np
import psutil
import torch

logger = logging.getLogger(__name__)


def collect_environment_info() -> dict[str, Any]:
    """Collect comprehensive environment metadata.

    Returns:
        Dictionary with environment information including:
        - Python version
        - Library versions
        - OS and hardware details
        - GPU info if available
        - Git info if available
        - Timestamp
    """
    info: dict[str, Any] = {}

    # Timestamp
    info["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"

    # Python
    info["python_version"] = sys.version
    info["python_executable"] = sys.executable

    # Core libraries
    info["torch_version"] = torch.__version__
    info["numpy_version"] = np.__version__

    try:
        import torchvision

        info["torchvision_version"] = torchvision.__version__
    except ImportError:
        info["torchvision_version"] = "not installed"

    try:
        import pandas

        info["pandas_version"] = pandas.__version__
    except ImportError:
        info["pandas_version"] = "not installed"

    try:
        import sklearn

        info["scikit_learn_version"] = sklearn.__version__
    except ImportError:
        info["scikit_learn_version"] = "not installed"

    try:
        import umap

        info["umap_learn_version"] = umap.__version__
    except ImportError:
        info["umap_learn_version"] = "not installed"

    try:
        import matplotlib

        info["matplotlib_version"] = matplotlib.__version__
    except ImportError:
        info["matplotlib_version"] = "not installed"

    try:
        import seaborn

        info["seaborn_version"] = seaborn.__version__
    except ImportError:
        info["seaborn_version"] = "not installed"

    # OS
    info["os_system"] = platform.system()
    info["os_release"] = platform.release()
    info["os_version"] = platform.version()
    info["machine_type"] = platform.machine()
    info["processor"] = platform.processor()

    # Memory
    virtual_memory = psutil.virtual_memory()
    info["total_ram_gb"] = round(virtual_memory.total / (1024**3), 2)
    info["available_ram_gb"] = round(virtual_memory.available / (1024**3), 2)

    # CUDA / GPU
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version_pytorch"] = torch.version.cuda

    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_gb"] = round(gpu_props.total_memory / (1024**3), 2)
    else:
        info["gpu_count"] = 0
        info["gpu_name"] = "N/A"

    # Git info
    git_info = _get_git_info()
    info.update(git_info)

    return info


def _get_git_info() -> dict[str, str]:
    """Try to collect Git repository information.

    Returns:
        Dictionary with git commit hash and status, or placeholders if not available.
    """
    git_info: dict[str, str] = {}

    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],  # noqa: S607
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode("utf-8")
            .strip()
        )
        git_info["git_commit_hash"] = commit_hash
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        git_info["git_commit_hash"] = "unavailable"

    try:
        git_status = (
            subprocess.check_output(
                ["git", "status", "--short"],  # noqa: S607
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode("utf-8")
            .strip()
        )
        git_info["git_status"] = git_status if git_status else "clean"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        git_info["git_status"] = "unavailable"

    return git_info
