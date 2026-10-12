"""Device selection and CUDA availability utilities.

Handles CPU/CUDA device selection based on configuration policy with
clear, actionable error messages when CUDA is requested but unavailable.
"""

from __future__ import annotations

import logging

import torch

from vae_project.config.models import DevicePolicy

logger = logging.getLogger(__name__)


class DeviceSetupError(Exception):
    """Raised when device setup fails (e.g., CUDA requested but unavailable)."""


def get_device(policy: DevicePolicy, fallback_to_cpu: bool = False) -> torch.device:
    """Select the compute device based on the configured policy.

    Args:
        policy: The device selection policy (auto, cpu, cuda).
        fallback_to_cpu: If True and CUDA is requested but unavailable,
            fall back to CPU silently. If False, raise DeviceSetupError.

    Returns:
        The selected torch.device.

    Raises:
        DeviceSetupError: If policy is 'cuda' and CUDA is not available
            and fallback_to_cpu is False.
    """
    if policy == DevicePolicy.CPU:
        device = torch.device("cpu")
        logger.info("Device: CPU (explicitly configured)")
        return device

    cuda_available = torch.cuda.is_available()

    if policy == DevicePolicy.AUTO:
        if cuda_available:
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            logger.info("Device: CUDA (auto-detected) — %s", device_name)
        else:
            device = torch.device("cpu")
            logger.info("Device: CPU (auto-selected, CUDA not available)")
        return device

    if policy == DevicePolicy.CUDA:
        if cuda_available:
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            logger.info("Device: CUDA (explicitly configured) — %s", device_name)
            return device

        if fallback_to_cpu:
            logger.warning(
                "CUDA was requested but is not available. "
                "Falling back to CPU because cuda_fallback_to_cpu=true is configured."
            )
            return torch.device("cpu")

        raise DeviceSetupError(
            "CUDA was requested (runtime.device: cuda) but is not available.\n"
            "Possible causes:\n"
            "  1. NVIDIA GPU not present in this machine.\n"
            "  2. CUDA drivers not installed or outdated.\n"
            "  3. PyTorch was installed without CUDA support.\n"
            "How to fix:\n"
            "  - For CPU execution: set runtime.device to 'cpu' or 'auto'.\n"
            "  - For CUDA: install the correct PyTorch build from https://pytorch.org\n"
            "  - To allow automatic fallback: set runtime.cuda_fallback_to_cpu: true\n"
            f"  - PyTorch version: {torch.__version__}\n"
            f"  - torch.cuda.is_available(): {cuda_available}"
        )

    raise DeviceSetupError(f"Unknown device policy: {policy}")


def get_device_info(device: torch.device) -> dict[str, str]:
    """Collect device information for logging and artifact metadata.

    Args:
        device: The torch device in use.

    Returns:
        Dictionary with device information fields.
    """
    info: dict[str, str] = {"device_type": device.type}

    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = str(torch.cuda.device_count())
        info["cuda_version"] = torch.version.cuda or "unknown"
        total_memory = torch.cuda.get_device_properties(0).total_memory
        info["cuda_total_memory_gb"] = f"{total_memory / (1024 ** 3):.2f}"
    else:
        info["cuda_available"] = str(torch.cuda.is_available())

    return info
