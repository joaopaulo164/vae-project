"""Reconstruction quality metrics.

Computes pixel-level reconstruction quality metrics for evaluating
how well the VAE reconstructs its inputs.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from vae_project.models.vae import VariationalAutoencoder

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_reconstruction_metrics(
    model: VariationalAutoencoder,
    data_loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> dict[str, float]:
    """Compute mean reconstruction MSE and BCE on a dataset split.

    Args:
        model: Trained VAE in eval mode.
        data_loader: DataLoader for the split to evaluate.
        device: Compute device.

    Returns:
        Dictionary with 'mean_mse' and 'mean_bce' values.
    """
    model.eval()
    total_mse = 0.0
    total_bce = 0.0
    num_batches = 0

    for batch_images, _ in data_loader:
        batch_images = batch_images.to(device)
        flat_input = batch_images.view(batch_images.size(0), -1)

        vae_output = model(batch_images)
        reconstruction = vae_output.reconstruction

        mse = torch.nn.functional.mse_loss(reconstruction, flat_input).item()
        bce = torch.nn.functional.binary_cross_entropy(reconstruction, flat_input).item()

        total_mse += mse
        total_bce += bce
        num_batches += 1

    num_batches = max(num_batches, 1)
    return {
        "mean_mse": total_mse / num_batches,
        "mean_bce": total_bce / num_batches,
    }
