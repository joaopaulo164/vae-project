"""Reconstruction and sample generation visualizations.

Generates:
- Side-by-side original vs reconstruction grids
- Generated samples from the prior distribution
- Latent space traversal grids (for 2D latent spaces)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from vae_project.config.models import PlottingConfig
from vae_project.models.vae import VariationalAutoencoder
from vae_project.visualization.plot_settings import save_figure

logger = logging.getLogger(__name__)

# Image dimensions for MNIST/Fashion-MNIST
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28


def _get_fixed_sample_indices(
    data_loader: DataLoader,  # type: ignore[type-arg]
    num_samples: int,
    seed: int = 42,
) -> list[int]:
    """Get reproducible sample indices from a DataLoader.

    Args:
        data_loader: DataLoader to sample from.
        num_samples: Number of samples to select.
        seed: Random seed.

    Returns:
        List of sample indices.
    """
    rng = np.random.default_rng(seed)
    dataset_size = len(data_loader.dataset)  # type: ignore[arg-type]
    n_select = min(num_samples, dataset_size)
    return list(rng.choice(dataset_size, size=n_select, replace=False))


def _collect_images_by_position(
    data_loader: DataLoader,  # type: ignore[type-arg]
    positions: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Return the images (and labels) at fixed positions of a loader's dataset.

    Args:
        data_loader: Loader whose dataset is indexed.
        positions: Positions in the dataset, in display order.
        device: Device receiving the stacked images.

    Returns:
        Tuple ``(images, labels)`` ordered like ``positions``.
    """
    dataset = data_loader.dataset
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for position in positions:
        image, label = dataset[position]
        images.append(image.unsqueeze(0))
        labels.append(int(label))
    return torch.cat(images, dim=0).to(device), labels


@torch.no_grad()
def plot_reconstruction_comparison(
    model: VariationalAutoencoder,
    data_loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    output_dir: Path,
    config: PlottingConfig,
    dataset_name: str,
    split_name: str,
    class_names: list[str] | None = None,
    is_quick_mode: bool = False,
    seed: int = 42,
    sample_indices: list[int] | None = None,
    title_suffix: str = "",
) -> list[Path]:
    """Generate a grid comparing original images with their reconstructions.

    Args:
        model: Trained VAE in eval mode.
        data_loader: DataLoader for the split.
        device: Compute device.
        output_dir: Output directory for figures.
        config: Plotting configuration.
        dataset_name: Dataset identifier for figure title.
        split_name: Split name ('train', 'val', 'test').
        class_names: Optional list of class names for labeling.
        is_quick_mode: Whether to add quick-mode warning.
        seed: Seed for reproducible sample selection when ``sample_indices``
            is not given.
        sample_indices: Fixed positions in the split to reconstruct. Sharing
            them across models makes the figures directly comparable. When
            None, the first images of the loader are used.
        title_suffix: Extra text appended to the title (topology, dimension).

    Returns:
        List of saved figure paths.
    """
    model.eval()

    n_cols = config.reconstruction_grid_cols
    num_samples = n_cols  # One row of originals, one row of reconstructions

    if sample_indices is not None:
        images_tensor, labels_list = _collect_images_by_position(
            data_loader, list(sample_indices[:num_samples]), device
        )
        num_samples = len(labels_list)
    else:
        collected_images: list[torch.Tensor] = []
        collected_labels: list[int] = []

        for batch_images, batch_labels in data_loader:
            batch_images = batch_images.to(device)
            for img, lbl in zip(batch_images, batch_labels, strict=True):
                collected_images.append(img.unsqueeze(0))
                collected_labels.append(int(lbl))
            if len(collected_images) >= num_samples:
                break

        images_tensor = torch.cat(collected_images[:num_samples], dim=0)
        labels_list = collected_labels[:num_samples]

    vae_output = model(images_tensor)
    reconstructions = vae_output.reconstruction

    # Reshape to image grids
    originals_np = images_tensor.cpu().view(-1, IMAGE_HEIGHT, IMAGE_WIDTH).numpy()
    reconstructions_np = reconstructions.cpu().view(-1, IMAGE_HEIGHT, IMAGE_WIDTH).numpy()

    fig, axes = plt.subplots(
        2,
        num_samples,
        figsize=(config.figure_width, config.figure_height * 0.4),
    )

    title = f"Reconstruction Comparison — {dataset_name} ({split_name}){title_suffix}"
    if is_quick_mode:
        title += " [QUICK MODE]"
    fig.suptitle(title, fontsize=11)

    for col_idx in range(num_samples):
        # Original
        axes[0, col_idx].imshow(originals_np[col_idx], cmap="gray", vmin=0, vmax=1)
        axes[0, col_idx].axis("off")
        if class_names is not None and col_idx < len(labels_list):
            axes[0, col_idx].set_title(class_names[labels_list[col_idx]], fontsize=6)

        # Reconstruction
        axes[1, col_idx].imshow(reconstructions_np[col_idx], cmap="gray", vmin=0, vmax=1)
        axes[1, col_idx].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=9)

    fig.tight_layout()
    output_path = Path(output_dir) / "reconstruction_comparison"
    return save_figure(fig, output_path, save_svg=config.save_svg)


@torch.no_grad()
def plot_generated_samples(
    model: VariationalAutoencoder,
    device: torch.device,
    output_dir: Path,
    config: PlottingConfig,
    dataset_name: str,
    num_samples: int = 64,
    is_quick_mode: bool = False,
    seed: int = 42,
    title_suffix: str = "",
) -> list[Path]:
    """Generate samples by decoding vectors sampled from the prior N(0, I).

    Args:
        model: Trained VAE.
        device: Compute device.
        output_dir: Output directory.
        config: Plotting configuration.
        dataset_name: Dataset identifier.
        num_samples: Number of samples to generate.
        is_quick_mode: Whether to add quick-mode warning.
        seed: Random seed for reproducible generation.
        title_suffix: Extra text appended to the title (topology, samples).

    Returns:
        List of saved figure paths.
    """
    torch.manual_seed(seed)
    model.eval()

    generated = model.generate_samples(num_samples=num_samples, device=device)
    generated_np = generated.cpu().view(-1, IMAGE_HEIGHT, IMAGE_WIDTH).numpy()

    n_rows = int(np.sqrt(num_samples))
    n_cols = int(np.ceil(num_samples / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.8, n_rows * 0.8))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    title = (
        f"Generated Samples from Prior — {dataset_name} "
        f"(latent_dim={model.latent_dimension}){title_suffix}"
    )
    if is_quick_mode:
        title += " [QUICK MODE]"
    fig.suptitle(title, fontsize=9)

    for img_idx, axis in enumerate(axes_flat):
        if img_idx < len(generated_np):
            axis.imshow(generated_np[img_idx], cmap="gray", vmin=0, vmax=1)
        axis.axis("off")

    fig.tight_layout()
    output_path = Path(output_dir) / "generated_samples"
    return save_figure(fig, output_path, save_svg=config.save_svg)


@torch.no_grad()
def plot_latent_traversal_2d(
    model: VariationalAutoencoder,
    device: torch.device,
    output_dir: Path,
    config: PlottingConfig,
    dataset_name: str,
    grid_size: int = 15,
    value_range: float = 3.0,
    is_quick_mode: bool = False,
) -> list[Path]:
    """Generate a 2D latent space traversal grid (manifold visualization).

    Only applicable when latent_dimension == 2.
    Decodes a uniform grid of latent vectors to visualize the learned manifold.

    Args:
        model: Trained VAE with latent_dimension == 2.
        device: Compute device.
        output_dir: Output directory.
        config: Plotting configuration.
        dataset_name: Dataset identifier.
        grid_size: Number of points along each latent dimension.
        value_range: Range of latent values to traverse (±value_range).
        is_quick_mode: Whether to add quick-mode warning.

    Returns:
        List of saved figure paths. Empty list if latent_dim != 2.
    """
    if model.latent_dimension != 2:
        logger.info(
            "Latent traversal skipped: latent_dim=%d (requires latent_dim=2)",
            model.latent_dimension,
        )
        return []

    model.eval()

    grid_x = np.linspace(-value_range, value_range, grid_size)
    grid_y = np.linspace(-value_range, value_range, grid_size)

    canvas_height = grid_size * IMAGE_HEIGHT
    canvas_width = grid_size * IMAGE_WIDTH
    canvas = np.zeros((canvas_height, canvas_width))

    for row_idx, y_val in enumerate(grid_y):
        for col_idx, x_val in enumerate(grid_x):
            latent_vector = torch.tensor([[x_val, y_val]], dtype=torch.float32, device=device)
            decoded = model.decode(latent_vector)
            image = decoded.cpu().view(IMAGE_HEIGHT, IMAGE_WIDTH).numpy()

            row_start = row_idx * IMAGE_HEIGHT
            col_start = col_idx * IMAGE_WIDTH
            canvas[row_start : row_start + IMAGE_HEIGHT, col_start : col_start + IMAGE_WIDTH] = (
                image
            )

    fig, axis = plt.subplots(figsize=(config.figure_width, config.figure_height))

    title = (
        f"2D Latent Space Traversal — {dataset_name} "
        f"(range: +/-{value_range}, grid: {grid_size}x{grid_size})"
    )
    if is_quick_mode:
        title += " [QUICK MODE]"

    axis.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    axis.set_xticks(
        np.linspace(0, canvas_width, grid_size),
        [f"{v:.1f}" for v in grid_x],
        rotation=90,
        fontsize=7,
    )
    axis.set_yticks(
        np.linspace(0, canvas_height, grid_size),
        [f"{v:.1f}" for v in grid_y],
        fontsize=7,
    )
    axis.set_xlabel("Latent Dimension 1")
    axis.set_ylabel("Latent Dimension 2")
    axis.set_title(title)

    output_path = Path(output_dir) / "latent_traversal_2d"
    return save_figure(fig, output_path, save_svg=config.save_svg)
