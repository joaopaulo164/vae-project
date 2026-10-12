"""Latent space visualization: scatter plots for 2D and projected spaces.

Generates scatter plots of latent representations colored by ground-truth
class labels. Visual analysis is the PRIMARY method for interpreting
latent space structure.

For latent_dimension == 2: plot directly.
For latent_dimension > 2: plot PCA, UMAP, t-SNE projections.

Important: Labels are used ONLY for coloring, not for training the VAE.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vae_project.config.models import PlottingConfig
from vae_project.projections.pca import PCAResult
from vae_project.projections.tsne import TSNEResult
from vae_project.projections.umap_projection import UMAPResult
from vae_project.visualization.plot_settings import get_class_colors, save_figure

logger = logging.getLogger(__name__)


def _scatter_with_labels(
    axis: plt.Axes,
    points: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    config: PlottingConfig,
    x_label: str = "Component 1",
    y_label: str = "Component 2",
) -> None:
    """Draw a scatter plot with per-class coloring and legend.

    Args:
        axis: Matplotlib axes to draw on.
        points: 2D coordinates, shape (N, 2).
        labels: Integer class labels, shape (N,).
        class_names: List of class name strings.
        config: Plotting configuration for point size and alpha.
        x_label: X-axis label.
        y_label: Y-axis label.
    """
    colors = get_class_colors(len(class_names), config.colormap)
    unique_labels = np.unique(labels)

    for class_idx in unique_labels:
        mask = labels == class_idx
        color = colors[class_idx]
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            c=[color],
            label=class_names[class_idx] if class_idx < len(class_names) else str(class_idx),
            s=config.point_size,
            alpha=config.point_alpha,
            edgecolors="none",
        )

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.legend(
        markerscale=3,
        fontsize=8,
        loc="best",
        framealpha=0.7,
    )


def _build_figure_title(
    method: str,
    dataset_name: str,
    latent_dimension: int,
    split_name: str,
    n_samples: int,
    is_quick_mode: bool,
    extra_info: str = "",
) -> str:
    """Build an informative figure title.

    Args:
        method: Projection method name.
        dataset_name: Dataset identifier.
        latent_dimension: Latent space dimensionality.
        split_name: 'train', 'val', or 'test'.
        n_samples: Number of points displayed.
        is_quick_mode: Whether to add quick-mode warning.
        extra_info: Optional additional information string.

    Returns:
        Title string.
    """
    title = (
        f"{method} — {dataset_name} | "
        f"latent_dim={latent_dimension} | "
        f"{split_name} | n={n_samples:,}"
    )
    if extra_info:
        title += f" | {extra_info}"
    if is_quick_mode:
        title += " [QUICK MODE — NOT FOR SCIENTIFIC REPORTING]"
    return title


def plot_latent_space_direct_2d(
    latent_means: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    config: PlottingConfig,
    dataset_name: str,
    split_name: str = "val",
    is_quick_mode: bool = False,
) -> list[Path]:
    """Plot the 2D latent space directly (only when latent_dimension == 2).

    Args:
        latent_means: Latent mean vectors, shape (N, 2).
        labels: Ground-truth labels, shape (N,).
        class_names: Class name strings.
        output_dir: Output directory.
        config: Plotting configuration.
        dataset_name: Dataset identifier.
        split_name: Split name for title.
        is_quick_mode: Whether to add quick-mode warning.

    Returns:
        List of saved figure paths.
    """
    if latent_means.shape[1] != 2:
        raise ValueError(f"Direct 2D plot requires latent_dim=2, got {latent_means.shape[1]}")

    n_display = min(len(labels), config.max_display_points)
    if n_display < len(labels):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(labels), size=n_display, replace=False)
        plot_points = latent_means[indices]
        plot_labels = labels[indices]
        logger.info(
            "Latent 2D plot: displaying %d/%d points (subsampled for readability).",
            n_display,
            len(labels),
        )
    else:
        plot_points = latent_means
        plot_labels = labels

    fig, axis = plt.subplots(figsize=(config.figure_width, config.figure_height))

    title = _build_figure_title(
        method="Latent Space (Direct 2D)",
        dataset_name=dataset_name,
        latent_dimension=2,
        split_name=split_name,
        n_samples=n_display,
        is_quick_mode=is_quick_mode,
    )
    axis.set_title(title, fontsize=10)

    _scatter_with_labels(
        axis=axis,
        points=plot_points,
        labels=plot_labels,
        class_names=class_names,
        config=config,
        x_label="Latent Dimension 1",
        y_label="Latent Dimension 2",
    )

    output_path = Path(output_dir) / "latent_space_direct_2d"
    return save_figure(fig, output_path, save_svg=config.save_svg)


def plot_pca_projection(
    pca_result: PCAResult,
    labels: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    config: PlottingConfig,
    dataset_name: str,
    split_name: str = "val",
    is_quick_mode: bool = False,
) -> list[Path]:
    """Plot PCA 2D projection with explained variance annotation.

    Args:
        pca_result: PCA result with projection and explained variance.
        labels: Ground-truth labels, shape (N,).
        class_names: Class name strings.
        output_dir: Output directory.
        config: Plotting configuration.
        dataset_name: Dataset identifier.
        split_name: Split name.
        is_quick_mode: Whether to add quick-mode warning.

    Returns:
        List of saved figure paths.
    """
    ev1 = float(pca_result.explained_variance_ratio[0]) * 100
    ev2 = (
        float(pca_result.explained_variance_ratio[1]) * 100 if pca_result.n_components > 1 else 0.0
    )
    ev_total = ev1 + ev2

    extra_info = f"PC1={ev1:.1f}%, PC2={ev2:.1f}% (total={ev_total:.1f}%)"

    n_display = min(len(labels), config.max_display_points)
    if n_display < len(labels):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(labels), size=n_display, replace=False)
        plot_points = pca_result.projection_2d[indices]
        plot_labels = labels[indices]
    else:
        plot_points = pca_result.projection_2d
        plot_labels = labels

    fig, axis = plt.subplots(figsize=(config.figure_width, config.figure_height))

    title = _build_figure_title(
        method="PCA Projection",
        dataset_name=dataset_name,
        latent_dimension=pca_result.latent_dimension,
        split_name=split_name,
        n_samples=n_display,
        is_quick_mode=is_quick_mode,
        extra_info=extra_info,
    )
    axis.set_title(title, fontsize=9)

    _scatter_with_labels(
        axis=axis,
        points=plot_points,
        labels=plot_labels,
        class_names=class_names,
        config=config,
        x_label=f"PC1 ({ev1:.1f}% var.)",
        y_label=f"PC2 ({ev2:.1f}% var.)",
    )

    output_filename = f"latent_space_pca_{pca_result.latent_dimension}d"
    output_path = Path(output_dir) / output_filename
    return save_figure(fig, output_path, save_svg=config.save_svg)


def plot_pca_explained_variance(
    pca_result: PCAResult,
    output_dir: Path,
    config: PlottingConfig,
    dataset_name: str,
    is_quick_mode: bool = False,
) -> list[Path]:
    """Plot PCA explained variance bar chart.

    Args:
        pca_result: PCA result with explained variance.
        output_dir: Output directory.
        config: Plotting configuration.
        dataset_name: Dataset identifier.
        is_quick_mode: Whether to add quick-mode warning.

    Returns:
        List of saved figure paths.
    """
    fig, axis = plt.subplots(figsize=(config.figure_width * 0.6, config.figure_height * 0.7))

    components = list(range(1, pca_result.n_components + 1))
    ev_ratios = [float(v) * 100 for v in pca_result.explained_variance_ratio]

    axis.bar(components, ev_ratios, color="#3b82d4", edgecolor="white")

    title = (
        f"PCA Explained Variance — {dataset_name} " f"(latent_dim={pca_result.latent_dimension})"
    )
    if is_quick_mode:
        title += " [QUICK MODE]"
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Principal Component")
    axis.set_ylabel("Explained Variance (%)")
    axis.set_xticks(components)

    ev_total = sum(ev_ratios[:2]) if len(ev_ratios) >= 2 else sum(ev_ratios)
    axis.annotate(
        f"Top-2 components: {ev_total:.1f}% of linear variance",
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
    )

    output_filename = f"pca_explained_variance_{pca_result.latent_dimension}d"
    output_path = Path(output_dir) / output_filename
    return save_figure(fig, output_path, save_svg=config.save_svg)


def plot_umap_projection(
    umap_result: UMAPResult,
    labels: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    config: PlottingConfig,
    dataset_name: str,
    split_name: str = "val",
    is_quick_mode: bool = False,
) -> list[Path]:
    """Plot UMAP 2D projection.

    Args:
        umap_result: UMAP result with 2D projection.
        labels: Ground-truth labels.
        class_names: Class name strings.
        output_dir: Output directory.
        config: Plotting configuration.
        dataset_name: Dataset identifier.
        split_name: Split name.
        is_quick_mode: Whether to add quick-mode warning.

    Returns:
        List of saved figure paths.
    """
    n_display = len(labels)
    extra_info = f"n_neighbors={umap_result.n_neighbors}, min_dist={umap_result.min_dist}"

    fig, axis = plt.subplots(figsize=(config.figure_width, config.figure_height))

    title = _build_figure_title(
        method="UMAP Projection",
        dataset_name=dataset_name,
        latent_dimension=umap_result.latent_dimension,
        split_name=split_name,
        n_samples=n_display,
        is_quick_mode=is_quick_mode,
        extra_info=extra_info,
    )
    axis.set_title(title, fontsize=9)

    note = "Note: UMAP does not provide explained variance. Local structure is preserved."
    axis.annotate(
        note,
        xy=(0.5, -0.08),
        xycoords="axes fraction",
        ha="center",
        fontsize=7,
        style="italic",
        color="gray",
    )

    _scatter_with_labels(
        axis=axis,
        points=umap_result.projection_2d,
        labels=labels,
        class_names=class_names,
        config=config,
        x_label="UMAP 1",
        y_label="UMAP 2",
    )

    output_filename = f"latent_space_umap_{umap_result.latent_dimension}d"
    output_path = Path(output_dir) / output_filename
    return save_figure(fig, output_path, save_svg=config.save_svg)


def plot_tsne_projection(
    tsne_result: TSNEResult,
    labels: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    config: PlottingConfig,
    dataset_name: str,
    split_name: str = "val",
    is_quick_mode: bool = False,
) -> list[Path]:
    """Plot t-SNE 2D projection.

    Args:
        tsne_result: t-SNE result with 2D projection.
        labels: Ground-truth labels.
        class_names: Class name strings.
        output_dir: Output directory.
        config: Plotting configuration.
        dataset_name: Dataset identifier.
        split_name: Split name.
        is_quick_mode: Whether to add quick-mode warning.

    Returns:
        List of saved figure paths.
    """
    n_display = len(labels)
    extra_info = f"perplexity={tsne_result.perplexity}"

    fig, axis = plt.subplots(figsize=(config.figure_width, config.figure_height))

    title = _build_figure_title(
        method="t-SNE Projection",
        dataset_name=dataset_name,
        latent_dimension=tsne_result.latent_dimension,
        split_name=split_name,
        n_samples=n_display,
        is_quick_mode=is_quick_mode,
        extra_info=extra_info,
    )
    axis.set_title(title, fontsize=9)

    note = (
        "Note: t-SNE does not provide explained variance. "
        "Global distances and cluster sizes are not directly interpretable."
    )
    axis.annotate(
        note,
        xy=(0.5, -0.08),
        xycoords="axes fraction",
        ha="center",
        fontsize=7,
        style="italic",
        color="gray",
    )

    _scatter_with_labels(
        axis=axis,
        points=tsne_result.projection_2d,
        labels=labels,
        class_names=class_names,
        config=config,
        x_label="t-SNE 1",
        y_label="t-SNE 2",
    )

    output_filename = f"latent_space_tsne_{tsne_result.latent_dimension}d"
    output_path = Path(output_dir) / output_filename
    return save_figure(fig, output_path, save_svg=config.save_svg)
