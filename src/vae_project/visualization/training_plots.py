"""Training curve visualization: loss vs epoch plots.

Generates separate and combined plots for:
- Total loss (train and validation)
- Reconstruction loss
- KL divergence
- Classification loss (when supervised component is enabled)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from vae_project.config.models import PlottingConfig
from vae_project.visualization.plot_settings import save_figure

logger = logging.getLogger(__name__)


def _mark_best_and_final_epochs(axes: list[Axes], history_df: pd.DataFrame) -> None:
    """Draw the best-epoch and the last-epoch markers on every axis.

    ``is_best_epoch`` is true for every epoch that improved the validation loss, so
    the best epoch is the last flagged row. When training ended after a later epoch
    (early stopping), the last trained epoch is marked as well.

    Args:
        axes: Axes that share the epoch axis.
        history_df: Training history with ``epoch`` and ``is_best_epoch`` columns.
    """
    if history_df.empty:
        return
    best_epoch: int | None = None
    if "is_best_epoch" in history_df.columns:
        flagged = history_df[history_df["is_best_epoch"].astype(bool)]
        if not flagged.empty:
            best_epoch = int(flagged["epoch"].iloc[-1])
    last_epoch = int(history_df["epoch"].iloc[-1])

    for axis in axes:
        if best_epoch is not None:
            axis.axvline(
                x=best_epoch,
                color="gray",
                linestyle=":",
                alpha=0.7,
                label=f"Best epoch ({best_epoch})",
            )
        if best_epoch is not None and last_epoch > best_epoch:
            axis.axvline(
                x=last_epoch,
                color="firebrick",
                linestyle="-.",
                alpha=0.5,
                label=f"Early stopping / last epoch ({last_epoch})",
            )


def plot_training_total_loss(
    history_df: pd.DataFrame,
    output_dir: Path,
    config: PlottingConfig,
    dataset_name: str = "",
    is_quick_mode: bool = False,
) -> list[Path]:
    """Plot total training and validation loss curves.

    Args:
        history_df: DataFrame with columns epoch, train_total_loss, val_total_loss.
        output_dir: Directory for output figures.
        config: Plotting configuration.
        dataset_name: Dataset identifier for figure title.
        is_quick_mode: Whether to add quick-mode warning to title.

    Returns:
        List of saved figure paths.
    """
    fig, axis = plt.subplots(figsize=(config.figure_width, config.figure_height))

    title = f"Training and Validation Total Loss — {dataset_name}"
    if is_quick_mode:
        title += " [QUICK MODE — NOT FOR SCIENTIFIC REPORTING]"

    axis.plot(
        history_df["epoch"],
        history_df["train_total_loss"],
        label="Train Total Loss",
        linewidth=2,
    )
    axis.plot(
        history_df["epoch"],
        history_df["val_total_loss"],
        label="Validation Total Loss",
        linewidth=2,
        linestyle="--",
    )

    _mark_best_and_final_epochs([axis], history_df)

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(title)
    axis.legend()

    output_path = Path(output_dir) / "training_total_loss"
    return save_figure(fig, output_path, save_svg=config.save_svg)


def plot_training_loss_components(
    history_df: pd.DataFrame,
    output_dir: Path,
    config: PlottingConfig,
    dataset_name: str = "",
    is_quick_mode: bool = False,
) -> list[Path]:
    """Plot separate panels for each loss component.

    Args:
        history_df: Training history DataFrame.
        output_dir: Output figure directory.
        config: Plotting configuration.
        dataset_name: Dataset identifier.
        is_quick_mode: Whether to mark as quick mode.

    Returns:
        List of saved figure paths.
    """
    has_classification = (
        "train_classification_loss" in history_df.columns
        and history_df["train_classification_loss"].sum() > 0
    )
    n_panels = 3 if has_classification else 2

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(config.figure_width * n_panels / 2, config.figure_height * 0.7),
    )

    title = f"Loss Components — {dataset_name}"
    if is_quick_mode:
        title += " [QUICK MODE]"
    fig.suptitle(title)

    # Panel 1: Reconstruction loss
    axes[0].plot(
        history_df["epoch"],
        history_df["train_reconstruction_loss"],
        label="Train",
        linewidth=2,
    )
    axes[0].plot(
        history_df["epoch"],
        history_df["val_reconstruction_loss"],
        label="Val",
        linewidth=2,
        linestyle="--",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Reconstruction Loss")

    # Panel 2: KL divergence
    axes[1].plot(
        history_df["epoch"],
        history_df["train_kl_divergence_loss"],
        label="Train",
        linewidth=2,
    )
    axes[1].plot(
        history_df["epoch"],
        history_df["val_kl_divergence_loss"],
        label="Val",
        linewidth=2,
        linestyle="--",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("KL Divergence")
    axes[1].set_title("KL Divergence Loss")

    # Panel 3: Classification loss (if applicable)
    if has_classification:
        axes[2].plot(
            history_df["epoch"],
            history_df["train_classification_loss"],
            label="Train",
            linewidth=2,
        )
        axes[2].plot(
            history_df["epoch"],
            history_df["val_classification_loss"],
            label="Val",
            linewidth=2,
            linestyle="--",
        )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Classification Loss")
        axes[2].set_title("Classification Loss")

    _mark_best_and_final_epochs(list(axes), history_df)
    for panel in axes:
        panel.legend()

    fig.tight_layout()
    output_path = Path(output_dir) / "training_loss_components"
    return save_figure(fig, output_path, save_svg=config.save_svg)
