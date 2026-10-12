"""Centralized plot settings and styling configuration.

All visual parameters are defined here to avoid scattered style changes
and to enable consistent figure appearance across all plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — safe for headless execution

import matplotlib.pyplot as plt

from vae_project.config.models import PlottingConfig


def apply_plot_settings(config: PlottingConfig) -> None:
    """Apply global matplotlib style settings from configuration.

    Args:
        config: Plotting configuration with DPI, colors, etc.
    """
    plt.rcParams.update(
        {
            "figure.dpi": config.dpi,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": config.dpi,
        }
    )


def get_class_colors(
    num_classes: int,
    colormap_name: str = "tab10",
) -> list:
    """Get a list of colors for class labeling.

    Args:
        num_classes: Number of classes to color.
        colormap_name: Matplotlib colormap name.

    Returns:
        List of color values, one per class.
    """
    colormap = matplotlib.colormaps[colormap_name]
    return [colormap(idx / max(num_classes - 1, 1)) for idx in range(num_classes)]


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    save_svg: bool = True,
) -> list[Path]:
    """Save a matplotlib figure to PNG and optionally SVG.

    Args:
        fig: Matplotlib figure to save.
        output_path: Base path (without extension) or with .png extension.
        save_svg: Whether to also save an SVG version.

    Returns:
        List of paths to saved files.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, format="png", bbox_inches="tight")
    saved.append(png_path)

    if save_svg:
        svg_path = output_path.with_suffix(".svg")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        saved.append(svg_path)

    plt.close(fig)
    return saved
