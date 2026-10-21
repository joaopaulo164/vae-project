"""Tests for the best-epoch and early-stopping markers of the training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from vae_project.config.models import PlottingConfig
from vae_project.visualization.training_plots import (
    _mark_best_and_final_epochs,
    plot_training_loss_components,
    plot_training_total_loss,
)

pytestmark = pytest.mark.unit


def _history(improving: list[bool]) -> pd.DataFrame:
    epochs = list(range(len(improving)))
    return pd.DataFrame(
        {
            "epoch": epochs,
            "train_total_loss": [10.0 - e for e in epochs],
            "val_total_loss": [11.0 - e for e in epochs],
            "train_reconstruction_loss": [9.0 - e for e in epochs],
            "val_reconstruction_loss": [10.0 - e for e in epochs],
            "train_kl_divergence_loss": [1.0 for _ in epochs],
            "val_kl_divergence_loss": [1.0 for _ in epochs],
            "is_best_epoch": improving,
        }
    )


def _labels(axis: plt.Axes) -> list[str]:
    return [str(line.get_label()) for line in axis.get_lines()]


def test_best_epoch_marker_uses_the_last_improving_epoch() -> None:
    figure, axis = plt.subplots()
    history = _history([True, True, True, False, False])

    _mark_best_and_final_epochs([axis], history)

    labels = _labels(axis)
    assert "Best epoch (2)" in labels
    assert "Early stopping / last epoch (4)" in labels
    plt.close(figure)


def test_no_early_stopping_marker_when_the_last_epoch_is_the_best() -> None:
    figure, axis = plt.subplots()

    _mark_best_and_final_epochs([axis], _history([True, True, True]))

    labels = _labels(axis)
    assert "Best epoch (2)" in labels
    assert not any(label.startswith("Early stopping") for label in labels)
    plt.close(figure)


def test_markers_are_skipped_without_history_or_flags() -> None:
    figure, axis = plt.subplots()

    _mark_best_and_final_epochs([axis], _history([]))
    _mark_best_and_final_epochs([axis], _history([True, False]).drop(columns="is_best_epoch"))

    assert axis.get_lines() == []
    plt.close(figure)


def test_both_training_figures_are_written(tmp_path: Path) -> None:
    history = _history([True, True, False])
    config = PlottingConfig(save_svg=True)

    totals = plot_training_total_loss(history, tmp_path, config, "mnist")
    components = plot_training_loss_components(history, tmp_path, config, "mnist")

    names = sorted(path.name for path in [*totals, *components])
    assert names == [
        "training_loss_components.png",
        "training_loss_components.svg",
        "training_total_loss.png",
        "training_total_loss.svg",
    ]
    assert all(path.stat().st_size > 0 for path in [*totals, *components])
