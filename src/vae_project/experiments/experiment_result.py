"""Typed result of one complete, single-configuration experiment.

The dataclass here is deliberately free of heavy imports so that the service
(:mod:`vae_project.experiments.single_experiment`), the reporting writers and
the factorial protocol can all depend on it without import cycles.

Status vocabulary of one execution:

* ``completed``: training and every evidence step succeeded.
* ``partial``: training succeeded, but at least one evidence step failed
  (equivalent to "completed with warnings"; artifacts are preserved).
* ``failed``: training or checkpoint handling failed.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

DATA_ORIGIN_REAL = "real"
DATA_ORIGIN_SYNTHETIC = "synthetic"

FIGURE_CATEGORY_TRAINING = "training"
FIGURE_CATEGORY_RECONSTRUCTIONS = "reconstructions"
FIGURE_CATEGORY_GENERATED = "generated_samples"
FIGURE_CATEGORY_LATENT = "latent_space"
FIGURE_CATEGORIES = (
    FIGURE_CATEGORY_TRAINING,
    FIGURE_CATEGORY_RECONSTRUCTIONS,
    FIGURE_CATEGORY_GENERATED,
    FIGURE_CATEGORY_LATENT,
)

LATENT_KIND_DIRECT = "direct"
LATENT_KIND_PCA = "pca"
LATENT_KIND_UMAP = "umap"
LATENT_KIND_TSNE = "tsne"
LATENT_KIND_PCA_VARIANCE = "pca_explained_variance"


class EvidenceProfile(str, Enum):
    """How much evidence a single execution generates after training.

    Attributes:
        FULL: Training curves, reconstructions, generated samples and latent
            space evidence (direct 2D plot or PCA/UMAP/t-SNE) with metrics.
        TRAINING_ONLY: Training, history and training curves only. Used by the
            specialized topology selection, which ranks by validation loss.
    """

    FULL = "full"
    TRAINING_ONLY = "training_only"


def relative_posix(path: Path | None, base: Path) -> str | None:
    """Return ``path`` relative to ``base`` with forward slashes (None-safe).

    A path outside ``base`` is expressed with ``..`` segments.
    """
    if path is None:
        return None
    return Path(os.path.relpath(path, base)).as_posix()


@dataclass
class SingleExperimentResult:
    """Outcome of one independent training and its evidence generation.

    Path attributes hold paths as produced by the service (rooted at the run
    directory). ``relative_*`` helpers express them relative to any base.
    Values that do not apply stay ``None`` and never become zero.
    """

    dataset: str
    experiment_type: str
    topology_name: str
    hidden_dimensions: list[int]
    latent_dimension: int
    seed: int
    status: str = STATUS_FAILED
    split: str = "val"
    device: str = ""
    quick_mode: bool = False
    synthetic_data: bool = False
    execution_id: str = ""
    parameter_count: int | None = None
    best_epoch: int | None = None
    epochs_trained: int | None = None
    validation_reconstruction_loss: float | None = None
    validation_kl_divergence: float | None = None
    validation_classification_loss: float | None = None
    validation_total_loss: float | None = None
    training_duration_seconds: float | None = None
    total_duration_seconds: float | None = None
    visualization_mode: str | None = None
    n_samples_total: int | None = None
    n_projection_samples: int | None = None
    used_sampling: bool = False
    pca_explained_variance: tuple[float, float, float] | None = None
    projection_metrics: list[dict[str, object]] = field(default_factory=list)
    run_directory: Path | None = None
    best_checkpoint_path: Path | None = None
    run_summary_path: Path | None = None
    manifest_path: Path | None = None
    metrics_path: Path | None = None
    config_path: Path | None = None
    log_path: Path | None = None
    environment_reference: str | None = None
    figure_paths: dict[str, list[Path]] = field(default_factory=dict)
    latent_figure_paths: dict[str, list[Path]] = field(default_factory=dict)
    table_paths: list[Path] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def data_origin(self) -> str:
        """'synthetic' for test/smoke-test data, otherwise 'real'."""
        return DATA_ORIGIN_SYNTHETIC if self.synthetic_data else DATA_ORIGIN_REAL

    @property
    def has_valid_training(self) -> bool:
        """True when training finished and a finite best validation loss exists."""
        loss = self.validation_total_loss
        return self.status != STATUS_FAILED and loss is not None and math.isfinite(loss)

    def error_summary(self) -> str | None:
        """Return failed operations as a single line, or None."""
        if not self.errors:
            return None
        return "; ".join(f"{item['operation']}: {item['error']}" for item in self.errors)

    def all_figure_files(self) -> list[Path]:
        """Return every figure file in a stable order (training first)."""
        files: list[Path] = []
        for category in FIGURE_CATEGORIES:
            files.extend(self.figure_paths.get(category, []))
        return files

    def relative_figures(self, base: Path) -> dict[str, list[str]]:
        """Return figure files per category, relative to ``base``."""
        return {
            category: [path.relative_to(base).as_posix() for path in paths]
            for category, paths in self.figure_paths.items()
        }

    def relative_tables(self, base: Path) -> list[str]:
        """Return table and data files relative to ``base``."""
        return [path.relative_to(base).as_posix() for path in self.table_paths]

    def latent_figure(self, kind: str, suffix: str = ".png") -> Path | None:
        """Return the first latent-space figure of a kind with ``suffix``."""
        for path in self.latent_figure_paths.get(kind, []):
            if path.suffix == suffix:
                return path
        return None
