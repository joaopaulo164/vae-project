"""Reusable generation of latent-space evidence for one trained VAE.

Given a trained model (best checkpoint already loaded) and a data loader, this
module extracts ``latent_mean`` representations, draws the visualizations that
are appropriate for the latent dimension, and computes subsidiary metrics:

* ``latent_dimension == 2``: direct plot ``latent_mean_0 x latent_mean_1``
  (no PCA/UMAP/t-SNE is used as a substitute).
* ``latent_dimension > 2``: PCA, UMAP and t-SNE (as enabled in
  ``projections.methods``), PCA explained variance plot and table.

It is shared by :class:`~vae_project.experiments.runner.ExperimentRunner` and
by the latent dimension comparison, so no pipeline logic is duplicated.

Labels are used only after training (coloring and subsidiary metrics).
Visual analysis is the primary evidence; metrics are subsidiary.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vae_project.config.models import ExperimentConfig
from vae_project.evaluation.clustering_metrics import compute_clustering_metrics
from vae_project.evaluation.latent_extraction import (
    LatentRepresentations,
    extract_latent_representations,
)
from vae_project.models.vae import VariationalAutoencoder
from vae_project.projections.pca import PCAProjection
from vae_project.projections.sampling import stratified_subsample
from vae_project.projections.tsne import TSNEProjection
from vae_project.projections.umap_projection import UMAPProjection
from vae_project.visualization.latent_plots import (
    plot_latent_space_direct_2d,
    plot_pca_explained_variance,
    plot_pca_projection,
    plot_tsne_projection,
    plot_umap_projection,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

LATENT_SOURCE = "latent_mean"
METRIC_COLUMNS = (
    "silhouette_score",
    "calinski_harabasz_index",
    "davies_bouldin_index",
    "trustworthiness",
)


@dataclass
class EvidenceDirs:
    """Output directories used by the evidence generation.

    Attributes:
        figures_latent: Directory for latent-space figures.
        latent_data: Directory for latent representations and sample indices.
        tables: Directory for CSV tables.
    """

    figures_latent: Path
    latent_data: Path
    tables: Path


@dataclass
class EvidenceFailure:
    """A failed evidence step, kept so that failures are never hidden.

    Attributes:
        operation: Name of the failed step (e.g. ``umap_projection``).
        error: Error type and message.
        probable_cause: Short human-readable hint.
    """

    operation: str
    error: str
    probable_cause: str

    def as_dict(self) -> dict[str, str]:
        """Return the failure as a JSON-serializable dictionary."""
        return {
            "operation": self.operation,
            "error": self.error,
            "probable_cause": self.probable_cause,
        }


@dataclass
class EvidenceResult:
    """Everything produced (or failed) by one evidence generation.

    Attributes:
        split_name: Split analyzed.
        latent_dimension: Latent dimension of the analyzed model.
        n_samples_total: Samples in the analyzed split.
        n_projection_samples: Samples used by PCA/UMAP/t-SNE (None for 2D).
        visualization_mode: ``direct_2d`` or ``projections``.
        latent_paths: Saved latent representation files (CSV and optional NPZ).
        sample_indices_path: Projection sample indices CSV (projections only).
        pca_variance_path: PCA explained variance CSV (projections only).
        projection_metrics_path: Subsidiary metrics CSV.
        figures: Saved figure files grouped by kind.
        pca_explained_variance: ``(pc1, pc2, cumulative)`` ratios, if PCA ran.
        metrics: Rows written to the projection metrics table.
        failures: Failed steps.
    """

    split_name: str
    latent_dimension: int
    n_samples_total: int
    visualization_mode: str
    n_projection_samples: int | None = None
    latent_paths: list[Path] = field(default_factory=list)
    sample_indices_path: Path | None = None
    pca_variance_path: Path | None = None
    projection_metrics_path: Path | None = None
    figures: dict[str, list[Path]] = field(default_factory=dict)
    pca_explained_variance: tuple[float, float, float] | None = None
    metrics: list[dict[str, object]] = field(default_factory=list)
    failures: list[EvidenceFailure] = field(default_factory=list)

    @property
    def used_sampling(self) -> bool:
        """Whether projections used a subsample of the split."""
        return (
            self.n_projection_samples is not None
            and self.n_projection_samples < self.n_samples_total
        )

    @property
    def is_complete(self) -> bool:
        """True when no evidence step failed."""
        return not self.failures

    def figure_paths(self, kind: str) -> list[Path]:
        """Return saved figure files of a kind (empty list if none)."""
        return list(self.figures.get(kind, []))

    def all_files(self) -> list[Path]:
        """Return every file produced, in a stable order."""
        files: list[Path] = list(self.latent_paths)
        for optional in (
            self.sample_indices_path,
            self.pca_variance_path,
            self.projection_metrics_path,
        ):
            if optional is not None:
                files.append(optional)
        for kind in sorted(self.figures):
            files.extend(self.figures[kind])
        return files


def _attempt(
    result: EvidenceResult,
    operation: str,
    probable_cause: str,
    action: Callable[[], T],
    raise_on_failure: bool,
) -> T | None:
    """Run one evidence step, recording (not hiding) any failure.

    Args:
        result: Result object receiving the failure record.
        operation: Step name.
        probable_cause: Hint stored with the failure.
        action: Zero-argument callable performing the step.
        raise_on_failure: Re-raise instead of recording the failure.

    Returns:
        The action's return value, or None when it failed.
    """
    try:
        return action()
    except Exception as exc:
        if raise_on_failure:
            raise
        plt.close("all")
        logger.error(
            "Evidence step failed: operation=%s, latent_dimension=%d, error=%s",
            operation,
            result.latent_dimension,
            exc,
            exc_info=True,
        )
        result.failures.append(
            EvidenceFailure(
                operation=operation,
                error=f"{type(exc).__name__}: {exc}",
                probable_cause=probable_cause,
            )
        )
        return None


def _metric_row(
    vectors: np.ndarray,
    labels: np.ndarray,
    space_name: str,
    space_type: str,
    scope: str,
    num_classes: int,
    config: ExperimentConfig,
    context: dict[str, object],
    original_space: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute one row of the projection metrics table.

    Unavailable metrics stay missing (None -> empty cell) and are listed in
    ``unavailable_metrics``; no artificial value is ever filled in.
    """
    metrics_config = config.metrics
    clustering = compute_clustering_metrics(
        latent_vectors=vectors,
        labels=labels,
        space_name=space_name,
        num_classes=num_classes,
        max_samples_silhouette=metrics_config.silhouette_sample_size,
        trustworthiness_n_neighbors=metrics_config.trustworthiness_n_neighbors,
        original_space=original_space if metrics_config.compute_trustworthiness else None,
    )
    row = clustering.as_dict()
    # K-Means is not computed here; drop columns that would always be empty.
    row.pop("kmeans_ari", None)
    row.pop("kmeans_nmi", None)
    unavailable = [name for name in METRIC_COLUMNS if row.get(name) is None]
    if original_space is None or not metrics_config.compute_trustworthiness:
        unavailable = [name for name in unavailable if name != "trustworthiness"]
        trustworthiness_note = "not_applicable"
    else:
        trustworthiness_note = "computed" if row["trustworthiness"] is not None else "failed"
    row.update(
        {
            "space_type": space_type,
            "scope": scope,
            "unavailable_metrics": ";".join(unavailable),
            "trustworthiness_status": trustworthiness_note,
        }
    )
    row.update(context)
    return row


def generate_latent_evidence(
    model: VariationalAutoencoder,
    data_loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    class_names: list[str],
    config: ExperimentConfig,
    dirs: EvidenceDirs,
    split_name: str = "val",
    run_id: str = "",
    raise_on_failure: bool = False,
) -> EvidenceResult:
    """Extract ``latent_mean`` and generate figures, tables and metrics.

    The model must already hold the weights of the best checkpoint.

    Args:
        model: Trained VAE (best checkpoint loaded).
        data_loader: Loader of the split to analyze (non-shuffled).
        device: Compute device.
        class_names: Class names, indexed by label.
        config: Experiment configuration (projections, metrics, plotting).
        dirs: Output directories.
        split_name: Split label stored in tables and figures.
        run_id: Run identifier stored in the latent representations.
        raise_on_failure: Raise on projection/plot/metric failures instead of
            recording them. Latent extraction failures always raise.

    Returns:
        EvidenceResult describing produced files, metrics and failures.
    """
    latent_dim = model.latent_dimension
    dataset_name = config.dataset.name.value
    seed = config.reproducibility.global_seed
    experiment_type = config.experiment_type
    is_quick = config.project.quick_mode
    num_classes = len(class_names)

    context: dict[str, object] = {
        "dataset": dataset_name,
        "split": split_name,
        "latent_dimension": latent_dim,
        "seed": seed,
        "experiment_type": experiment_type,
    }

    model.eval()
    reps = extract_latent_representations(
        model=model,
        data_loader=data_loader,
        device=device,
        split_name=split_name,
        class_names=class_names,
        dataset_name=dataset_name,
        is_quick_mode=is_quick,
        metadata={
            "run_id": run_id,
            "seed": seed,
            "experiment_type": experiment_type,
            "latent_source": LATENT_SOURCE,
        },
    )
    result = EvidenceResult(
        split_name=split_name,
        latent_dimension=latent_dim,
        n_samples_total=len(reps.labels),
        visualization_mode="direct_2d" if latent_dim == 2 else "projections",
    )
    result.latent_paths = reps.save(
        output_dir=dirs.latent_data, save_npz=config.artifacts.save_latent_npz
    )

    if latent_dim == 2:
        _direct_2d(result, reps, class_names, config, dirs, raise_on_failure)
    else:
        _projections(result, reps, class_names, config, dirs, context, raise_on_failure)

    _original_space_metrics(result, reps, num_classes, config, context, raise_on_failure)
    _save_metrics(result, dirs, raise_on_failure)
    return result


def _direct_2d(
    result: EvidenceResult,
    reps: LatentRepresentations,
    class_names: list[str],
    config: ExperimentConfig,
    dirs: EvidenceDirs,
    raise_on_failure: bool,
) -> None:
    """Draw ``latent_mean_0 x latent_mean_1`` directly (no projection)."""
    paths = _attempt(
        result,
        "direct_2d_plot",
        "matplotlib failed to render or write the figure (disk space or permissions).",
        lambda: plot_latent_space_direct_2d(
            latent_means=reps.latent_means,
            labels=reps.labels,
            class_names=class_names,
            output_dir=dirs.figures_latent,
            config=config.plotting,
            dataset_name=config.dataset.name.value,
            split_name=result.split_name,
            is_quick_mode=config.project.quick_mode,
            experiment_type=config.experiment_type,
        ),
        raise_on_failure,
    )
    if paths:
        result.figures["direct"] = paths


def _projections(
    result: EvidenceResult,
    reps: LatentRepresentations,
    class_names: list[str],
    config: ExperimentConfig,
    dirs: EvidenceDirs,
    context: dict[str, object],
    raise_on_failure: bool,
) -> None:
    """Run PCA/UMAP/t-SNE on a stratified sample of ``latent_mean``."""
    proj_config = config.projections
    sample_cfg = proj_config.sampling
    max_samples = sample_cfg.max_samples if sample_cfg.enabled else len(reps.labels)

    # The selection depends only on labels and seed, so every latent dimension
    # of the same split receives exactly the same sample indices.
    sample_vecs, sample_labels, selected = stratified_subsample(
        vectors=reps.latent_means,
        labels=reps.labels,
        max_samples=max_samples,
        seed=sample_cfg.seed,
    )
    result.n_projection_samples = len(sample_labels)

    indices_df = pd.DataFrame(
        {
            "sample_index": reps.sample_indices[selected],
            "label": sample_labels,
            "class_name": [class_names[int(label)] for label in sample_labels],
        }
    )
    for column, value in context.items():
        indices_df[column] = pd.Series([value] * len(indices_df), dtype=object)
    indices_df["n_projection_samples"] = len(sample_labels)
    indices_df["n_samples_total"] = result.n_samples_total
    indices_path = dirs.latent_data / "projection_sample_indices.csv"
    dirs.latent_data.mkdir(parents=True, exist_ok=True)
    indices_df.to_csv(indices_path, index=False)
    result.sample_indices_path = indices_path

    plot_kwargs: dict[str, Any] = {
        "labels": sample_labels,
        "output_dir": dirs.figures_latent,
        "config": config.plotting,
        "dataset_name": config.dataset.name.value,
        "split_name": result.split_name,
        "is_quick_mode": config.project.quick_mode,
    }
    label_kwargs: dict[str, Any] = {
        **plot_kwargs,
        "class_names": class_names,
        "n_total": result.n_samples_total,
        "experiment_type": config.experiment_type,
    }
    scope = "sample" if result.used_sampling else "full_split"

    def add_metrics(method: str, embedding: np.ndarray) -> None:
        row = _attempt(
            result,
            f"{method}_metrics",
            "metric computation failed (too few samples or classes in the sample).",
            lambda: _metric_row(
                vectors=embedding,
                labels=sample_labels,
                space_name=f"{result.split_name}_{method}_2d",
                space_type=f"{method}_2d",
                scope=scope,
                num_classes=len(class_names),
                config=config,
                context=context,
                original_space=sample_vecs,
            ),
            raise_on_failure,
        )
        if row is not None:
            result.metrics.append(row)

    if "pca" in proj_config.methods:
        pca_result = _attempt(
            result,
            "pca_projection",
            "PCA could not be fitted (fewer samples than components?).",
            lambda: PCAProjection(
                n_components=2, random_state=config.reproducibility.global_seed
            ).fit_transform(sample_vecs),
            raise_on_failure,
        )
        if pca_result is not None:
            result.pca_explained_variance = _explained_variance_summary(pca_result)
            variance_path = dirs.tables / "pca_explained_variance.csv"
            _attempt(
                result,
                "pca_explained_variance_table",
                "the tables directory is not writable.",
                lambda: pca_result.save_explained_variance_csv(
                    variance_path,
                    extra_columns={
                        "dataset": context["dataset"],
                        "split": context["split"],
                        "seed": context["seed"],
                        "experiment_type": context["experiment_type"],
                    },
                ),
                raise_on_failure,
            )
            if variance_path.exists():
                result.pca_variance_path = variance_path
            _record_figures(
                result,
                "pca",
                "pca_plot",
                lambda: plot_pca_projection(pca_result=pca_result, **label_kwargs),
                raise_on_failure,
            )
            _record_figures(
                result,
                "pca_explained_variance",
                "pca_explained_variance_plot",
                lambda: plot_pca_explained_variance(
                    pca_result=pca_result,
                    output_dir=dirs.figures_latent,
                    config=config.plotting,
                    dataset_name=config.dataset.name.value,
                    is_quick_mode=config.project.quick_mode,
                ),
                raise_on_failure,
            )
            add_metrics("pca", pca_result.projection_2d)

    if "umap" in proj_config.methods:
        umap_result = _attempt(
            result,
            "umap_projection",
            "umap-learn is missing or failed (check the log for numba errors).",
            lambda: UMAPProjection(proj_config.umap).fit_transform(sample_vecs),
            raise_on_failure,
        )
        if umap_result is not None:
            _record_figures(
                result,
                "umap",
                "umap_plot",
                lambda: plot_umap_projection(umap_result=umap_result, **label_kwargs),
                raise_on_failure,
            )
            add_metrics("umap", umap_result.projection_2d)

    if "tsne" in proj_config.methods:
        tsne_result = _attempt(
            result,
            "tsne_projection",
            "t-SNE failed (perplexity incompatible with the sample size?).",
            lambda: TSNEProjection(proj_config.tsne).fit_transform(sample_vecs),
            raise_on_failure,
        )
        if tsne_result is not None:
            _record_figures(
                result,
                "tsne",
                "tsne_plot",
                lambda: plot_tsne_projection(tsne_result=tsne_result, **label_kwargs),
                raise_on_failure,
            )
            add_metrics("tsne", tsne_result.projection_2d)


def _explained_variance_summary(pca_result) -> tuple[float, float, float]:  # type: ignore[no-untyped-def]
    """Return ``(pc1, pc2, cumulative)`` explained variance ratios."""
    ratios = pca_result.explained_variance_ratio
    first = float(ratios[0])
    second = float(ratios[1]) if len(ratios) > 1 else float("nan")
    cumulative = float(pca_result.cumulative_explained_variance[min(1, len(ratios) - 1)])
    return first, second, cumulative


def _record_figures(
    result: EvidenceResult,
    kind: str,
    operation: str,
    action: Callable[[], list[Path]],
    raise_on_failure: bool,
) -> None:
    """Run a plotting action and store its saved files under ``kind``."""
    paths = _attempt(
        result,
        operation,
        "matplotlib failed to render or write the figure (disk space or permissions).",
        action,
        raise_on_failure,
    )
    if paths:
        result.figures[kind] = paths


def _original_space_metrics(
    result: EvidenceResult,
    reps: LatentRepresentations,
    num_classes: int,
    config: ExperimentConfig,
    context: dict[str, object],
    raise_on_failure: bool,
) -> None:
    """Compute metrics in the original latent space (full split)."""
    row = _attempt(
        result,
        "original_latent_metrics",
        "metric computation failed on the original latent space.",
        lambda: _metric_row(
            vectors=reps.latent_means,
            labels=reps.labels,
            space_name=f"{result.split_name}_latent_original",
            space_type="latent_original",
            scope="full_split",
            num_classes=num_classes,
            config=config,
            context=context,
        ),
        raise_on_failure,
    )
    if row is not None:
        result.metrics.insert(0, row)


def _save_metrics(result: EvidenceResult, dirs: EvidenceDirs, raise_on_failure: bool) -> None:
    """Write ``projection_metrics.csv``."""
    if not result.metrics:
        return
    path = dirs.tables / "projection_metrics.csv"

    def write() -> Path:
        dirs.tables.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.metrics).to_csv(path, index=False)
        return path

    saved = _attempt(
        result,
        "projection_metrics_table",
        "the tables directory is not writable.",
        write,
        raise_on_failure,
    )
    if saved is not None:
        result.projection_metrics_path = saved
        logger.info("Subsidiary metrics saved: %s", saved)
