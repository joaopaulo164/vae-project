"""Latent dimension comparison experiment.

Trains the same, previously selected topology at several latent dimensions and
generates the complete evidence of each dimension right after its training.
Every dimension is one call of the common service
:func:`vae_project.experiments.single_experiment.execute_single_experiment`,
so each ``latent_dim_<N>`` directory holds the same four figure categories as
any other pipeline:

    figures/training, figures/reconstructions, figures/generated_samples,
    figures/latent_space

Every dimension is an independent training run: a new model, optimizer,
scheduler and early stopping are created and the seed is applied again. The
latent dimension is the only experimental variable; data splits, seed,
hyperparameters, topology, projection parameters and sampling are shared.

The topology is not searched here (use ``select-topology`` first). The
supervised component stays disabled by default; labels are used only after
training to color plots and compute subsidiary metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from vae_project.config.models import (
    STANDARD_TOPOLOGIES,
    ExperimentConfig,
    TopologyConfig,
)
from vae_project.config.validation import validate_comparison_dimensions
from vae_project.data.splits import DataSplits
from vae_project.experiments.experiment_result import (
    STATUS_FAILED,
    EvidenceProfile,
    SingleExperimentResult,
)
from vae_project.experiments.single_experiment import (
    SingleExperimentOptions,
    execute_single_experiment,
    prepare_experiment_data,
)
from vae_project.reporting.comparison_report import (
    DimensionRunRecord,
    build_comparison_table,
    write_comparison_summary,
)
from vae_project.utils.logging import setup_logging
from vae_project.utils.paths import ensure_dir, ensure_run_dir_is_free, ensure_smoke_test_dir

logger = logging.getLogger(__name__)

ANALYZED_SPLIT = "val"
LATENT_REPRESENTATIONS_FILENAME = "latent_representations.csv"
PROJECTION_METRICS_FILENAME = "projection_metrics.csv"
LATENT_FIGURE_PATH_KEYS = (
    ("direct", "direct_plot"),
    ("pca", "pca_plot"),
    ("umap", "umap_plot"),
    ("tsne", "tsne_plot"),
    ("pca_explained_variance", "pca_explained_variance_plot"),
)


class LatentDimensionComparisonError(Exception):
    """Raised when the comparison cannot start or no dimension succeeded."""


def resolve_topology(config: ExperimentConfig) -> TopologyConfig:
    """Return the fixed topology used by every model of the comparison.

    The topology comes from ``model.topology``. ``fixed_topology`` must name it,
    which guards against a config whose fixed topology and hidden dimensions
    disagree. No topology search is performed here.

    Args:
        config: Experiment configuration.

    Returns:
        TopologyConfig shared by all dimensions.

    Raises:
        LatentDimensionComparisonError: If the names do not match, or the
            standard name does not match the standard hidden dimensions.
    """
    fixed_name = config.latent_dimension_comparison.fixed_topology
    topology = config.model.topology
    if topology.name != fixed_name:
        raise LatentDimensionComparisonError(
            f"latent_dimension_comparison.fixed_topology='{fixed_name}' does not match "
            f"model.topology.name='{topology.name}'. The comparison uses one fixed topology."
        )
    standard = STANDARD_TOPOLOGIES.get(fixed_name)
    if standard is not None and list(topology.hidden_dimensions) != standard:
        logger.warning(
            "Topology '%s' uses hidden_dimensions=%s, which differs from the project "
            "standard %s. Proceeding with the configured values.",
            fixed_name,
            topology.hidden_dimensions,
            standard,
        )
    return TopologyConfig(name=topology.name, hidden_dimensions=list(topology.hidden_dimensions))


def _first_table(result: SingleExperimentResult, filename: str) -> Path | None:
    """Return the table or data file named ``filename`` produced by the run."""
    for path in result.table_paths:
        if path.name == filename:
            return path
    return None


def result_to_record(result: SingleExperimentResult, base_dir: Path) -> DimensionRunRecord:
    """Convert a service result into the row model of the comparison report.

    Args:
        result: Outcome of one dimension, rooted at its own directory.
        base_dir: Comparison directory; ``paths`` become relative to it.

    Returns:
        DimensionRunRecord used by the comparison CSV and summary.
    """
    run_dir = result.run_directory
    if run_dir is None:
        raise LatentDimensionComparisonError("the execution result has no run directory")
    record = DimensionRunRecord(
        dataset=result.dataset,
        experiment_type=result.experiment_type,
        topology_name=result.topology_name,
        hidden_dimensions=list(result.hidden_dimensions),
        latent_dimension=result.latent_dimension,
        seed=result.seed,
        status=result.status,
        split=result.split,
        device=result.device,
        quick_mode=result.quick_mode,
        synthetic_data=result.synthetic_data,
        parameter_count=result.parameter_count,
        best_epoch=result.best_epoch,
        epochs_trained=result.epochs_trained,
        val_reconstruction_loss=result.validation_reconstruction_loss,
        val_kl_divergence=result.validation_kl_divergence,
        val_classification_loss=result.validation_classification_loss,
        val_total_loss=result.validation_total_loss,
        training_time_seconds=result.training_duration_seconds,
        visualization_mode=result.visualization_mode,
        n_samples_total=result.n_samples_total,
        n_projection_samples=result.n_projection_samples,
        used_sampling=result.used_sampling,
        pca_explained_variance=result.pca_explained_variance,
        projection_metrics=list(result.projection_metrics),
        figures=[path.relative_to(run_dir).as_posix() for path in result.all_figure_files()],
        tables=result.relative_tables(run_dir),
        failures=list(result.errors),
        warnings=list(result.warnings),
    )
    record.paths = _record_paths(result, base_dir)
    return record


def _relative_to(path: Path | None, base_dir: Path) -> str | None:
    """Return ``path`` relative to ``base_dir`` with forward slashes (None-safe)."""
    return None if path is None else path.relative_to(base_dir).as_posix()


def _record_paths(result: SingleExperimentResult, base_dir: Path) -> dict[str, str | None]:
    """Build the ``paths`` mapping of a record (relative to the comparison dir)."""
    paths: dict[str, str | None] = {
        "checkpoint": _relative_to(result.best_checkpoint_path, base_dir),
        "latent_representations": _relative_to(
            _first_table(result, LATENT_REPRESENTATIONS_FILENAME), base_dir
        ),
        "projection_metrics": _relative_to(
            _first_table(result, PROJECTION_METRICS_FILENAME), base_dir
        ),
        "run_summary": _relative_to(result.run_summary_path, base_dir),
        "artifact_manifest": _relative_to(result.manifest_path, base_dir),
        "log": _relative_to(result.log_path, base_dir),
    }
    for kind, key in LATENT_FIGURE_PATH_KEYS:
        paths[key] = _relative_to(result.latent_figure(kind), base_dir)
    return paths


def run_latent_dimension_comparison(
    config: ExperimentConfig,
    output_dir: Path,
    dimensions: list[int] | None = None,
    splits: DataSplits | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Train each latent dimension independently and generate its evidence.

    Holds constant: dataset, topology, seed, learning rate, batch size,
    max epochs, early stopping, loss definition, data splits, projection
    parameters and sampling.

    A failure in one dimension is recorded and does not stop the others; failed
    dimensions never take part in the best-dimension statement.

    ``output_dir`` is the directory of *this run*. It must be new or empty: an
    existing directory that already holds artifacts is rejected unless
    ``overwrite`` is True, so a finished run is never replaced silently.

    Args:
        config: Base experiment configuration.
        output_dir: Run directory for comparison artifacts.
        dimensions: Latent dimensions to compare. Defaults to
            ``config.latent_dimension_comparison.dimensions``.
        splits: Optional pre-built data splits. They are accepted **only for
            synthetic data** (tests and smoke tests, no downloads): the run is
            then labelled synthetic in every report and ``output_dir`` must
            contain ``smoke_test`` or ``synthetic`` in its path. When None, the
            configured real dataset is loaded.
        overwrite: Allow writing into a non-empty ``output_dir``. Files with the
            same name are replaced; nothing else is removed.

    Returns:
        Consolidated comparison table sorted by ``latent_dimension``.

    Raises:
        LatentDimensionComparisonError: If the comparison is disabled, the
            dimensions are invalid, the topology is inconsistent, or no
            dimension finished training.
        ArtifactError: If ``output_dir`` already holds artifacts and
            ``overwrite`` is False, or a synthetic run targets a directory
            without a ``smoke_test``/``synthetic`` marker.
    """
    if not config.latent_dimension_comparison.enabled:
        raise LatentDimensionComparisonError(
            "latent_dimension_comparison.enabled is false; nothing to run."
        )

    compare_dims = list(
        dimensions if dimensions is not None else config.latent_dimension_comparison.dimensions
    )
    dimension_errors = validate_comparison_dimensions(compare_dims)
    if dimension_errors:
        raise LatentDimensionComparisonError("; ".join(dimension_errors))

    topology = resolve_topology(config)
    if splits is not None:
        ensure_smoke_test_dir(Path(output_dir))
    ensure_run_dir_is_free(Path(output_dir), overwrite=overwrite)
    output_dir = ensure_dir(Path(output_dir))
    setup_logging(config.logging, log_file=output_dir / "latent_dimension_comparison.log")

    # One split and one set of loaders for all dimensions.
    data = prepare_experiment_data(config, splits)

    records: list[DimensionRunRecord] = []
    for latent_dim in compare_dims:
        result = execute_single_experiment(
            config=config,
            topology=topology,
            latent_dimension=latent_dim,
            supervised=config.supervised_component.enabled,
            evidence_profile=EvidenceProfile.FULL,
            output_directory=output_dir / f"latent_dim_{latent_dim}",
            data=data,
            options=SingleExperimentOptions(run_id=f"latent_dim_{latent_dim}"),
        )
        records.append(result_to_record(result, output_dir))
        logger.info("latent_dim=%d finished with status=%s", latent_dim, result.status)

    table = build_comparison_table(records)
    output_csv = output_dir / "latent_dimension_comparison.csv"
    write_comparison_summary(records, output_dir)
    table.to_csv(output_csv, index=False)
    logger.info("Latent dimension comparison saved: %s", output_csv)

    if all(record.status == STATUS_FAILED for record in records):
        raise LatentDimensionComparisonError(
            f"All latent dimensions failed for dataset={config.dataset.name.value}. "
            f"See {output_dir / 'latent_dimension_comparison_summary.md'} and the "
            "per-dimension logs."
        )
    return table
