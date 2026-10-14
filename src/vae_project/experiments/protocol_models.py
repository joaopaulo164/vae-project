"""Data model of the factorial protocol: grid combinations and run records.

A *record* summarizes one finished grid combination. It is always rebuilt from
the files on disk (``metrics.json`` and the figure directories), never from
in-memory state. A freshly trained combination and a combination restored by
``--resume`` therefore produce identical rows of the factorial table.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from vae_project.config.models import STANDARD_TOPOLOGIES, ExperimentConfig, TopologyConfig
from vae_project.experiments.experiment_result import (
    FIGURE_CATEGORY_GENERATED,
    FIGURE_CATEGORY_RECONSTRUCTIONS,
    FIGURE_CATEGORY_TRAINING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
)
from vae_project.utils.json_io import read_json

logger = logging.getLogger(__name__)

GRID_DIRECTORY = "grid"
SHARED_DIRECTORY = "shared"
SELECTION_DIRECTORY = "selection"
FINAL_EVALUATION_DIRECTORY = "final_evaluation"
SUPERVISED_DIRECTORY = "supervised_extension"
GENERATED_CONFIGS_DIRECTORY = "generated_configs"

METRICS_FILENAME = "metrics.json"
SUMMARY_FILENAME = "run_summary.md"
MANIFEST_FILENAME = "artifact_manifest.json"
CONFIG_FILENAME = "config_resolved.yaml"
CHECKPOINT_FILENAME = "best_model.pt"
FIGURE_SUFFIXES = (".png", ".svg")
LIST_SEPARATOR = ";"

LATENT_SPACE_DIRECTORY = "latent_space"
FIGURE_DIRECTORIES = ("training", "reconstructions", "generated_samples", LATENT_SPACE_DIRECTORY)
DIRECT_PLOT_STEM = "latent_space_direct_2d"
PCA_PLOT_PREFIX = "latent_space_pca_"
UMAP_PLOT_PREFIX = "latent_space_umap_"
TSNE_PLOT_PREFIX = "latent_space_tsne_"
PCA_VARIANCE_PLOT_PREFIX = "pca_explained_variance_"

# Required file sets used to decide whether a combination is really complete.
BASE_REQUIRED_FILES = (
    "checkpoints/best_model.pt",
    "checkpoints/checkpoint_metadata.json",
    "tables/training_history.csv",
    "tables/training_history.json",
    "figures/training/training_total_loss.png",
    "figures/training/training_loss_components.png",
    CONFIG_FILENAME,
    METRICS_FILENAME,
    SUMMARY_FILENAME,
    MANIFEST_FILENAME,
)
FULL_EVIDENCE_REQUIRED_FILES = (
    "figures/reconstructions/reconstruction_comparison.png",
    "figures/generated_samples/generated_samples.png",
    "latent_data/latent_representations.csv",
)

COMBINATION_COLUMNS = (
    "dataset",
    "experiment_type",
    "topology_name",
    "hidden_dimensions",
    "latent_dimension",
    "seed",
    "parameter_count",
    "best_epoch",
    "epochs_trained",
    "validation_reconstruction_loss",
    "validation_kl_divergence",
    "validation_classification_loss",
    "validation_total_loss",
    "training_duration_seconds",
    "total_duration_seconds",
    "checkpoint_path",
    "run_directory",
    "training_figure_paths",
    "reconstruction_figure_paths",
    "generated_samples_paths",
    "direct_latent_plot_path",
    "pca_plot_path",
    "umap_plot_path",
    "tsne_plot_path",
    "pca_explained_variance_plot_path",
    "pca_component_1_explained_variance",
    "pca_component_2_explained_variance",
    "pca_cumulative_explained_variance",
    "projection_metrics_path",
    "run_summary_path",
    "manifest_path",
    "execution_status",
    "warnings_count",
    "errors_count",
)


class ProtocolError(Exception):
    """Raised when the factorial protocol cannot start or continue."""


@dataclass(frozen=True)
class FactorialCombination:
    """One cell of the topology x latent dimension grid.

    Attributes:
        topology_name: Standard topology name (small, medium, deep).
        hidden_dimensions: Encoder hidden layer sizes of the topology.
        latent_dimension: Latent dimension of the model.
    """

    topology_name: str
    hidden_dimensions: tuple[int, ...]
    latent_dimension: int

    @property
    def key(self) -> str:
        """Stable identifier such as ``small/latent_dim_2``."""
        return f"{self.topology_name}/latent_dim_{self.latent_dimension}"

    @property
    def relative_directory(self) -> Path:
        """Directory of the combination relative to the protocol run."""
        return Path(GRID_DIRECTORY) / self.topology_name / f"latent_dim_{self.latent_dimension}"

    @property
    def topology(self) -> TopologyConfig:
        """Topology configuration of the combination."""
        return TopologyConfig(
            name=self.topology_name, hidden_dimensions=list(self.hidden_dimensions)
        )


def build_factorial_grid(config: ExperimentConfig) -> list[FactorialCombination]:
    """Build the full factorial grid in a deterministic order.

    The order is topology-major (small, medium, deep, as configured) and
    dimension-minor, so the default grid is small x [2, 4, 8, 16, 32], medium x
    [2, 4, 8, 16, 32], deep x [2, 4, 8, 16, 32] (15 combinations; the count always
    follows the configured lists).

    Args:
        config: Configuration with ``factorial_comparison`` filled in.

    Returns:
        The combinations to train.

    Raises:
        ProtocolError: If a topology is unknown or a combination is duplicated.
    """
    factorial = config.factorial_comparison
    combinations: list[FactorialCombination] = []
    for topology_name in factorial.topologies:
        if topology_name not in STANDARD_TOPOLOGIES:
            raise ProtocolError(f"Unknown topology {topology_name!r} in the factorial grid.")
        hidden = tuple(STANDARD_TOPOLOGIES[topology_name])
        for latent_dimension in factorial.latent_dimensions:
            combinations.append(FactorialCombination(topology_name, hidden, latent_dimension))
    keys = [combination.key for combination in combinations]
    if len(set(keys)) != len(keys):
        raise ProtocolError(f"The factorial grid contains duplicated combinations: {keys}.")
    return combinations


@dataclass
class CombinationRecord:
    """Validation-side summary of one grid combination (rebuilt from disk).

    Path fields are relative to the protocol run directory and use forward
    slashes. Values that do not apply stay ``None``.
    """

    dataset: str
    experiment_type: str
    topology_name: str
    hidden_dimensions: list[int]
    latent_dimension: int
    seed: int
    status: str
    run_directory: str
    parameter_count: int | None = None
    best_epoch: int | None = None
    epochs_trained: int | None = None
    validation_reconstruction_loss: float | None = None
    validation_kl_divergence: float | None = None
    validation_classification_loss: float | None = None
    validation_total_loss: float | None = None
    training_duration_seconds: float | None = None
    total_duration_seconds: float | None = None
    checkpoint_path: str | None = None
    figure_files: dict[str, list[str]] = field(default_factory=dict)
    latent_plots: dict[str, str | None] = field(default_factory=dict)
    projection_metrics_path: str | None = None
    run_summary_path: str | None = None
    manifest_path: str | None = None
    pca_explained_variance: tuple[float | None, float | None, float | None] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable identifier such as ``small/latent_dim_2``."""
        return f"{self.topology_name}/latent_dim_{self.latent_dimension}"

    def to_row(self) -> dict[str, object]:
        """Return the row of ``factorial_comparison.csv`` (missing -> None)."""
        variance = self.pca_explained_variance or (None, None, None)
        return {
            "dataset": self.dataset,
            "experiment_type": self.experiment_type,
            "topology_name": self.topology_name,
            "hidden_dimensions": str(self.hidden_dimensions),
            "latent_dimension": self.latent_dimension,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
            "best_epoch": self.best_epoch,
            "epochs_trained": self.epochs_trained,
            "validation_reconstruction_loss": self.validation_reconstruction_loss,
            "validation_kl_divergence": self.validation_kl_divergence,
            "validation_classification_loss": self.validation_classification_loss,
            "validation_total_loss": self.validation_total_loss,
            "training_duration_seconds": self.training_duration_seconds,
            "total_duration_seconds": self.total_duration_seconds,
            "checkpoint_path": self.checkpoint_path,
            "run_directory": self.run_directory,
            "training_figure_paths": LIST_SEPARATOR.join(
                self.figure_files.get(FIGURE_CATEGORY_TRAINING, [])
            ),
            "reconstruction_figure_paths": LIST_SEPARATOR.join(
                self.figure_files.get(FIGURE_CATEGORY_RECONSTRUCTIONS, [])
            ),
            "generated_samples_paths": LIST_SEPARATOR.join(
                self.figure_files.get(FIGURE_CATEGORY_GENERATED, [])
            ),
            "direct_latent_plot_path": self.latent_plots.get("direct"),
            "pca_plot_path": self.latent_plots.get("pca"),
            "umap_plot_path": self.latent_plots.get("umap"),
            "tsne_plot_path": self.latent_plots.get("tsne"),
            "pca_explained_variance_plot_path": self.latent_plots.get("pca_explained_variance"),
            "pca_component_1_explained_variance": variance[0],
            "pca_component_2_explained_variance": variance[1],
            "pca_cumulative_explained_variance": variance[2],
            "projection_metrics_path": self.projection_metrics_path,
            "run_summary_path": self.run_summary_path,
            "manifest_path": self.manifest_path,
            "execution_status": self.status,
            "warnings_count": len(self.warnings),
            "errors_count": len(self.errors),
        }


def is_finite_number(value: object) -> bool:
    """Return True for real, finite, non-boolean numbers."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _relative(protocol_dir: Path, path: Path) -> str:
    """Return ``path`` relative to the protocol directory (forward slashes)."""
    return path.relative_to(protocol_dir).as_posix()


def figure_files(protocol_dir: Path, run_dir: Path, category: str) -> list[str]:
    """List the PNG/SVG files of a figure category (relative paths)."""
    directory = run_dir / "figures" / category
    if not directory.is_dir():
        return []
    files = sorted(path for path in directory.iterdir() if path.suffix in FIGURE_SUFFIXES)
    return [_relative(protocol_dir, path) for path in files]


def _latent_plot(protocol_dir: Path, run_dir: Path, prefix: str) -> str | None:
    """Return the PNG latent-space figure whose name starts with ``prefix``."""
    directory = run_dir / "figures" / LATENT_SPACE_DIRECTORY
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"{prefix}*.png"))
    return _relative(protocol_dir, matches[0]) if matches else None


def _optional_file(protocol_dir: Path, path: Path) -> str | None:
    """Return the relative path of ``path`` when the file exists."""
    return _relative(protocol_dir, path) if path.is_file() else None


def _variance_tuple(
    payload: dict[str, object] | None,
) -> tuple[float | None, float | None, float | None] | None:
    """Convert the ``pca_explained_variance`` block of metrics.json."""
    if not payload:
        return None
    return (
        _as_float(payload.get("component_1")),
        _as_float(payload.get("component_2")),
        _as_float(payload.get("cumulative_2_components")),
    )


def _as_float(value: object) -> float | None:
    """Return a finite float or None."""
    return float(value) if is_finite_number(value) else None  # type: ignore[arg-type]


def _as_int(value: object) -> int | None:
    """Return an int or None."""
    return int(value) if is_finite_number(value) else None  # type: ignore[call-overload]


def _load_metrics(run_dir: Path) -> dict[str, object] | None:
    """Read ``metrics.json`` of a run, or None when unreadable."""
    metrics_path = run_dir / METRICS_FILENAME
    if not metrics_path.is_file():
        return None
    try:
        payload = read_json(metrics_path)
    except (json.JSONDecodeError, OSError) as error:
        logger.warning("Unreadable metrics file %s: %s", metrics_path, error)
        return None
    return payload if isinstance(payload, dict) else None


def _int_list(value: object) -> list[int]:
    """Convert a JSON list of numbers into a list of ints."""
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _required_int(value: object) -> int:
    """Convert a JSON number into an int (0 when absent)."""
    return int(value) if is_finite_number(value) else 0  # type: ignore[call-overload]


def _dict_list(value: object) -> list[dict[str, str]]:
    """Convert a JSON list of objects into a list of string dictionaries."""
    if not isinstance(value, list):
        return []
    return [{str(key): str(item) for key, item in entry.items()} for entry in value]


def read_combination_record(
    protocol_dir: Path,
    combination: FactorialCombination,
) -> CombinationRecord | None:
    """Rebuild the record of a combination from its run directory.

    Args:
        protocol_dir: Protocol run directory.
        combination: Grid cell to read.

    Returns:
        The record, or None when the combination has no readable
        ``metrics.json`` (never started or interrupted before the reports).
    """
    return read_run_record(Path(protocol_dir), Path(protocol_dir) / combination.relative_directory)


def failed_record(
    config: ExperimentConfig,
    combination: FactorialCombination,
    message: str,
) -> CombinationRecord:
    """Create the record of a combination that failed before writing any metrics."""
    return CombinationRecord(
        dataset=config.dataset.name.value,
        experiment_type="unsupervised",
        topology_name=combination.topology_name,
        hidden_dimensions=list(combination.hidden_dimensions),
        latent_dimension=combination.latent_dimension,
        seed=config.reproducibility.global_seed,
        status=STATUS_FAILED,
        run_directory=combination.relative_directory.as_posix(),
        errors=[
            {
                "operation": "combination_execution",
                "error": message,
                "probable_cause": "the combination raised before its reports were written.",
            }
        ],
    )


def read_run_record(protocol_dir: Path, run_dir: Path) -> CombinationRecord | None:
    """Rebuild a record from any run directory that holds a ``metrics.json``.

    Args:
        protocol_dir: Protocol run directory (paths are relative to it).
        run_dir: Directory of the run (a grid cell or the supervised training).

    Returns:
        The record, or None when ``metrics.json`` is missing or unreadable.
    """
    protocol_dir = Path(protocol_dir)
    run_dir = Path(run_dir)
    payload = _load_metrics(run_dir)
    if payload is None:
        return None
    checkpoint = payload.get("best_checkpoint")
    checkpoint_path = (
        _optional_file(protocol_dir, run_dir / str(checkpoint)) if checkpoint else None
    )
    variance = payload.get("pca_explained_variance")
    return CombinationRecord(
        dataset=str(payload.get("dataset")),
        experiment_type=str(payload.get("experiment_type")),
        topology_name=str(payload.get("topology_name")),
        hidden_dimensions=_int_list(payload.get("hidden_dimensions")),
        latent_dimension=_required_int(payload.get("latent_dimension")),
        seed=_required_int(payload.get("seed")),
        status=str(payload.get("execution_status")),
        run_directory=_relative(protocol_dir, run_dir),
        parameter_count=_as_int(payload.get("parameter_count")),
        best_epoch=_as_int(payload.get("best_epoch")),
        epochs_trained=_as_int(payload.get("epochs_trained")),
        validation_reconstruction_loss=_as_float(payload.get("validation_reconstruction_loss")),
        validation_kl_divergence=_as_float(payload.get("validation_kl_divergence")),
        validation_classification_loss=_as_float(payload.get("validation_classification_loss")),
        validation_total_loss=_as_float(payload.get("validation_total_loss")),
        training_duration_seconds=_as_float(payload.get("training_time_seconds")),
        total_duration_seconds=_as_float(payload.get("total_time_seconds")),
        checkpoint_path=checkpoint_path,
        figure_files={
            category: figure_files(protocol_dir, run_dir, category)
            for category in FIGURE_DIRECTORIES
        },
        latent_plots={
            "direct": _latent_plot(protocol_dir, run_dir, DIRECT_PLOT_STEM),
            "pca": _latent_plot(protocol_dir, run_dir, PCA_PLOT_PREFIX),
            "umap": _latent_plot(protocol_dir, run_dir, UMAP_PLOT_PREFIX),
            "tsne": _latent_plot(protocol_dir, run_dir, TSNE_PLOT_PREFIX),
            "pca_explained_variance": _latent_plot(protocol_dir, run_dir, PCA_VARIANCE_PLOT_PREFIX),
        },
        projection_metrics_path=_optional_file(
            protocol_dir, run_dir / "tables" / "projection_metrics.csv"
        ),
        run_summary_path=_optional_file(protocol_dir, run_dir / SUMMARY_FILENAME),
        manifest_path=_optional_file(protocol_dir, run_dir / MANIFEST_FILENAME),
        pca_explained_variance=_variance_tuple(variance if isinstance(variance, dict) else None),
        warnings=[str(item) for item in _string_list(payload.get("warnings"))],
        errors=_dict_list(payload.get("errors")),
    )


def _string_list(value: object) -> list[str]:
    """Convert a JSON list into a list of strings."""
    return [str(item) for item in value] if isinstance(value, list) else []


def missing_run_files(
    run_dir: Path,
    status: str,
    require_full_evidence: bool = True,
) -> list[str]:
    """List the minimum expected files that are absent from a run directory.

    A run counts as complete only when status, checkpoint, metrics, summary,
    manifest and the minimal evidence exist; an existing directory is never
    sufficient. ``completed`` runs additionally need the reconstruction,
    generated-sample and latent evidence.

    Args:
        run_dir: Directory of the run.
        status: Execution status stored in its ``metrics.json``.
        require_full_evidence: Also require reconstruction, generated-sample and
            latent evidence for ``completed`` runs.

    Returns:
        Missing relative paths (empty when everything is present).
    """
    required = list(BASE_REQUIRED_FILES)
    if status == STATUS_COMPLETED and require_full_evidence:
        required.extend(FULL_EVIDENCE_REQUIRED_FILES)
    return [name for name in required if not (Path(run_dir) / name).is_file()]


def missing_artifacts(
    protocol_dir: Path,
    combination: FactorialCombination,
    record: CombinationRecord,
    require_full_evidence: bool = True,
) -> list[str]:
    """List the minimum expected files that are absent from a grid combination."""
    run_dir = Path(protocol_dir) / combination.relative_directory
    return missing_run_files(run_dir, record.status, require_full_evidence)


def record_matches_combination(
    record: CombinationRecord,
    combination: FactorialCombination,
    config: ExperimentConfig,
) -> bool:
    """Check that a stored record belongs to the combination and configuration."""
    return (
        record.topology_name == combination.topology_name
        and record.hidden_dimensions == list(combination.hidden_dimensions)
        and record.latent_dimension == combination.latent_dimension
        and record.seed == config.reproducibility.global_seed
        and record.dataset == config.dataset.name.value
        and record.experiment_type == "unsupervised"
    )


def is_reusable(
    protocol_dir: Path,
    combination: FactorialCombination,
    record: CombinationRecord | None,
    config: ExperimentConfig,
) -> bool:
    """Tell whether a combination on disk is complete, compatible and reusable."""
    if record is None or record.status not in (STATUS_COMPLETED, STATUS_PARTIAL):
        return False
    if not record_matches_combination(record, combination, config):
        return False
    full_evidence = config.factorial_comparison.generate_full_evidence
    return not missing_artifacts(protocol_dir, combination, record, full_evidence)
