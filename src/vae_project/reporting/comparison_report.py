"""Consolidated comparison report generation across experiments.

Besides the original topology/dimension consolidation helper, this module
holds the per-dimension record of the latent dimension comparison and the
writers for the consolidated table, the per-dimension ``run_summary.md`` and
the ``latent_dimension_comparison_summary.md``.

No scientific conclusion is generated here: the reports only state numbers,
paths and failures, and tell the reader how to interpret the figures.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

DATA_ORIGIN_REAL = "real"
DATA_ORIGIN_SYNTHETIC = "synthetic"
SYNTHETIC_NOTICE = (
    "**SYNTHETIC DATA: this run uses artificial data for testing/smoke-testing only. "
    "Its numbers and figures are NOT scientific results and must not be reported "
    "or mixed with the MNIST/Fashion-MNIST comparison.**"
)

COMPARISON_CSV_COLUMNS = (
    "dataset",
    "data_origin",
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
    "training_time",
    "checkpoint_path",
    "latent_representations_path",
    "direct_plot_path",
    "pca_plot_path",
    "umap_plot_path",
    "tsne_plot_path",
    "pca_explained_variance_plot_path",
    "pca_component_1_explained_variance",
    "pca_component_2_explained_variance",
    "pca_cumulative_explained_variance",
    "projection_metrics_path",
    "run_summary_path",
    "artifact_manifest_path",
    "execution_status",
    "error_summary",
)

PATH_KEYS = (
    "checkpoint",
    "latent_representations",
    "direct_plot",
    "pca_plot",
    "umap_plot",
    "tsne_plot",
    "pca_explained_variance_plot",
    "projection_metrics",
    "run_summary",
    "artifact_manifest",
    "log",
)


@dataclass
class DimensionRunRecord:
    """Outcome of one independent latent dimension training and analysis.

    All paths in ``paths`` are relative to the comparison directory and use
    forward slashes. ``figures`` and ``tables`` are relative to the dimension
    directory.
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
    parameter_count: int | None = None
    best_epoch: int | None = None
    epochs_trained: int | None = None
    val_reconstruction_loss: float | None = None
    val_kl_divergence: float | None = None
    val_classification_loss: float | None = None
    val_total_loss: float | None = None
    training_time_seconds: float | None = None
    visualization_mode: str | None = None
    n_samples_total: int | None = None
    n_projection_samples: int | None = None
    used_sampling: bool = False
    pca_explained_variance: tuple[float, float, float] | None = None
    projection_metrics: list[dict[str, object]] = field(default_factory=list)
    paths: dict[str, str | None] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def data_origin(self) -> str:
        """'synthetic' for test/smoke-test data, otherwise 'real'."""
        return DATA_ORIGIN_SYNTHETIC if self.synthetic_data else DATA_ORIGIN_REAL

    @property
    def has_valid_training(self) -> bool:
        """True when training finished and a best checkpoint was evaluated."""
        return self.status != STATUS_FAILED and self.val_total_loss is not None

    def error_summary(self) -> str | None:
        """Return failed operations as a single line, or None."""
        if not self.failures:
            return None
        return "; ".join(f"{item['operation']}: {item['error']}" for item in self.failures)


def _pca_value(record: DimensionRunRecord, index: int) -> float | None:
    """Return a PCA explained variance value, or None when not applicable."""
    if record.pca_explained_variance is None:
        return None
    value = record.pca_explained_variance[index]
    return None if math.isnan(value) else value


def record_to_row(record: DimensionRunRecord) -> dict[str, object]:
    """Convert a record into one row of ``latent_dimension_comparison.csv``.

    Values that do not apply (e.g. PCA on the 2D model) are ``None`` and thus
    empty cells; zero is never used for "not applicable".
    """
    paths = record.paths
    return {
        "dataset": record.dataset,
        "data_origin": record.data_origin,
        "experiment_type": record.experiment_type,
        "topology_name": record.topology_name,
        "hidden_dimensions": str(record.hidden_dimensions),
        "latent_dimension": record.latent_dimension,
        "seed": record.seed,
        "parameter_count": record.parameter_count,
        "best_epoch": record.best_epoch,
        "epochs_trained": record.epochs_trained,
        "validation_reconstruction_loss": record.val_reconstruction_loss,
        "validation_kl_divergence": record.val_kl_divergence,
        "validation_classification_loss": record.val_classification_loss,
        "validation_total_loss": record.val_total_loss,
        "training_time": record.training_time_seconds,
        "checkpoint_path": paths.get("checkpoint"),
        "latent_representations_path": paths.get("latent_representations"),
        "direct_plot_path": paths.get("direct_plot"),
        "pca_plot_path": paths.get("pca_plot"),
        "umap_plot_path": paths.get("umap_plot"),
        "tsne_plot_path": paths.get("tsne_plot"),
        "pca_explained_variance_plot_path": paths.get("pca_explained_variance_plot"),
        "pca_component_1_explained_variance": _pca_value(record, 0),
        "pca_component_2_explained_variance": _pca_value(record, 1),
        "pca_cumulative_explained_variance": _pca_value(record, 2),
        "projection_metrics_path": paths.get("projection_metrics"),
        "run_summary_path": paths.get("run_summary"),
        "artifact_manifest_path": paths.get("artifact_manifest"),
        "execution_status": record.status,
        "error_summary": record.error_summary(),
    }


def build_comparison_table(records: list[DimensionRunRecord]) -> pd.DataFrame:
    """Build the consolidated table, sorted by latent dimension."""
    rows = [record_to_row(record) for record in records]
    table = pd.DataFrame(rows, columns=list(COMPARISON_CSV_COLUMNS))
    return table.sort_values("latent_dimension").reset_index(drop=True)


def select_lowest_validation_loss(records: list[DimensionRunRecord]) -> DimensionRunRecord | None:
    """Return the record with the lowest validation total loss.

    Failed dimensions never take part in the selection.
    """
    valid = [record for record in records if record.has_valid_training]
    if not valid:
        return None
    return min(valid, key=lambda record: record.val_total_loss)  # type: ignore[arg-type,return-value]


def _fmt(value: object, digits: int = 6) -> str:
    """Format a possibly missing value for Markdown."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def _bullets(items: list[str], empty: str = "none") -> list[str]:
    """Render a list of strings as Markdown bullets."""
    return [f"- `{item}`" for item in items] if items else [f"- {empty}"]


def write_dimension_run_summary(record: DimensionRunRecord, dimension_dir: Path) -> Path:
    """Write ``run_summary.md`` for one latent dimension.

    Args:
        record: Outcome of the dimension.
        dimension_dir: Directory of this dimension.

    Returns:
        Path to the written Markdown file.
    """
    if record.visualization_mode == "direct_2d":
        visual = (
            "Direct visualization of `latent_mean_0` x `latent_mean_1` was performed. "
            "PCA, UMAP and t-SNE are not used for the 2D model."
        )
    elif record.visualization_mode == "projections":
        visual = "PCA, UMAP and t-SNE projections of `latent_mean` were performed."
    else:
        visual = "No latent-space visualization was produced (see status and errors)."

    lines = [
        f"# Run Summary: latent_dim_{record.latent_dimension}",
        "",
        f"**Execution status: {record.status.upper()}**",
        "",
    ]
    if record.synthetic_data:
        lines += [SYNTHETIC_NOTICE, ""]
    lines += [
        "## Experiment",
        "",
        f"- **Dataset**: {record.dataset}",
        f"- **Data origin**: {record.data_origin}",
        f"- **Experiment type**: {record.experiment_type}",
        f"- **Split analyzed**: {record.split}",
        f"- **Topology**: {record.topology_name} = {record.hidden_dimensions}",
        f"- **Latent dimension**: {record.latent_dimension}",
        f"- **Seed**: {record.seed}",
        f"- **Device**: {record.device or 'unknown'}",
        f"- **Parameter count**: {_fmt(record.parameter_count)}",
    ]
    if record.experiment_type == "unsupervised":
        lines.append(
            "- Labels did not take part in training; they are used only after training "
            "to color plots and compute subsidiary metrics."
        )

    lines += [
        "",
        "## Training (best checkpoint by validation total loss)",
        "",
        f"- **Best epoch**: {_fmt(record.best_epoch)}",
        f"- **Epochs trained**: {_fmt(record.epochs_trained)}",
        f"- **Validation reconstruction loss**: {_fmt(record.val_reconstruction_loss)}",
        f"- **Validation KL divergence**: {_fmt(record.val_kl_divergence)}",
        f"- **Validation classification loss**: {_fmt(record.val_classification_loss)}",
        f"- **Validation total loss**: {_fmt(record.val_total_loss)}",
        f"- **Training time (s)**: {_fmt(record.training_time_seconds, 1)}",
        f"- **Checkpoint used**: `{_fmt(record.paths.get('checkpoint'))}`",
        "",
        "## Latent-space evidence",
        "",
        f"- {visual}",
        f"- **Samples in split**: {_fmt(record.n_samples_total)}",
    ]
    if record.visualization_mode == "projections":
        sampling = "yes (stratified sample)" if record.used_sampling else "no (full split)"
        lines += [
            f"- **Samples in projections**: {_fmt(record.n_projection_samples)}",
            f"- **Sampling used**: {sampling}",
        ]
        if record.pca_explained_variance is not None:
            pc1, pc2, cumulative = record.pca_explained_variance
            lines += [
                f"- **PCA explained variance PC1**: {_fmt(pc1)}",
                f"- **PCA explained variance PC2**: {_fmt(pc2)}",
                f"- **PCA cumulative (2 components)**: {_fmt(cumulative)}",
                "- Explained variance exists only for PCA; UMAP and t-SNE do not provide it.",
            ]

    lines += ["", "### Figures", "", *_bullets(record.figures)]
    lines += ["", "### Tables and data", "", *_bullets(record.tables)]

    lines += ["", "## Subsidiary metrics", ""]
    if record.projection_metrics:
        lines += [
            "| space | scope | n | silhouette | calinski_harabasz | davies_bouldin "
            "| trustworthiness | unavailable |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in record.projection_metrics:
            lines.append(
                f"| {row.get('space_type')} | {row.get('scope')} | {row.get('n_samples')} "
                f"| {_fmt(row.get('silhouette_score'), 4)} "
                f"| {_fmt(row.get('calinski_harabasz_index'), 2)} "
                f"| {_fmt(row.get('davies_bouldin_index'), 4)} "
                f"| {_fmt(row.get('trustworthiness'), 4)} "
                f"| {row.get('unavailable_metrics') or '-'} |"
            )
        lines.append("")
        lines.append("Metrics are subsidiary; they are not proof of clusters on their own.")
    else:
        lines.append("- No metrics were computed.")

    lines += ["", "## Warnings", "", *_bullets(record.warnings)]
    lines += ["", "## Errors", ""]
    if record.failures:
        for item in record.failures:
            lines.append(
                f"- **{item['operation']}** (dimension={record.latent_dimension}, "
                f"dataset={record.dataset}): {item['error']}. "
                f"Probable cause: {item['probable_cause']} "
                f"Log: `{_fmt(record.paths.get('log'))}`"
            )
    else:
        lines.append("- none")

    lines += [
        "",
        "## Limitations",
        "",
        "- Visual analysis is the primary evidence; metrics are subsidiary.",
        "- A single seed is used; run-to-run variability is not estimated.",
        "- t-SNE and UMAP distort global distances and cluster sizes.",
        "- Model selection uses the validation split only; the test set is not used.",
    ]
    if record.quick_mode:
        lines += [
            "- **QUICK MODE**: results are for smoke testing only and must not be reported.",
        ]

    lines += ["", "---", "*Generated automatically by vae_project*", ""]

    summary_path = Path(dimension_dir) / "run_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Run summary saved: %s", summary_path)
    return summary_path


def write_comparison_summary(
    records: list[DimensionRunRecord],
    output_dir: Path,
) -> Path:
    """Write ``latent_dimension_comparison_summary.md``.

    Args:
        records: Outcomes of all dimensions.
        output_dir: Comparison directory.

    Returns:
        Path to the written Markdown file.
    """
    first = records[0]
    dims = sorted(record.latent_dimension for record in records)
    best = select_lowest_validation_loss(records)
    not_completed = [record for record in records if record.status != STATUS_COMPLETED]

    synthetic = any(record.synthetic_data for record in records)
    lines = [
        "# Latent Dimension Comparison Summary",
        "",
    ]
    if synthetic:
        lines += [SYNTHETIC_NOTICE, ""]
    lines += [
        f"- **Dataset**: {first.dataset}",
        f"- **Data origin**: {first.data_origin}",
        f"- **Experiment type**: {first.experiment_type}",
        f"- **Topology (fixed)**: {first.topology_name} = {first.hidden_dimensions}",
        f"- **Seed**: {first.seed}",
        f"- **Dimensions executed**: {dims}",
        "- Each dimension is an independent VAE; the latent dimension is the only "
        "experimental variable.",
        "",
        "## Results",
        "",
        "| latent_dim | status | best_epoch | val_total_loss | val_recon | val_kl "
        "| params | time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in sorted(records, key=lambda item: item.latent_dimension):
        lines.append(
            f"| {record.latent_dimension} | {record.status} | {_fmt(record.best_epoch)} "
            f"| {_fmt(record.val_total_loss)} | {_fmt(record.val_reconstruction_loss)} "
            f"| {_fmt(record.val_kl_divergence)} | {_fmt(record.parameter_count)} "
            f"| {_fmt(record.training_time_seconds, 1)} |"
        )

    lines += ["", "## Lowest validation total loss", ""]
    if best is None:
        lines.append("- No dimension finished training; nothing to select.")
    else:
        lines += [
            f"- Lowest validation total loss: **latent_dim={best.latent_dimension}** "
            f"({_fmt(best.val_total_loss)}).",
            "- This is a numeric statement about the loss only. It does not say which "
            "dimension has the best clusters; that requires visual analysis of the figures.",
        ]

    lines += ["", "## Files produced", ""]
    for record in sorted(records, key=lambda item: item.latent_dimension):
        lines.append(f"### latent_dim_{record.latent_dimension}")
        lines.append("")
        lines += _bullets(
            [
                f"latent_dim_{record.latent_dimension}/{name}"
                for name in [*record.figures, *record.tables]
            ]
        )
        lines.append("")
    lines.append("- Consolidated table: `latent_dimension_comparison.csv`")

    lines += ["", "## Partial failures", ""]
    if not_completed:
        for record in not_completed:
            detail = record.error_summary() or "see run_summary.md"
            lines.append(
                f"- latent_dim={record.latent_dimension}: **{record.status}** - {detail} "
                f"(summary: `{_fmt(record.paths.get('run_summary'))}`)"
            )
    else:
        lines.append("- none")

    lines += [
        "",
        "## How to interpret",
        "",
        "- Start with the figures: compare the direct 2D plot (dimension 2) with the PCA, "
        "UMAP and t-SNE plots of the other dimensions, using the same class colors.",
        "- Check whether apparent groups appear in more than one projection before "
        "trusting them; a single method can create artifacts.",
        "- Only PCA provides explained variance; do not attribute it to UMAP or t-SNE.",
        "- Subsidiary metrics (`projection_metrics.csv`) support, but never replace, "
        "the visual analysis.",
        "- When projections used a sample, the figure titles state it.",
        "",
        "## Limitations",
        "",
        "- Single seed: variability across seeds is not estimated.",
        "- Fully-connected MLP VAE; no spatial structure is modeled.",
        "- Metrics use ground-truth labels as cluster assignments.",
        "- Validation split only; the test set is not used.",
    ]
    if any(record.quick_mode for record in records):
        lines.append("- **QUICK MODE**: smoke-test results, not for scientific reporting.")

    lines += ["", "---", "*Generated automatically by vae_project*", ""]

    summary_path = Path(output_dir) / "latent_dimension_comparison_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Comparison summary saved: %s", summary_path)
    return summary_path


def generate_consolidated_comparison(
    topology_results: pd.DataFrame | None,
    dimension_results: pd.DataFrame | None,
    output_dir: Path,
    is_quick_mode: bool = False,
) -> list[Path]:
    """Generate consolidated comparison tables and reports.

    Args:
        topology_results: DataFrame from topology selection.
        dimension_results: DataFrame from dimension comparison.
        output_dir: Output directory.
        is_quick_mode: Whether to add quick-mode warnings.

    Returns:
        List of generated file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    if topology_results is not None and not topology_results.empty:
        path = output_dir / "consolidated_topology_comparison.csv"
        topology_results.to_csv(path, index=False)
        generated.append(path)
        logger.info("Consolidated topology comparison saved: %s", path)

    if dimension_results is not None and not dimension_results.empty:
        path = output_dir / "consolidated_dimension_comparison.csv"
        dimension_results.to_csv(path, index=False)
        generated.append(path)
        logger.info("Consolidated dimension comparison saved: %s", path)

    return generated
