"""Comparison of the unsupervised winner with its supervised extension.

The two evaluations are read from their ``final_evaluation_results.json``
files, so the comparison can be rebuilt after a resume without keeping
in-memory objects. The report states measured values and arithmetic
differences only; interpretation is left to the researcher. The total losses
of the two modes are listed separately and are never compared numerically,
because they have a different composition:

    unsupervised total = reconstruction + beta * KL
    supervised total   = reconstruction + beta * KL + weight * classification
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vae_project.experiments.heldout_evaluation import RESULTS_FILENAME
from vae_project.reporting.artifact_manifest import ArtifactManifest
from vae_project.reporting.formatting import format_value
from vae_project.utils.json_io import read_json

logger = logging.getLogger(__name__)

COMPARISON_CSV_FILENAME = "comparison_with_unsupervised.csv"
COMPARISON_SUMMARY_FILENAME = "comparison_summary.md"
MANIFEST_FILENAME = "artifact_manifest.json"
FIGURE_SUFFIXES = (".png", ".svg")
LIST_SEPARATOR = ";"
STATUS_MISSING = "missing"
METRIC_NAMES = (
    "silhouette_score",
    "calinski_harabasz_index",
    "davies_bouldin_index",
    "trustworthiness",
)
BASE_COLUMNS = (
    "dataset",
    "topology_name",
    "latent_dimension",
    "seed",
    "experiment_type",
    "validation_reconstruction_loss",
    "validation_kl_divergence",
    "validation_classification_loss",
    "test_reconstruction_loss",
    "test_kl_divergence",
    "test_classification_loss",
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_f1_macro",
    "pca_component_1_explained_variance",
    "pca_component_2_explained_variance",
    "pca_cumulative_explained_variance",
)
TRAILING_COLUMNS = (
    "projection_metrics_path",
    "checkpoint_path",
    "training_figure_paths",
    "evaluation_figure_paths",
    "status",
)
RESEARCHER_FIELDS = (
    "Does supervision make the classes visually more separable in the latent space?",
    "Does supervision change the sharpness of the reconstructions?",
    "Which classes overlap in each mode?",
    "Do the subsidiary metrics agree with the visual impression?",
)


@dataclass(frozen=True)
class ModeArtifacts:
    """Where the artifacts of one mode (unsupervised or supervised) live.

    Attributes:
        label: ``unsupervised`` or ``supervised``.
        training_directory: Run directory with the checkpoint and the
            training figures.
        evaluation_directory: Test evaluation directory.
    """

    label: str
    training_directory: Path
    evaluation_directory: Path


def _load_results(artifacts: ModeArtifacts) -> dict[str, Any] | None:
    """Read the test-evaluation results of a mode, or None when missing."""
    path = artifacts.evaluation_directory / RESULTS_FILENAME
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _relative(protocol_dir: Path, path: Path | str | None) -> str | None:
    """Return ``path`` relative to the protocol directory when possible."""
    if path is None:
        return None
    candidate = Path(path)
    try:
        return candidate.relative_to(protocol_dir).as_posix()
    except ValueError:
        return candidate.as_posix()


def _figure_list(protocol_dir: Path, directory: Path) -> list[str]:
    """List every figure file below ``directory/figures`` (relative paths)."""
    figures = directory / "figures"
    if not figures.is_dir():
        return []
    files = sorted(path for path in figures.rglob("*") if path.suffix in FIGURE_SUFFIXES)
    return [path.relative_to(protocol_dir).as_posix() for path in files]


def _flatten_projection_metrics(rows: object) -> dict[str, object]:
    """Flatten projection metric rows into ``<space>_<metric>`` columns."""
    flat: dict[str, object] = {}
    if not isinstance(rows, list):
        return flat
    for row in rows:
        space = str(row.get("space_type"))
        for name in METRIC_NAMES:
            flat[f"{space}_{name}"] = row.get(name)
    return flat


def _mode_row(protocol_dir: Path, artifacts: ModeArtifacts) -> dict[str, object]:
    """Build the comparison row of one mode (missing values stay None)."""
    results = _load_results(artifacts)
    if results is None:
        return {"experiment_type": artifacts.label, "status": STATUS_MISSING}
    test = results.get("test_metrics") or {}
    variance = results.get("pca_explained_variance") or [None, None, None]
    row: dict[str, object] = {
        "dataset": results.get("dataset"),
        "topology_name": results.get("topology_name"),
        "latent_dimension": results.get("latent_dimension"),
        "seed": results.get("seed"),
        "experiment_type": results.get("experiment_type"),
        "validation_reconstruction_loss": results.get("validation_reconstruction_loss"),
        "validation_kl_divergence": results.get("validation_kl_divergence"),
        "validation_classification_loss": results.get("validation_classification_loss"),
        "test_reconstruction_loss": test.get("test_reconstruction_loss"),
        "test_kl_divergence": test.get("test_kl_divergence"),
        "test_classification_loss": test.get("test_classification_loss"),
        "test_accuracy": test.get("test_accuracy"),
        "test_precision_macro": test.get("test_precision_macro"),
        "test_recall_macro": test.get("test_recall_macro"),
        "test_f1_macro": test.get("test_f1_macro"),
        "pca_component_1_explained_variance": variance[0],
        "pca_component_2_explained_variance": variance[1],
        "pca_cumulative_explained_variance": variance[2],
    }
    row.update(_flatten_projection_metrics(results.get("projection_metrics")))
    metrics_table = artifacts.evaluation_directory / "tables" / "projection_metrics.csv"
    row["projection_metrics_path"] = (
        _relative(protocol_dir, metrics_table) if metrics_table.is_file() else None
    )
    row["checkpoint_path"] = _relative(protocol_dir, results.get("checkpoint"))
    row["training_figure_paths"] = LIST_SEPARATOR.join(
        _figure_list(protocol_dir, artifacts.training_directory)
    )
    row["evaluation_figure_paths"] = LIST_SEPARATOR.join(
        _figure_list(protocol_dir, artifacts.evaluation_directory)
    )
    row["status"] = results.get("execution_status")
    return row


def build_comparison_table(
    protocol_dir: Path,
    unsupervised: ModeArtifacts,
    supervised: ModeArtifacts,
) -> pd.DataFrame:
    """Build the two-row comparison table (unsupervised first).

    Args:
        protocol_dir: Protocol run directory (paths are relative to it).
        unsupervised: Artifacts of the unsupervised winner.
        supervised: Artifacts of the supervised extension.

    Returns:
        DataFrame with one row per mode; not-applicable values are empty.
    """
    rows = [_mode_row(protocol_dir, unsupervised), _mode_row(protocol_dir, supervised)]
    metric_columns = sorted(
        {key for row in rows for key in row} - set(BASE_COLUMNS) - set(TRAILING_COLUMNS)
    )
    columns = [*BASE_COLUMNS, *metric_columns, *TRAILING_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def _difference(table: pd.DataFrame, column: str) -> str:
    """Return ``supervised - unsupervised`` for a column, or ``n/a``."""
    values = pd.to_numeric(table[column], errors="coerce")
    if len(values) < 2 or values.isna().any():
        return "n/a"
    return format_value(float(values.iloc[1] - values.iloc[0]))


def _value(table: pd.DataFrame, position: int, column: str) -> str:
    """Format one table cell for Markdown."""
    value = table.iloc[position][column]
    return format_value(None if pd.isna(value) else value)


def _loss_table(table: pd.DataFrame) -> list[str]:
    """Return the reconstruction/KL comparison with arithmetic differences."""
    lines = [
        "| quantity | unsupervised | supervised | supervised - unsupervised |",
        "|---|---|---|---|",
    ]
    for column in (
        "validation_reconstruction_loss",
        "validation_kl_divergence",
        "test_reconstruction_loss",
        "test_kl_divergence",
    ):
        lines.append(
            f"| {column} | {_value(table, 0, column)} | {_value(table, 1, column)} "
            f"| {_difference(table, column)} |"
        )
    return lines


def _classification_table(table: pd.DataFrame) -> list[str]:
    """Return the classification values (supervised mode only)."""
    lines = ["| quantity | supervised value |", "|---|---|"]
    for column in (
        "validation_classification_loss",
        "test_classification_loss",
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
    ):
        lines.append(f"| {column} | {_value(table, 1, column)} |")
    return lines


def _projection_metric_table(table: pd.DataFrame) -> list[str]:
    """Return the subsidiary projection metrics of both modes."""
    columns = [
        name
        for name in table.columns
        if name.endswith(METRIC_NAMES) and not table[name].isna().all()
    ]
    if not columns:
        return ["No subsidiary metric is available for either mode."]
    lines = ["| metric | unsupervised | supervised |", "|---|---|---|"]
    for column in columns:
        lines.append(f"| {column} | {_value(table, 0, column)} | {_value(table, 1, column)} |")
    return lines


def _figure_links(table: pd.DataFrame, summary_dir: Path, protocol_dir: Path) -> list[str]:
    """Return links, relative to the summary, to the evaluation figures of both modes."""
    lines: list[str] = []
    for position in range(len(table)):
        mode = table.iloc[position]["experiment_type"]
        paths = table.iloc[position]["evaluation_figure_paths"]
        names = [] if pd.isna(paths) else str(paths).split(LIST_SEPARATOR)
        png_files = [name for name in names if name.endswith(".png")]
        lines += [f"### {mode}", ""]
        for name in png_files:
            link = Path(os.path.relpath(Path(protocol_dir) / name, summary_dir)).as_posix()
            lines.append(f"- [{name}]({link})")
        if not png_files:
            lines.append("- none")
        lines.append("")
    return lines


def write_comparison_summary(
    table: pd.DataFrame,
    summary_dir: Path,
    protocol_dir: Path,
    synthetic_data: bool,
) -> Path:
    """Write ``comparison_summary.md`` without scientific conclusions.

    Args:
        table: Table from :func:`build_comparison_table`.
        summary_dir: ``supervised_extension`` directory.
        protocol_dir: Protocol run directory.
        synthetic_data: Whether the run used synthetic data.

    Returns:
        Path of the Markdown file.
    """
    lines = ["# Supervised vs Unsupervised Comparison", ""]
    if synthetic_data:
        lines += ["**SYNTHETIC DATA: smoke test, non-scientific results.**", ""]
    identity = table.iloc[0]
    lines += [
        "Both modes use the same dataset, split, seed, topology, hidden dimensions, latent "
        "dimension, batch size, optimizer, epochs and projection parameters. They differ only "
        "in the supervised component (classification head, classification loss and its "
        "weight).",
        "",
        f"- **Dataset**: {format_value(identity['dataset'])}",
        f"- **Topology**: {format_value(identity['topology_name'])}",
        f"- **Latent dimension**: {format_value(identity['latent_dimension'])}",
        f"- **Seed**: {format_value(identity['seed'])}",
        "",
        "## Results by mode",
        "",
        f"- **Unsupervised status**: {_value(table, 0, 'status')}",
        f"- **Supervised status**: {_value(table, 1, 'status')}",
        "",
        "## Loss composition",
        "",
        "- Unsupervised total loss = reconstruction + beta x KL divergence.",
        "- Supervised total loss = reconstruction + beta x KL divergence + "
        "classification weight x classification loss.",
        "- The two total losses have a different composition and are intentionally not "
        "compared numerically.",
        "",
        "## Reconstruction and KL divergence",
        "",
        "The last column is an arithmetic difference, not an interpretation.",
        "",
        *_loss_table(table),
        "",
        "## Classification (supervised mode only)",
        "",
        *_classification_table(table),
        "",
        "## Subsidiary projection metrics",
        "",
        "Metrics are subsidiary evidence; they do not replace the visual inspection.",
        "",
        *_projection_metric_table(table),
        "",
        "## Visual differences",
        "",
        "Not assessed automatically. Compare the figures listed below.",
        "",
        *_figure_links(table, summary_dir, protocol_dir),
        "## Limitations",
        "",
        "- A single seed and a single run per mode: no estimate of variability.",
        "- The test split was used once per mode, for the selected configuration only.",
        "- Subsidiary metrics depend on the projection and on the sampled examples.",
        "",
        "## Manual interpretation (to be completed by the researcher)",
        "",
    ]
    for question in RESEARCHER_FIELDS:
        lines += [f"- **{question}**", "  - Answer: _(to be completed)_", ""]
    lines += ["---", "*Generated automatically by vae_project*", ""]
    path = Path(summary_dir) / COMPARISON_SUMMARY_FILENAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_supervised_comparison(
    protocol_dir: Path,
    summary_dir: Path,
    unsupervised: ModeArtifacts,
    supervised: ModeArtifacts,
    synthetic_data: bool,
) -> list[Path]:
    """Write the comparison CSV, summary and the manifest of the extension.

    Args:
        protocol_dir: Protocol run directory.
        summary_dir: ``supervised_extension`` directory.
        unsupervised: Artifacts of the unsupervised winner.
        supervised: Artifacts of the supervised extension.
        synthetic_data: Whether the run used synthetic data.

    Returns:
        The CSV, Markdown and manifest paths.
    """
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    table = build_comparison_table(protocol_dir, unsupervised, supervised)
    csv_path = summary_dir / COMPARISON_CSV_FILENAME
    table.to_csv(csv_path, index=False)
    summary_path = write_comparison_summary(table, summary_dir, protocol_dir, synthetic_data)

    manifest = ArtifactManifest(
        summary_dir,
        context={
            "dataset": _value(table, 0, "dataset"),
            "phase": "supervised extension",
            "data_origin": "synthetic" if synthetic_data else "real",
        },
    )
    manifest.register(csv_path, "table", "Supervised vs unsupervised comparison", True)
    manifest.register(summary_path, "report", "Comparison summary", True)
    for directory in (supervised.training_directory, supervised.evaluation_directory):
        local_manifest = directory / MANIFEST_FILENAME
        if local_manifest.is_file():
            manifest.register(local_manifest, "manifest", f"Manifest of {directory.name}", True)
    manifest_path = manifest.save()
    logger.info("Supervised comparison written to %s", summary_dir)
    return [csv_path, summary_path, manifest_path]
