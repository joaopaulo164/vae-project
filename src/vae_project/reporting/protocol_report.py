"""Reports of a factorial protocol run.

Three documents are produced from the state, the combination records and the
files already on disk (never from invented values):

* ``protocol_summary.md``: status of every phase and combination.
* ``scientific_evidence_index.md``: where each kind of evidence is stored,
  plus empty questions for the researcher. No visual observation is written
  here: the visual analysis is the main evidence and belongs to the researcher.
* ``protocol_manifest.json``: global manifest that references the local
  manifests of the combinations instead of repeating their entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vae_project.config.models import ExperimentConfig
from vae_project.experiments.protocol_models import CombinationRecord
from vae_project.experiments.protocol_selection import SelectionOutcome
from vae_project.experiments.protocol_state import ProtocolState
from vae_project.reporting.artifact_manifest import compute_file_hash
from vae_project.reporting.formatting import bullets, format_value
from vae_project.utils.json_io import read_json, write_json

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "protocol_summary.md"
INDEX_FILENAME = "scientific_evidence_index.md"
MANIFEST_FILENAME = "protocol_manifest.json"
RESULTS_FILENAME = "final_evaluation_results.json"
SYNTHETIC_NOTICE = "**SYNTHETIC DATA: smoke test, non-scientific results.**"
SYNTHETIC_ORIGIN = "synthetic"
FOOTER = "*Generated automatically by vae_project*"
RESEARCHER_QUESTIONS = (
    "Are there visually identifiable clusters?",
    "Are the labels separated in the latent space?",
    "Which classes overlap?",
    "Which topology produces the sharpest reconstructions?",
    "How does the latent dimension affect the organization of the space?",
    "Does the configuration with the lowest validation loss also show the best visual "
    "separation?",
    "Does supervision improve the separation?",
    "Does supervision affect the reconstruction?",
)
LIMITATIONS = (
    "A single seed (one run per combination): no estimate of run-to-run variability.",
    "Validation losses select the configuration; they are not a measure of visual cluster "
    "separation.",
    "Subsidiary metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin, trustworthiness) "
    "depend on the projection and on the sampled examples.",
    "UMAP and t-SNE distances are not directly interpretable; only PCA reports explained "
    "variance.",
)
ENVIRONMENT_KEYS = (
    "python_version",
    "torch_version",
    "cuda_available",
    "gpu_name",
    "os_system",
    "os_release",
    "git_commit_hash",
)


@dataclass
class ProtocolReportContext:
    """Everything the protocol reports are built from.

    Attributes:
        protocol_dir: Protocol run directory (all links are relative to it).
        config: Protocol configuration.
        state: Current protocol state.
        records: One record per grid combination (any status).
        selection: Selection outcome, or None when selection did not run.
        manual_command: Command to run the supervised extension manually.
    """

    protocol_dir: Path
    config: ExperimentConfig
    state: ProtocolState
    records: list[CombinationRecord]
    selection: SelectionOutcome | None = None
    manual_command: str | None = None

    @property
    def synthetic(self) -> bool:
        """True when the run used synthetic data."""
        return self.state.data_origin == SYNTHETIC_ORIGIN


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, or None when the file is missing."""
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _link(text: str, target: str | None) -> str:
    """Return a Markdown link, or ``n/a`` when the target is missing."""
    return f"[{text}]({target})" if target else "n/a"


def _figure_files(directory: Path, protocol_dir: Path) -> list[str]:
    """List the PNG figures below ``directory/figures`` (protocol-relative)."""
    figures = directory / "figures"
    if not figures.is_dir():
        return []
    files = sorted(figures.rglob("*.png"))
    return [path.relative_to(protocol_dir).as_posix() for path in files]


def _grid_table(context: ProtocolReportContext) -> list[str]:
    """Return the Markdown table of every combination."""
    lines = [
        "| topology | latent_dim | status | best_epoch | val_total_loss | val_reconstruction "
        "| val_kl | parameters | summary |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in context.records:
        lines.append(
            f"| {record.topology_name} | {record.latent_dimension} | {record.status} "
            f"| {format_value(record.best_epoch)} "
            f"| {format_value(record.validation_total_loss)} "
            f"| {format_value(record.validation_reconstruction_loss)} "
            f"| {format_value(record.validation_kl_divergence)} "
            f"| {format_value(record.parameter_count)} "
            f"| {_link('run_summary', record.run_summary_path)} |"
        )
    return lines


def _environment_lines(protocol_dir: Path) -> list[str]:
    """Return the environment section read from ``shared/environment.json``."""
    environment = _read_optional_json(protocol_dir / "shared" / "environment.json")
    if environment is None:
        return ["- Environment file not available."]
    lines = [f"- **{key}**: {format_value(environment.get(key))}" for key in ENVIRONMENT_KEYS]
    return [*lines, "- Full details: [shared/environment.json](shared/environment.json)"]


def _selection_lines(context: ProtocolReportContext) -> list[str]:
    """Return the selection section."""
    selection_rule = context.config.factorial_comparison.selection
    lines = [
        f"- **Primary metric**: `{selection_rule.primary_metric}` ({selection_rule.direction})",
        f"- **Tolerance**: {selection_rule.tolerance}",
        "- **Tie-breakers**: "
        + ", ".join(f"`{item.metric}` ({item.direction})" for item in selection_rule.tie_breakers),
        "- Selection used validation results only; the test split took no part in it.",
    ]
    winner = context.selection.winner if context.selection else None
    if winner is None:
        return [*lines, "- **Selected combination**: none (no valid combination)."]
    return [
        *lines,
        f"- **Selected combination**: {winner.topology_name} (latent dimension "
        f"{winner.latent_dimension})",
        f"- **Hidden dimensions**: {winner.hidden_dimensions}",
        f"- **Validation total loss**: {format_value(winner.validation_total_loss)}",
        "- Details: [selection/selection_summary.md](selection/selection_summary.md), "
        "[selection/factorial_comparison.csv](selection/factorial_comparison.csv)",
    ]


def _evaluation_lines(protocol_dir: Path, relative: str, title: str, status: str) -> list[str]:
    """Return the test-evaluation section of a phase."""
    lines = [f"- **{title} status**: {status}"]
    results = _read_optional_json(protocol_dir / relative / RESULTS_FILENAME)
    if results is None:
        return [*lines, "- No test results were produced."]
    test = results.get("test_metrics") or {}
    for key in (
        "test_reconstruction_loss",
        "test_kl_divergence",
        "test_classification_loss",
        "test_total_loss",
        "test_accuracy",
    ):
        lines.append(f"- **{key}**: {format_value(test.get(key))}")
    lines.append(f"- Details: [{relative}/run_summary.md]({relative}/run_summary.md)")
    return lines


def _issue_lines(context: ProtocolReportContext) -> list[str]:
    """Return warnings and errors of the whole run."""
    combination_errors = [
        f"{record.key}: {item.get('operation')}: {item.get('error')}"
        for record in context.records
        for item in record.errors
    ]
    return [
        "## Warnings",
        "",
        *bullets(context.state.warnings),
        "",
        "## Failures",
        "",
        *bullets([*context.state.errors, *combination_errors]),
    ]


def write_protocol_summary(context: ProtocolReportContext) -> Path:
    """Write ``protocol_summary.md`` (English, no scientific conclusion).

    Args:
        context: Report inputs.

    Returns:
        Path of the Markdown file.
    """
    state = context.state
    factorial = context.config.factorial_comparison
    lines = [f"# Protocol Summary: {context.config.protocol.name}", ""]
    if context.synthetic:
        lines += [SYNTHETIC_NOTICE, ""]
    lines += [
        f"**Protocol status: {state.status.upper()}**",
        "",
        "## Overview",
        "",
        f"- **Dataset**: {state.dataset}",
        f"- **Run identifier**: {state.run_id}",
        f"- **Data origin**: {state.data_origin}",
        f"- **Topologies**: {', '.join(factorial.topologies)}",
        f"- **Latent dimensions**: {factorial.latent_dimensions}",
        f"- **Combinations**: {len(context.records)} (unsupervised)",
        f"- **Seed**: {context.config.reproducibility.global_seed}",
        f"- **Device policy**: {context.config.runtime.device}",
        f"- **Configuration hash**: {state.config_hash}",
        f"- **Created**: {state.created_at}; **updated**: {state.updated_at}",
        f"- **Resumed at**: {', '.join(state.resumed_at) if state.resumed_at else 'never'}",
        "",
        "## Environment",
        "",
        *_environment_lines(context.protocol_dir),
        "",
        "## Grid (validation results)",
        "",
        *_grid_table(context),
        "",
        "## Selection",
        "",
        *_selection_lines(context),
        "",
        "## Final evaluation (test split, selected configuration only)",
        "",
        *_evaluation_lines(
            context.protocol_dir,
            "final_evaluation",
            "Final evaluation",
            state.final_evaluation_status,
        ),
        "",
        "## Supervised extension",
        "",
        f"- **Training status**: {state.supervised_training_status}",
        *_evaluation_lines(
            context.protocol_dir,
            "supervised_extension/final_evaluation",
            "Supervised evaluation",
            state.supervised_evaluation_status,
        ),
    ]
    if context.manual_command:
        lines += [f"- **Manual command**: `{context.manual_command}`"]
    comparison = context.protocol_dir / "supervised_extension" / "comparison_summary.md"
    if comparison.is_file():
        lines += [
            "- Comparison: "
            "[supervised_extension/comparison_summary.md]"
            "(supervised_extension/comparison_summary.md)"
        ]
    lines += ["", *_issue_lines(context), "", "## Limitations", "", *bullets(list(LIMITATIONS))]
    lines += [
        "",
        "## Visual analysis",
        "",
        "The visual analysis of the latent spaces is the main evidence and must be performed "
        "by the researcher. See [scientific_evidence_index.md](scientific_evidence_index.md).",
        "",
        "---",
        FOOTER,
        "",
    ]
    path = context.protocol_dir / SUMMARY_FILENAME
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Protocol summary saved: %s", path)
    return path


def _category_section(
    title: str,
    records: list[CombinationRecord],
    paths_of: Any,
) -> list[str]:
    """Return a table linking one evidence category for every combination."""
    lines = [f"## {title}", "", "| topology | latent_dim | files |", "|---|---|---|"]
    for record in records:
        paths = [path for path in paths_of(record) if path]
        links = ", ".join(_link(Path(path).name, path) for path in paths) or "n/a"
        lines.append(f"| {record.topology_name} | {record.latent_dimension} | {links} |")
    return [*lines, ""]


def _png_only(paths: list[str]) -> list[str]:
    """Keep the PNG files of a path list."""
    return [path for path in paths if path.endswith(".png")]


def _variance_section(records: list[CombinationRecord]) -> list[str]:
    """Return the PCA explained variance table (PCA only, never UMAP/t-SNE)."""
    lines = [
        "## PCA explained variance",
        "",
        "Only PCA has an explained variance. UMAP and t-SNE do not.",
        "",
        "| topology | latent_dim | component 1 | component 2 | cumulative |",
        "|---|---|---|---|---|",
    ]
    for record in records:
        first, second, cumulative = record.pca_explained_variance or (None, None, None)
        lines.append(
            f"| {record.topology_name} | {record.latent_dimension} | {format_value(first)} "
            f"| {format_value(second)} | {format_value(cumulative)} |"
        )
    return [*lines, ""]


def _phase_figures(protocol_dir: Path, title: str, relative: str) -> list[str]:
    """Return links to the figures of the final or supervised evaluation."""
    figures = _figure_files(protocol_dir / relative, protocol_dir)
    return (
        [f"## {title}", "", *[f"- {_link(path, path)}" for path in figures], ""]
        if (figures)
        else [f"## {title}", "", "- not available", ""]
    )


def write_evidence_index(context: ProtocolReportContext) -> Path:
    """Write ``scientific_evidence_index.md``.

    The index only points to evidence and lists open questions. It does not
    describe what the figures show.

    Args:
        context: Report inputs.

    Returns:
        Path of the Markdown file.
    """
    records = context.records
    lines = ["# Scientific Evidence Index", ""]
    if context.synthetic:
        lines += [SYNTHETIC_NOTICE, ""]
    lines += [
        f"Dataset: **{context.state.dataset}**. Protocol run: `{context.state.run_id}`.",
        "",
        "Visual inspection of the latent spaces is the primary evidence. Quantitative "
        "metrics are subsidiary evidence. Paths are relative to the protocol directory.",
        "",
    ]
    lines += _category_section(
        "Training curves", records, lambda r: _png_only(r.figure_files.get("training", []))
    )
    lines += _category_section(
        "Reconstructions",
        records,
        lambda r: _png_only(r.figure_files.get("reconstructions", [])),
    )
    lines += _category_section(
        "Generated samples",
        records,
        lambda r: _png_only(r.figure_files.get("generated_samples", [])),
    )
    lines += _category_section(
        "Direct latent space (2D)", records, lambda r: [r.latent_plots.get("direct")]
    )
    lines += _category_section("PCA", records, lambda r: [r.latent_plots.get("pca")])
    lines += _category_section("UMAP", records, lambda r: [r.latent_plots.get("umap")])
    lines += _category_section("t-SNE", records, lambda r: [r.latent_plots.get("tsne")])
    lines += _category_section(
        "PCA explained variance figures",
        records,
        lambda r: [r.latent_plots.get("pca_explained_variance")],
    )
    lines += _variance_section(records)
    lines += _category_section(
        "Subsidiary metrics (tables)", records, lambda r: [r.projection_metrics_path]
    )
    lines += _phase_figures(context.protocol_dir, "Final evaluation (test)", "final_evaluation")
    lines += _phase_figures(
        context.protocol_dir,
        "Supervised extension (test)",
        "supervised_extension/final_evaluation",
    )
    lines += [
        "## Comparison between modes",
        "",
        "- [supervised_extension/comparison_with_unsupervised.csv]"
        "(supervised_extension/comparison_with_unsupervised.csv)",
        "- [supervised_extension/comparison_summary.md]"
        "(supervised_extension/comparison_summary.md)",
        "",
        "## Researcher questions (to be answered after the visual analysis)",
        "",
    ]
    for question in RESEARCHER_QUESTIONS:
        lines += [f"- **{question}**", "  - Answer: _(to be completed)_", ""]
    lines += ["---", FOOTER, ""]
    path = context.protocol_dir / INDEX_FILENAME
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Scientific evidence index saved: %s", path)
    return path


def _file_entry(protocol_dir: Path, relative: str | None) -> dict[str, object]:
    """Describe a file with size and SHA-256 (when it exists)."""
    if not relative:
        return {"path": None, "exists": False}
    path = protocol_dir / relative
    exists = path.is_file()
    return {
        "path": relative,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": compute_file_hash(path) if exists else None,
    }


def _directory_files(protocol_dir: Path, relative: str) -> list[dict[str, object]]:
    """Describe the files directly inside a directory of the run."""
    directory = protocol_dir / relative
    if not directory.is_dir():
        return []
    files = sorted(path for path in directory.iterdir() if path.is_file())
    return [_file_entry(protocol_dir, path.relative_to(protocol_dir).as_posix()) for path in files]


def _combination_entry(protocol_dir: Path, record: CombinationRecord) -> dict[str, object]:
    """Reference one combination by its local manifest (no duplicated details)."""
    return {
        "key": record.key,
        "topology_name": record.topology_name,
        "hidden_dimensions": record.hidden_dimensions,
        "latent_dimension": record.latent_dimension,
        "status": record.status,
        "run_directory": record.run_directory,
        "validation_total_loss": record.validation_total_loss,
        "manifest": _file_entry(protocol_dir, record.manifest_path),
        "run_summary": _file_entry(protocol_dir, record.run_summary_path),
        "checkpoint": _file_entry(protocol_dir, record.checkpoint_path),
        "warnings_count": len(record.warnings),
        "errors_count": len(record.errors),
    }


def _phase_entry(protocol_dir: Path, relative: str, status: str) -> dict[str, object]:
    """Reference a final-evaluation style phase by its local manifest."""
    return {
        "directory": relative,
        "status": status,
        "manifest": _file_entry(protocol_dir, f"{relative}/artifact_manifest.json"),
        "run_summary": _file_entry(protocol_dir, f"{relative}/run_summary.md"),
        "results": _file_entry(protocol_dir, f"{relative}/{RESULTS_FILENAME}"),
    }


def write_protocol_manifest(context: ProtocolReportContext) -> Path:
    """Write ``protocol_manifest.json`` referencing the local manifests.

    Args:
        context: Report inputs.

    Returns:
        Path of the JSON file.
    """
    state = context.state
    protocol_dir = context.protocol_dir
    winner = context.selection.winner if context.selection else None
    supervised_dir = "supervised_extension"
    payload = {
        "protocol": context.config.protocol.name,
        "dataset": state.dataset,
        "data_origin": state.data_origin,
        "run_identifier": state.run_id,
        "status": state.status,
        "phase": state.phase,
        "configuration": {
            "config_hash": state.config_hash,
            "resolved_config": _file_entry(protocol_dir, "protocol_config_resolved.yaml"),
            "seed": context.config.reproducibility.global_seed,
            "topologies": context.config.factorial_comparison.topologies,
            "latent_dimensions": context.config.factorial_comparison.latent_dimensions,
        },
        "environment": _file_entry(protocol_dir, "shared/environment.json"),
        "shared_files": _directory_files(protocol_dir, "shared"),
        "combinations": [_combination_entry(protocol_dir, record) for record in context.records],
        "selection": {
            "selected_combination": winner.key if winner else None,
            "selected_configuration": _file_entry(
                protocol_dir, "selection/selected_configuration.yaml"
            ),
            "factorial_comparison": _file_entry(protocol_dir, "selection/factorial_comparison.csv"),
            "selection_summary": _file_entry(protocol_dir, "selection/selection_summary.md"),
        },
        "final_evaluation": _phase_entry(
            protocol_dir, "final_evaluation", state.final_evaluation_status
        ),
        "supervised_extension": {
            "training": _phase_entry(
                protocol_dir, f"{supervised_dir}/training", state.supervised_training_status
            ),
            "final_evaluation": _phase_entry(
                protocol_dir,
                f"{supervised_dir}/final_evaluation",
                state.supervised_evaluation_status,
            ),
            "comparison": _file_entry(
                protocol_dir, f"{supervised_dir}/comparison_with_unsupervised.csv"
            ),
            "manifest": _file_entry(protocol_dir, f"{supervised_dir}/artifact_manifest.json"),
            "generated_config": _file_entry(
                protocol_dir, "generated_configs/supervised_selected_configuration.yaml"
            ),
        },
        "reports": {
            "protocol_summary": _file_entry(protocol_dir, SUMMARY_FILENAME),
            "scientific_evidence_index": _file_entry(protocol_dir, INDEX_FILENAME),
        },
        "timestamps": {
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "resumed_at": state.resumed_at,
        },
        "warnings": state.warnings,
        "errors": state.errors,
    }
    path = write_json(protocol_dir / MANIFEST_FILENAME, payload)
    logger.info("Protocol manifest saved: %s", path)
    return path
