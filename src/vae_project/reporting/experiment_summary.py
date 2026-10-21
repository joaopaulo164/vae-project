"""Markdown ``run_summary.md`` writer for one single-configuration experiment.

The same writer serves the individual pipeline, the latent dimension
comparison, every combination of the factorial grid, the final test evaluation
and the supervised extension, so the summary format is identical everywhere.

No scientific conclusion is generated here: only numbers, paths, failures and
reading guidance.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vae_project.experiments.experiment_result import (
    FIGURE_CATEGORIES,
    LATENT_KIND_PCA,
    SingleExperimentResult,
    relative_posix,
)
from vae_project.reporting.formatting import bullets, format_value

logger = logging.getLogger(__name__)

SYNTHETIC_NOTICE = (
    "**SYNTHETIC DATA: this run uses artificial data for testing/smoke-testing only. "
    "Its numbers and figures are NOT scientific results and must not be reported "
    "or mixed with the MNIST/Fashion-MNIST comparison.**"
)
SUMMARY_FILENAME = "run_summary.md"

FIGURE_CATEGORY_TITLES = {
    "training": "Training curves",
    "reconstructions": "Reconstructions",
    "generated_samples": "Generated samples (prior N(0, I))",
    "latent_space": "Latent space",
}


def _visualization_text(result: SingleExperimentResult) -> str:
    """Describe which latent-space visualization was produced."""
    if result.visualization_mode == "direct_2d":
        return (
            "Direct visualization of `latent_mean_0` x `latent_mean_1` was performed. "
            "PCA, UMAP and t-SNE are not used for the 2D model."
        )
    if result.visualization_mode == "projections":
        return "PCA, UMAP and t-SNE projections of `latent_mean` were performed."
    return "No latent-space visualization was produced (see status and errors)."


def _experiment_lines(result: SingleExperimentResult) -> list[str]:
    """Return the experiment description section."""
    lines = [
        "## Experiment",
        "",
        f"- **Dataset**: {result.dataset}",
        f"- **Data origin**: {result.data_origin}",
        f"- **Experiment type**: {result.experiment_type}",
        f"- **Split analyzed**: {result.split}",
        f"- **Topology**: {result.topology_name} = {result.hidden_dimensions}",
        f"- **Latent dimension**: {result.latent_dimension}",
        f"- **Seed**: {result.seed}",
        f"- **Execution identifier**: {result.execution_id or 'n/a'}",
        f"- **Device**: {result.device or 'unknown'}",
        f"- **Parameter count (VAE)**: {format_value(result.parameter_count)}",
    ]
    if result.environment_reference:
        lines.append(f"- **Environment**: `{result.environment_reference}`")
    if result.experiment_type == "unsupervised":
        lines.append(
            "- Labels did not take part in training; they are used only after training "
            "to color plots and compute subsidiary metrics."
        )
    return lines


def _training_lines(result: SingleExperimentResult, run_dir: Path) -> list[str]:
    """Return the training and checkpoint section."""
    checkpoint = relative_posix(result.best_checkpoint_path, run_dir)
    return [
        "## Training (best checkpoint by validation total loss)",
        "",
        f"- **Best epoch**: {format_value(result.best_epoch)}",
        f"- **Epochs trained**: {format_value(result.epochs_trained)}",
        f"- **Validation reconstruction loss**: "
        f"{format_value(result.validation_reconstruction_loss)}",
        f"- **Validation KL divergence**: {format_value(result.validation_kl_divergence)}",
        f"- **Validation classification loss**: "
        f"{format_value(result.validation_classification_loss)}",
        f"- **Validation total loss**: {format_value(result.validation_total_loss)}",
        f"- **Training duration (s)**: {format_value(result.training_duration_seconds, 1)}",
        f"- **Total duration (s)**: {format_value(result.total_duration_seconds, 1)}",
        f"- **Checkpoint used**: `{format_value(checkpoint)}`",
    ]


def _latent_lines(result: SingleExperimentResult) -> list[str]:
    """Return the latent-space evidence section."""
    lines = [
        "## Latent-space evidence",
        "",
        f"- {_visualization_text(result)}",
        f"- **Samples in split**: {format_value(result.n_samples_total)}",
    ]
    if result.visualization_mode != "projections":
        return lines
    sampling = "yes (stratified sample)" if result.used_sampling else "no (full split)"
    lines += [
        f"- **Samples in projections**: {format_value(result.n_projection_samples)}",
        f"- **Sampling used**: {sampling}",
    ]
    if result.pca_explained_variance is not None:
        first, second, cumulative = result.pca_explained_variance
        lines += [
            f"- **PCA explained variance PC1**: {format_value(first)}",
            f"- **PCA explained variance PC2**: {format_value(second)}",
            f"- **PCA cumulative (2 components)**: {format_value(cumulative)}",
            "- Explained variance exists only for PCA; UMAP and t-SNE do not provide it.",
        ]
    elif LATENT_KIND_PCA not in result.latent_figure_paths:
        lines.append("- PCA was not produced; no explained variance is available.")
    return lines


def _figure_lines(result: SingleExperimentResult, run_dir: Path) -> list[str]:
    """Return one sub-section per figure category."""
    relative = result.relative_figures(run_dir)
    lines = ["## Figures", ""]
    for category in FIGURE_CATEGORIES:
        lines += [f"### {FIGURE_CATEGORY_TITLES[category]}", ""]
        lines += bullets(relative.get(category, []))
        lines.append("")
    return lines


def _metrics_lines(result: SingleExperimentResult) -> list[str]:
    """Return the subsidiary metrics table."""
    lines = ["## Subsidiary metrics", ""]
    if not result.projection_metrics:
        return [*lines, "- No metrics were computed."]
    lines += [
        "| space | scope | n | silhouette | calinski_harabasz | davies_bouldin "
        "| trustworthiness | unavailable |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result.projection_metrics:
        lines.append(
            f"| {row.get('space_type')} | {row.get('scope')} | {row.get('n_samples')} "
            f"| {format_value(row.get('silhouette_score'), 4)} "
            f"| {format_value(row.get('calinski_harabasz_index'), 2)} "
            f"| {format_value(row.get('davies_bouldin_index'), 4)} "
            f"| {format_value(row.get('trustworthiness'), 4)} "
            f"| {row.get('unavailable_metrics') or '-'} |"
        )
    lines += ["", "Metrics are subsidiary; they are not proof of clusters on their own."]
    return lines


def _error_lines(result: SingleExperimentResult, run_dir: Path) -> list[str]:
    """Return warnings and errors sections."""
    lines = ["## Warnings", "", *bullets(result.warnings), "", "## Errors", ""]
    if not result.errors:
        return [*lines, "- none"]
    log_reference = format_value(relative_posix(result.log_path, run_dir))
    for item in result.errors:
        lines.append(
            f"- **{item['operation']}** (topology={result.topology_name}, "
            f"latent_dimension={result.latent_dimension}, dataset={result.dataset}): "
            f"{item['error']}. Probable cause: {item.get('probable_cause', 'unknown')} "
            f"Log: `{log_reference}`"
        )
    return lines


def _limitation_lines(result: SingleExperimentResult) -> list[str]:
    """Return the limitations section."""
    split_note = (
        "- This summary reports the final TEST evaluation of an already selected "
        "configuration; the test set did not influence the selection."
        if result.split == "test"
        else "- Model selection uses the validation split only; the test set is not used."
    )
    lines = [
        "## Limitations",
        "",
        "- Visual analysis is the primary evidence; metrics are subsidiary.",
        "- A single seed is used; run-to-run variability is not estimated.",
        "- t-SNE and UMAP distort global distances and cluster sizes.",
        split_note,
    ]
    if result.quick_mode:
        lines.append(
            "- **QUICK MODE**: results are for smoke testing only and must not be reported."
        )
    return lines


def write_experiment_run_summary(
    result: SingleExperimentResult,
    run_dir: Path,
    extra_lines: list[str] | None = None,
) -> Path:
    """Write ``run_summary.md`` for one experiment execution.

    Args:
        result: Outcome of the execution (paths rooted at ``run_dir``).
        run_dir: Directory of the execution; the summary is written inside it
            and all listed paths are relative to it.
        extra_lines: Optional Markdown lines inserted after the experiment
            section (for example the phase of a final evaluation).

    Returns:
        Path to the written Markdown file.
    """
    run_dir = Path(run_dir)
    title = f"{result.topology_name} / latent_dim_{result.latent_dimension}"
    lines = [f"# Run Summary: {title}", "", f"**Execution status: {result.status.upper()}**", ""]
    if result.synthetic_data:
        lines += [SYNTHETIC_NOTICE, ""]
    lines += _experiment_lines(result)
    if extra_lines:
        lines += ["", *extra_lines]
    lines += ["", *_training_lines(result, run_dir)]
    lines += ["", *_latent_lines(result), ""]
    lines += _figure_lines(result, run_dir)
    lines += ["## Tables and data", "", *bullets(result.relative_tables(run_dir)), ""]
    lines += [*_metrics_lines(result), ""]
    lines += [*_error_lines(result, run_dir), ""]
    lines += [*_limitation_lines(result), "", "---", "*Generated automatically by vae_project*", ""]

    summary_path = run_dir / SUMMARY_FILENAME
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Run summary saved: %s", summary_path)
    return summary_path
