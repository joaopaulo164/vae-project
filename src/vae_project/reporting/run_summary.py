"""Run summary generation: Markdown report for each experiment run.

Generates a human-readable summary of the experiment configuration,
training results, and artifact locations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vae_project.config.models import ExperimentConfig
from vae_project.training.history import TrainingHistory

logger = logging.getLogger(__name__)


def generate_run_summary(
    run_dir: Path,
    config: ExperimentConfig,
    history: TrainingHistory,
    best_epoch: int,
    best_val_loss: float,
    training_duration: float,
    device_info: dict[str, str],
) -> Path:
    """Generate a Markdown summary file for the experiment run.

    Args:
        run_dir: Root directory of the run.
        config: Experiment configuration.
        history: Training history.
        best_epoch: Index of the best epoch.
        best_val_loss: Best validation total loss.
        training_duration: Total training time in seconds.
        device_info: Device information dictionary.

    Returns:
        Path to the generated Markdown file.
    """
    quick_mode_warning = ""
    if config.project.quick_mode:
        quick_mode_warning = (
            "\n> ⚠️ **QUICK MODE**: This run used a reduced dataset and fewer epochs. "
            "Results are for smoke testing ONLY and **must NOT be reported as "
            "scientific findings**.\n"
        )

    supervised_note = ""
    if config.supervised_component.enabled:
        supervised_note = (
            f"\n**Supervised Component**: ENABLED "
            f"(classification_loss_weight={config.supervised_component.classification_loss_weight})\n"
            "> Note: This is the optional supervised extension. Results are separate "
            "from the primary unsupervised VAE experiment.\n"
        )

    best_record = history.get_best_epoch()
    recon_loss_at_best = best_record.val_reconstruction_loss if best_record else "N/A"
    kl_loss_at_best = best_record.val_kl_divergence_loss if best_record else "N/A"

    lines = [
        f"# Run Summary: {run_dir.name}",
        "",
        quick_mode_warning,
        "## Experiment Configuration",
        "",
        f"- **Dataset**: {config.dataset.name.value}",
        f"- **Topology**: {config.model.topology.name} = {config.model.topology.hidden_dimensions}",
        f"- **Latent Dimension**: {config.model.latent_dimension}",
        f"- **Loss Type**: {config.loss.reconstruction_loss_type.value}",
        f"- **Beta (KL weight)**: {config.loss.beta}",
        f"- **Seed**: {config.reproducibility.global_seed}",
        f"- **Max Epochs**: {config.training.max_epochs}",
        f"- **Batch Size**: {config.training.batch_size}",
        f"- **Learning Rate**: {config.optimizer.learning_rate}",
        f"- **Device**: {device_info.get('device', 'unknown')}",
        supervised_note,
        "",
        "## Training Results",
        "",
        f"- **Epochs Completed**: {history.num_epochs_completed}",
        f"- **Best Epoch**: {best_epoch}",
        f"- **Best Val Total Loss**: {best_val_loss:.6f}",
        f"- **Val Reconstruction Loss at Best**: {recon_loss_at_best}",
        f"- **Val KL Divergence at Best**: {kl_loss_at_best}",
        f"- **Training Duration**: {training_duration:.1f}s",
        "",
        "## Artifact Locations",
        "",
        f"- **Run Directory**: `{run_dir}`",
        "- **Checkpoints**: `checkpoints/best_model.pt`",
        "- **Training History**: `tables/training_history.csv`",
        "- **Latent Representations**: `latent_data/`",
        "- **Figures**: `figures/`",
        "- **Logs**: `logs/experiment.log`",
        "- **Resolved Config**: `config_resolved.yaml`",
        "",
        "## Methodological Notes",
        "",
        "- Labels were used **only** for coloring visualizations, not for training.",
        "- Visual analysis of latent space is the **primary evidence**.",
        "- Quantitative metrics are **subsidiary** and must not replace visual analysis.",
        "- PCA explains linear variance; UMAP and t-SNE do NOT provide variance ratios.",
        "- Test set metrics must only be reported in the **final evaluation phase**.",
        "",
        "---",
        "*Generated automatically by vae_project*",
    ]

    summary_text = "\n".join(lines)
    summary_path = run_dir / "run_summary.md"

    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write(summary_text)

    logger.info("Run summary saved: %s", summary_path)
    return summary_path


def generate_comparison_report(
    results_df: Any,
    report_type: str,
    output_path: Path,
    is_quick_mode: bool = False,
) -> Path:
    """Generate a Markdown comparison report.

    Args:
        results_df: DataFrame with comparison results.
        report_type: Type of comparison ('topology', 'latent_dimensions').
        output_path: Output Markdown file path.
        is_quick_mode: Whether to add quick-mode warning.

    Returns:
        Path to the generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quick_note = ""
    if is_quick_mode:
        quick_note = (
            "> ⚠️ **QUICK MODE**: Results below are from smoke testing and "
            "must NOT be used in scientific publications.\n\n"
        )

    lines = [
        f"# {report_type.replace('_', ' ').title()} Comparison Report",
        "",
        quick_note,
        "## Results Table",
        "",
        (
            results_df.to_markdown(index=False)
            if hasattr(results_df, "to_markdown")
            else str(results_df)
        ),
        "",
        "## Notes",
        "",
        "- Results are sorted by validation total loss (lower is better).",
        "- The best configuration is the top row.",
        "- Test set was NOT used for this comparison.",
        "",
        "---",
        "*Generated automatically by vae_project*",
    ]

    report_text = "\n".join(lines)
    with output_path.open("w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    return output_path
