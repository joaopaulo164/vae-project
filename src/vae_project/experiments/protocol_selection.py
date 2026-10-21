"""Winner selection of the factorial protocol (validation results only).

The rule is deterministic and uses nothing but values measured on the
validation split. Test metrics, projection appearance and projection metrics
never take part. Candidates are filtered criterion by criterion: the primary
metric first, then each tie-breaker in the configured order. Two numeric
values are equal when they differ by at most ``tolerance``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from vae_project.config.models import (
    ExperimentConfig,
    SelectionConfig,
    TieBreakerConfig,
)
from vae_project.experiments.experiment_result import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
)
from vae_project.experiments.protocol_models import (
    COMBINATION_COLUMNS,
    CombinationRecord,
    is_finite_number,
)
from vae_project.reporting.formatting import bullets, format_value

logger = logging.getLogger(__name__)

SELECTED_CONFIGURATION_FILENAME = "selected_configuration.yaml"
SELECTION_SUMMARY_FILENAME = "selection_summary.md"
FACTORIAL_TABLE_FILENAME = "factorial_comparison.csv"
MINIMIZE = "minimize"
ASCENDING = "ascending"
TEXT_METRIC = "topology_name"


@dataclass
class ExcludedCombination:
    """A combination that could not take part in the selection."""

    key: str
    reason: str


@dataclass
class SelectionOutcome:
    """Result of the selection.

    Attributes:
        winner: Selected record, or None when no combination was valid.
        candidates: Valid records ordered by the primary metric.
        excluded: Combinations left out, with the reason.
        criteria_used: Criteria that actually decided between candidates.
        selected_at: ISO timestamp of the decision.
    """

    winner: CombinationRecord | None
    candidates: list[CombinationRecord] = field(default_factory=list)
    excluded: list[ExcludedCombination] = field(default_factory=list)
    criteria_used: list[str] = field(default_factory=list)
    selected_at: str = ""


def _metric_value(record: CombinationRecord, metric: str) -> object:
    """Return the value of a selection metric for a record."""
    return getattr(record, metric)


def exclusion_reason(
    record: CombinationRecord,
    protocol_dir: Path,
    selection: SelectionConfig,
    allow_partial: bool,
) -> str | None:
    """Return why a record cannot be selected, or None when it is valid.

    Args:
        record: Record of a grid combination.
        protocol_dir: Protocol run directory (to verify the checkpoint).
        selection: Selection rule.
        allow_partial: Accept combinations that finished training but lost
            some evidence (``partial``).
    """
    if record.status not in (STATUS_COMPLETED, STATUS_PARTIAL):
        return f"execution status is '{record.status}'"
    if record.status == STATUS_PARTIAL and not allow_partial:
        return "execution status is 'partial' and partial results are not allowed"
    if not is_finite_number(_metric_value(record, selection.primary_metric)):
        return f"{selection.primary_metric} is missing or not finite"
    if record.checkpoint_path is None or not (protocol_dir / record.checkpoint_path).is_file():
        return "best checkpoint is missing"
    return None


def _numeric_sort_value(value: object, minimize: bool) -> float:
    """Map a metric value to a number where smaller is better (None is worst)."""
    if not is_finite_number(value):
        return float("inf")
    return float(value) if minimize else -float(value)  # type: ignore[arg-type]


def _filter_numeric(
    records: list[CombinationRecord],
    metric: str,
    direction: str,
    tolerance: float,
) -> list[CombinationRecord]:
    """Keep the records whose value is within ``tolerance`` of the best one."""
    minimize = direction == MINIMIZE
    scores = [_numeric_sort_value(_metric_value(item, metric), minimize) for item in records]
    best = min(scores)
    return [item for item, score in zip(records, scores, strict=True) if score - best <= tolerance]


def _filter_text(
    records: list[CombinationRecord],
    metric: str,
    direction: str,
) -> list[CombinationRecord]:
    """Keep the records holding the first (or last) text value."""
    values = [str(_metric_value(item, metric)) for item in records]
    chosen = min(values) if direction == ASCENDING else max(values)
    return [item for item, value in zip(records, values, strict=True) if value == chosen]


def _apply_criterion(
    records: list[CombinationRecord],
    criterion: TieBreakerConfig,
    tolerance: float,
) -> list[CombinationRecord]:
    """Apply one ordering criterion to the surviving candidates."""
    if criterion.metric == TEXT_METRIC:
        return _filter_text(records, criterion.metric, criterion.direction)
    return _filter_numeric(records, criterion.metric, criterion.direction, tolerance)


def select_winner(
    records: list[CombinationRecord],
    selection: SelectionConfig,
    protocol_dir: Path,
    allow_partial: bool = True,
) -> SelectionOutcome:
    """Select the winning combination from validation results.

    Args:
        records: One record per combination that ran.
        selection: Primary metric, direction, tolerance and tie-breakers.
        protocol_dir: Protocol run directory.
        allow_partial: Accept ``partial`` combinations as candidates.

    Returns:
        SelectionOutcome; ``winner`` is None when no combination is valid.
    """
    excluded: list[ExcludedCombination] = []
    candidates: list[CombinationRecord] = []
    for record in records:
        reason = exclusion_reason(record, protocol_dir, selection, allow_partial)
        if reason is None:
            candidates.append(record)
        else:
            excluded.append(ExcludedCombination(record.key, reason))
    outcome = SelectionOutcome(
        winner=None,
        excluded=excluded,
        selected_at=datetime.now().isoformat(timespec="seconds"),
    )
    if not candidates:
        return outcome

    minimize = selection.direction == MINIMIZE
    outcome.candidates = sorted(
        candidates,
        key=lambda item: (
            _numeric_sort_value(_metric_value(item, selection.primary_metric), minimize),
            item.topology_name,
            item.latent_dimension,
        ),
    )
    surviving = _filter_numeric(
        outcome.candidates,
        selection.primary_metric,
        selection.direction,
        selection.tolerance,
    )
    outcome.criteria_used.append(selection.primary_metric)
    for tie_breaker in selection.tie_breakers:
        if len(surviving) == 1:
            break
        narrowed = _apply_criterion(surviving, tie_breaker, selection.tolerance)
        if len(narrowed) < len(surviving):
            outcome.criteria_used.append(tie_breaker.metric)
        surviving = narrowed
    surviving.sort(key=lambda item: (item.topology_name, item.latent_dimension))
    outcome.winner = surviving[0]
    return outcome


def write_factorial_table(records: list[CombinationRecord], selection_dir: Path) -> Path:
    """Write ``factorial_comparison.csv`` with one row per finished combination.

    Args:
        records: Combination records (any status).
        selection_dir: ``selection`` directory of the protocol run.

    Returns:
        Path of the CSV file.
    """
    selection_dir.mkdir(parents=True, exist_ok=True)
    rows = [record.to_row() for record in records]
    table = pd.DataFrame(rows, columns=list(COMBINATION_COLUMNS))
    path = selection_dir / FACTORIAL_TABLE_FILENAME
    table.to_csv(path, index=False)
    logger.info("Factorial comparison saved: %s (%d rows)", path, len(table))
    return path


def write_selected_configuration(
    outcome: SelectionOutcome,
    selection: SelectionConfig,
    config: ExperimentConfig,
    selection_dir: Path,
) -> Path:
    """Write ``selected_configuration.yaml`` for the winning combination.

    Args:
        outcome: Selection outcome with a winner.
        selection: Selection rule that produced it.
        config: Protocol configuration (seed, dataset).
        selection_dir: ``selection`` directory of the protocol run.

    Returns:
        Path of the YAML file. Paths inside are relative to the protocol run.

    Raises:
        ValueError: If the outcome has no winner.
    """
    winner = outcome.winner
    if winner is None:
        raise ValueError("cannot write a selected configuration without a winner")
    payload = {
        "dataset": winner.dataset,
        "topology_name": winner.topology_name,
        "hidden_dimensions": list(winner.hidden_dimensions),
        "latent_dimension": winner.latent_dimension,
        "seed": config.reproducibility.global_seed,
        "selected_metric": selection.primary_metric,
        "selected_value": _metric_value(winner, selection.primary_metric),
        "direction": selection.direction,
        "tolerance": selection.tolerance,
        "tie_breakers": [
            {"metric": item.metric, "direction": item.direction} for item in selection.tie_breakers
        ],
        "criteria_that_decided": list(outcome.criteria_used),
        "best_epoch": winner.best_epoch,
        "checkpoint_path": winner.checkpoint_path,
        "configuration_path": f"{winner.run_directory}/config_resolved.yaml",
        "run_summary_path": winner.run_summary_path,
        "manifest_path": winner.manifest_path,
        "selected_at": outcome.selected_at,
        "selection_data": "validation only (the test split was not used)",
    }
    path = selection_dir / SELECTED_CONFIGURATION_FILENAME
    selection_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    return path


def _candidate_table(outcome: SelectionOutcome) -> list[str]:
    """Return the Markdown table of valid candidates."""
    lines = [
        "| rank | topology | latent_dim | parameters | best_epoch | val_total_loss "
        "| val_reconstruction | val_kl | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, record in enumerate(outcome.candidates, start=1):
        lines.append(
            f"| {rank} | {record.topology_name} | {record.latent_dimension} "
            f"| {format_value(record.parameter_count)} | {format_value(record.best_epoch)} "
            f"| {format_value(record.validation_total_loss)} "
            f"| {format_value(record.validation_reconstruction_loss)} "
            f"| {format_value(record.validation_kl_divergence)} | {record.status} |"
        )
    return lines


def write_selection_summary(
    outcome: SelectionOutcome,
    selection: SelectionConfig,
    selection_dir: Path,
    synthetic_data: bool,
) -> Path:
    """Write ``selection_summary.md`` (English, no scientific conclusion).

    Args:
        outcome: Selection outcome.
        selection: Selection rule.
        selection_dir: ``selection`` directory of the protocol run.
        synthetic_data: Whether the run used synthetic data.

    Returns:
        Path of the Markdown file.
    """
    winner = outcome.winner
    lines = ["# Selection Summary", ""]
    if synthetic_data:
        lines += ["**SYNTHETIC DATA: smoke test, non-scientific results.**", ""]
    lines += [
        "## Rule",
        "",
        f"- **Primary metric**: `{selection.primary_metric}` ({selection.direction})",
        f"- **Tolerance for numeric equality**: {selection.tolerance}",
        "- **Tie-breakers (in order)**: "
        + ", ".join(f"`{item.metric}` ({item.direction})" for item in selection.tie_breakers),
        "- **Data used**: validation split only. The test split, visual appearance and "
        "projection metrics are never used for selection.",
        "",
        "## Result",
        "",
    ]
    if winner is None:
        lines += [
            "**No valid combination was found. The protocol stops before the test "
            "evaluation and the supervised extension.**",
            "",
            "Action: inspect `run_summary.md` and `logs/experiment.log` of the failed "
            "combinations listed below, fix the cause, and start a new protocol run.",
        ]
    else:
        lines += [
            f"- **Selected topology**: {winner.topology_name} = {winner.hidden_dimensions}",
            f"- **Selected latent dimension**: {winner.latent_dimension}",
            f"- **{selection.primary_metric}**: "
            f"{format_value(_metric_value(winner, selection.primary_metric))}",
            f"- **Best epoch**: {format_value(winner.best_epoch)}",
            f"- **Criteria that decided**: {', '.join(outcome.criteria_used)}",
            f"- **Checkpoint**: `{format_value(winner.checkpoint_path)}`",
            f"- **Selected at**: {outcome.selected_at}",
            "",
            "## Valid candidates",
            "",
            *_candidate_table(outcome),
        ]
    lines += ["", "## Excluded combinations", ""]
    lines += bullets([f"{item.key}: {item.reason}" for item in outcome.excluded])
    lines += ["", "---", "*Generated automatically by vae_project*", ""]
    path = selection_dir / SELECTION_SUMMARY_FILENAME
    selection_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
