"""Unit tests for the validation-only winner selection and its tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from vae_project.config.models import ExperimentConfig, SelectionConfig, TieBreakerConfig
from vae_project.experiments.experiment_result import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
)
from vae_project.experiments.protocol_models import COMBINATION_COLUMNS, CombinationRecord
from vae_project.experiments.protocol_selection import (
    exclusion_reason,
    select_winner,
    write_factorial_table,
    write_selected_configuration,
    write_selection_summary,
)

HIDDEN = {"small": [256, 128], "medium": [512, 256], "deep": [512, 256, 128]}


def _record(
    protocol_dir: Path,
    topology: str,
    latent_dimension: int,
    total: float | None,
    reconstruction: float | None = 100.0,
    parameters: int | None = 1000,
    status: str = STATUS_COMPLETED,
    with_checkpoint: bool = True,
) -> CombinationRecord:
    directory = f"grid/{topology}/latent_dim_{latent_dimension}"
    checkpoint = f"{directory}/checkpoints/best_model.pt"
    if with_checkpoint:
        (protocol_dir / checkpoint).parent.mkdir(parents=True, exist_ok=True)
        (protocol_dir / checkpoint).write_bytes(b"checkpoint")
    return CombinationRecord(
        dataset="mnist",
        experiment_type="unsupervised",
        topology_name=topology,
        hidden_dimensions=HIDDEN[topology],
        latent_dimension=latent_dimension,
        seed=42,
        status=status,
        run_directory=directory,
        parameter_count=parameters,
        best_epoch=3,
        validation_reconstruction_loss=reconstruction,
        validation_kl_divergence=1.0,
        validation_total_loss=total,
        checkpoint_path=checkpoint if with_checkpoint else None,
        run_summary_path=f"{directory}/run_summary.md",
        manifest_path=f"{directory}/artifact_manifest.json",
    )


def _winner_key(records: list[CombinationRecord], protocol_dir: Path, **kwargs: object) -> str:
    outcome = select_winner(records, SelectionConfig(**kwargs), protocol_dir)  # type: ignore[arg-type]
    assert outcome.winner is not None
    return outcome.winner.key


def test_lowest_validation_total_loss_wins(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "small", 2, 120.0),
        _record(tmp_path, "medium", 8, 95.0),
        _record(tmp_path, "deep", 16, 100.0),
    ]

    assert _winner_key(records, tmp_path) == "medium/latent_dim_8"


def test_maximize_direction_is_supported(tmp_path: Path) -> None:
    records = [_record(tmp_path, "small", 2, 120.0), _record(tmp_path, "medium", 8, 95.0)]

    assert _winner_key(records, tmp_path, direction="maximize") == "small/latent_dim_2"


def test_tie_is_broken_by_reconstruction_loss_first(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "small", 2, 100.0, reconstruction=90.0),
        _record(tmp_path, "medium", 8, 100.0 + 5e-7, reconstruction=80.0),
    ]

    outcome = select_winner(records, SelectionConfig(), tmp_path)

    assert outcome.winner is not None
    assert outcome.winner.key == "medium/latent_dim_8"
    assert outcome.criteria_used == ["validation_total_loss", "validation_reconstruction_loss"]


def test_tie_breaker_order_parameters_then_dimension_then_name(tmp_path: Path) -> None:
    same = {"total": 100.0, "reconstruction": 50.0}
    by_parameters = [
        _record(tmp_path, "medium", 8, parameters=900, **same),
        _record(tmp_path, "small", 2, parameters=1200, **same),
    ]
    assert _winner_key(by_parameters, tmp_path) == "medium/latent_dim_8"

    by_dimension = [
        _record(tmp_path, "medium", 16, parameters=900, **same),
        _record(tmp_path, "small", 8, parameters=900, **same),
    ]
    assert _winner_key(by_dimension, tmp_path) == "small/latent_dim_8"

    by_name = [
        _record(tmp_path, "small", 8, parameters=900, **same),
        _record(tmp_path, "deep", 8, parameters=900, **same),
    ]
    assert _winner_key(by_name, tmp_path) == "deep/latent_dim_8"


def test_a_difference_above_the_tolerance_is_not_a_tie(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "small", 2, 100.0, reconstruction=10.0),
        _record(tmp_path, "medium", 8, 100.0 + 1e-3, reconstruction=1.0),
    ]

    assert _winner_key(records, tmp_path) == "small/latent_dim_2"
    assert _winner_key(records, tmp_path, tolerance=1e-2) == "medium/latent_dim_8"


def test_selection_is_deterministic_regardless_of_input_order(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "small", 8, 100.0),
        _record(tmp_path, "deep", 8, 100.0),
        _record(tmp_path, "medium", 2, 100.0),
    ]

    forward = _winner_key(records, tmp_path)
    backward = _winner_key(list(reversed(records)), tmp_path)

    assert forward == backward


def test_failed_incomplete_and_non_finite_runs_are_excluded(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "small", 2, 1.0, status=STATUS_FAILED),
        _record(tmp_path, "small", 8, None),
        _record(tmp_path, "small", 16, float("nan")),
        _record(tmp_path, "small", 32, float("inf")),
        _record(tmp_path, "medium", 2, 5.0, with_checkpoint=False),
        _record(tmp_path, "medium", 8, 6.0, status="running"),
        _record(tmp_path, "deep", 2, 300.0),
    ]

    outcome = select_winner(records, SelectionConfig(), tmp_path)

    assert outcome.winner is not None
    assert outcome.winner.key == "deep/latent_dim_2"
    assert {item.key for item in outcome.excluded} == {
        "small/latent_dim_2",
        "small/latent_dim_8",
        "small/latent_dim_16",
        "small/latent_dim_32",
        "medium/latent_dim_2",
        "medium/latent_dim_8",
    }


def test_partial_runs_follow_the_noncritical_error_policy(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "small", 2, 50.0, status=STATUS_PARTIAL),
        _record(tmp_path, "medium", 8, 80.0),
    ]

    allowed = select_winner(records, SelectionConfig(), tmp_path, allow_partial=True)
    denied = select_winner(records, SelectionConfig(), tmp_path, allow_partial=False)

    assert allowed.winner is not None
    assert allowed.winner.key == "small/latent_dim_2"
    assert denied.winner is not None
    assert denied.winner.key == "medium/latent_dim_8"
    assert "partial" in (exclusion_reason(records[0], tmp_path, SelectionConfig(), False) or "")


def test_no_valid_combination_yields_no_winner(tmp_path: Path) -> None:
    records = [_record(tmp_path, "small", 2, None, status=STATUS_FAILED)]

    outcome = select_winner(records, SelectionConfig(), tmp_path)

    assert outcome.winner is None
    assert outcome.candidates == []
    assert len(outcome.excluded) == 1


def test_selection_never_reads_test_or_projection_values(tmp_path: Path) -> None:
    record = _record(tmp_path, "small", 2, 10.0)
    record.pca_explained_variance = (0.9, 0.05, 0.95)
    record.latent_plots = {"umap": "irrelevant.png"}
    other = _record(tmp_path, "medium", 8, 11.0)
    other.pca_explained_variance = (0.1, 0.05, 0.15)

    assert _winner_key([record, other], tmp_path) == "small/latent_dim_2"
    assert not hasattr(record, "test_total_loss")


def test_factorial_table_has_one_row_per_combination_and_missing_values(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "small", 2, 10.0),
        _record(tmp_path, "medium", 8, None, status=STATUS_FAILED, with_checkpoint=False),
    ]

    path = write_factorial_table(records, tmp_path / "selection")
    table = pd.read_csv(path)

    assert list(table.columns) == list(COMBINATION_COLUMNS)
    assert len(table) == 2
    assert table["validation_classification_loss"].isna().all()
    assert pd.isna(table.loc[1, "validation_total_loss"])
    assert pd.isna(table.loc[1, "pca_component_1_explained_variance"])
    assert table.loc[0, "execution_status"] == STATUS_COMPLETED


def test_selected_configuration_and_summary_files(tmp_path: Path) -> None:
    records = [_record(tmp_path, "small", 2, 10.0), _record(tmp_path, "deep", 8, 12.0)]
    rule = SelectionConfig()
    outcome = select_winner(records, rule, tmp_path)
    selection_dir = tmp_path / "selection"
    config = ExperimentConfig()

    yaml_path = write_selected_configuration(outcome, rule, config, selection_dir)
    summary_path = write_selection_summary(outcome, rule, selection_dir, synthetic_data=True)
    stored = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")

    assert stored["topology_name"] == "small"
    assert stored["latent_dimension"] == 2
    assert stored["hidden_dimensions"] == [256, 128]
    assert stored["selected_metric"] == "validation_total_loss"
    assert stored["selected_value"] == 10.0
    assert stored["seed"] == 42
    assert stored["checkpoint_path"].endswith("best_model.pt")
    assert stored["selected_at"]
    first_tie_breaker = stored["tie_breakers"][0]
    assert first_tie_breaker["metric"] == "validation_reconstruction_loss"
    assert "SYNTHETIC DATA" in summary
    assert "validation split only" in summary


def test_summary_of_a_failed_selection_is_actionable(tmp_path: Path) -> None:
    outcome = select_winner(
        [_record(tmp_path, "small", 2, None, status=STATUS_FAILED)], SelectionConfig(), tmp_path
    )

    summary_path = write_selection_summary(
        outcome, SelectionConfig(), tmp_path / "selection", synthetic_data=False
    )
    text = summary_path.read_text(encoding="utf-8")

    assert "No valid combination" in text
    assert "Action:" in text
    with pytest.raises(ValueError, match="without a winner"):
        write_selected_configuration(
            outcome, SelectionConfig(), ExperimentConfig(), tmp_path / "selection"
        )


def test_custom_tie_breaker_list_is_respected(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "small", 32, 100.0, parameters=500),
        _record(tmp_path, "medium", 2, 100.0, parameters=900),
    ]
    rule = SelectionConfig(tie_breakers=[TieBreakerConfig("latent_dimension", "minimize")])

    outcome = select_winner(records, rule, tmp_path)

    assert outcome.winner is not None
    assert outcome.winner.key == "medium/latent_dim_2"
