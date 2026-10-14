"""Unit tests for the factorial grid, its configuration validation and the state."""

from __future__ import annotations

from pathlib import Path

import pytest

from vae_project.config.loader import ConfigurationError, load_config, load_config_from_dict
from vae_project.config.models import ExperimentConfig, TieBreakerConfig
from vae_project.config.validation import validate_config
from vae_project.experiments.protocol_models import (
    FactorialCombination,
    ProtocolError,
    build_factorial_grid,
)
from vae_project.experiments.protocol_state import (
    PHASE_GRID,
    STATUS_COMPLETED,
    STATUS_FAILED,
    ProtocolState,
    ProtocolStateError,
    compute_config_hash,
    load_state,
    save_state,
    validate_resume,
)
from vae_project.utils.json_io import to_jsonable, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_CONFIGS = (
    "configs/factorial_protocol_mnist.yaml",
    "configs/factorial_protocol_fashion_mnist.yaml",
)
EXPECTED_HIDDEN = {"small": (256, 128), "medium": (512, 256), "deep": (512, 256, 128)}
EXPECTED_DIMENSIONS = [2, 4, 8, 16, 32]
LEGACY_DIMENSIONS = [2, 8, 16, 32]


def _factorial_config(**overrides: object) -> ExperimentConfig:
    config = ExperimentConfig()
    config.factorial_comparison.enabled = True
    for key, value in overrides.items():
        setattr(config.factorial_comparison, key, value)
    return config


@pytest.mark.parametrize("config_path", PROTOCOL_CONFIGS)
def test_protocol_configs_define_the_default_grid_with_4d(config_path: str) -> None:
    config = load_config(PROJECT_ROOT / config_path)
    grid = build_factorial_grid(config)
    expected_count = len(EXPECTED_HIDDEN) * len(EXPECTED_DIMENSIONS)

    assert config.factorial_comparison.latent_dimensions == EXPECTED_DIMENSIONS
    assert len(grid) == expected_count
    assert len({combination.key for combination in grid}) == expected_count
    assert {(c.topology_name, c.latent_dimension) for c in grid} == {
        (name, dim) for name in EXPECTED_HIDDEN for dim in EXPECTED_DIMENSIONS
    }
    for combination in grid:
        assert combination.hidden_dimensions == EXPECTED_HIDDEN[combination.topology_name]


def test_default_grid_has_fifteen_combinations_including_each_4d_topology() -> None:
    grid = build_factorial_grid(_factorial_config())
    four_dimensional = [c for c in grid if c.latent_dimension == 4]

    assert len(grid) == 15
    assert [c.topology_name for c in four_dimensional] == ["small", "medium", "deep"]
    assert [c.key for c in four_dimensional] == [
        "small/latent_dim_4",
        "medium/latent_dim_4",
        "deep/latent_dim_4",
    ]
    assert four_dimensional[0].relative_directory == Path("grid/small/latent_dim_4")


def test_grid_order_is_deterministic_and_topology_major() -> None:
    first = [c.key for c in build_factorial_grid(_factorial_config())]
    second = [c.key for c in build_factorial_grid(_factorial_config())]

    assert first == second
    assert first[:6] == [
        "small/latent_dim_2",
        "small/latent_dim_4",
        "small/latent_dim_8",
        "small/latent_dim_16",
        "small/latent_dim_32",
        "medium/latent_dim_2",
    ]
    assert first[-1] == "deep/latent_dim_32"


@pytest.mark.parametrize(
    "dimensions",
    [LEGACY_DIMENSIONS, [4], [2, 4], [2, 3, 5, 7, 11, 13]],
)
def test_grid_size_follows_the_configured_dimensions(dimensions: list[int]) -> None:
    config = _factorial_config(latent_dimensions=list(dimensions))
    grid = build_factorial_grid(config)

    assert validate_config(config) == []
    assert len(grid) == len(config.factorial_comparison.topologies) * len(dimensions)
    assert {c.latent_dimension for c in grid} == set(dimensions)


def test_legacy_yaml_with_four_dimensions_keeps_twelve_combinations() -> None:
    config = load_config_from_dict(
        {
            "factorial_comparison": {
                "enabled": True,
                "latent_dimensions": LEGACY_DIMENSIONS,
            }
        }
    )
    grid = build_factorial_grid(config)

    assert len(grid) == 12
    assert 4 not in {c.latent_dimension for c in grid}


def test_validation_accepts_four_dimensional_latent_space() -> None:
    config = _factorial_config(latent_dimensions=[2, 4])

    assert validate_config(config) == []


@pytest.mark.parametrize("config_path", PROTOCOL_CONFIGS)
def test_protocol_configs_keep_one_seed_and_an_unsupervised_grid(config_path: str) -> None:
    config = load_config(PROJECT_ROOT / config_path)

    assert config.reproducibility.global_seed == 42
    assert config.reproducibility.deterministic_mode is True
    assert config.splits.seed == 42
    assert config.projections.umap.random_state == 42
    assert config.projections.tsne.random_state == 42
    assert config.supervised_component.enabled is False
    assert config.supervised_extension.enabled is True
    assert config.supervised_extension.execution_mode == "automatic"
    assert config.final_evaluation.split == "test"
    assert config.final_evaluation.retrain_selected_configuration is False
    assert config.factorial_comparison.selection.primary_metric == "validation_total_loss"


def test_grid_is_topology_major_and_ordered() -> None:
    grid = build_factorial_grid(_factorial_config())

    assert [c.key for c in grid][:6] == [
        "small/latent_dim_2",
        "small/latent_dim_4",
        "small/latent_dim_8",
        "small/latent_dim_16",
        "small/latent_dim_32",
        "medium/latent_dim_2",
    ]
    assert grid[0].relative_directory == Path("grid/small/latent_dim_2")


def test_combination_exposes_its_topology() -> None:
    combination = FactorialCombination("deep", (512, 256, 128), 8)

    assert combination.topology.name == "deep"
    assert combination.topology.hidden_dimensions == [512, 256, 128]


def test_grid_rejects_unknown_topology_and_duplicates() -> None:
    with pytest.raises(ProtocolError, match="Unknown topology"):
        build_factorial_grid(_factorial_config(topologies=["huge"]))
    with pytest.raises(ProtocolError, match="duplicated"):
        build_factorial_grid(_factorial_config(topologies=["small", "small"]))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"topologies": []}, "non-empty"),
        ({"topologies": ["tiny"]}, "unknown topology"),
        ({"topologies": ["small", "small"]}, "duplicates"),
        ({"latent_dimensions": []}, "latent_dimensions"),
        ({"latent_dimensions": [2, 2, 8]}, "duplicate"),
        ({"latent_dimensions": [1, 8]}, "2"),
        ({"latent_dimensions": [2, 8.5]}, "integer"),
        ({"latent_dimensions": [2, "8"]}, "integer"),
    ],
)
def test_validation_rejects_invalid_grids(overrides: dict[str, object], message: str) -> None:
    errors = validate_config(_factorial_config(**overrides))

    assert any(message in error for error in errors), errors


def test_validation_rejects_unknown_selection_metric_and_tie_breaker() -> None:
    config = _factorial_config()
    config.factorial_comparison.selection.primary_metric = "test_total_loss"
    assert any("primary_metric" in error for error in validate_config(config))

    config = _factorial_config()
    config.factorial_comparison.selection.tie_breakers = [TieBreakerConfig("accuracy", "minimize")]
    assert any("tie_breakers" in error for error in validate_config(config))

    config = _factorial_config()
    config.factorial_comparison.selection.tie_breakers = [
        TieBreakerConfig("topology_name", "minimize")
    ]
    assert any("tie_breakers" in error for error in validate_config(config))


def test_validation_rejects_non_test_final_split_and_missing_classification() -> None:
    config = _factorial_config()
    config.final_evaluation.split = "val"
    assert any("final_evaluation.split" in error for error in validate_config(config))

    config = _factorial_config()
    config.supervised_extension.enabled = True
    config.supervised_extension.classification_loss_weight = 0.0
    assert any("classification" in error for error in validate_config(config))

    config = _factorial_config()
    config.supervised_extension.enabled = True
    config.supervised_component.hidden_dimensions = []
    assert any("classification head" in error for error in validate_config(config))


def test_validation_requires_an_unsupervised_grid_and_no_retraining() -> None:
    config = _factorial_config()
    config.supervised_component.enabled = True
    assert any("unsupervised" in error for error in validate_config(config))

    config = _factorial_config()
    config.final_evaluation.retrain_selected_configuration = True
    assert any("retrain" in error for error in validate_config(config))


def test_loader_rejects_an_invalid_protocol_yaml_block() -> None:
    with pytest.raises(ConfigurationError):
        load_config_from_dict(
            {
                "factorial_comparison": {"enabled": True, "topologies": []},
            }
        )


def test_default_configuration_keeps_the_protocol_disabled() -> None:
    config = ExperimentConfig()

    assert config.factorial_comparison.enabled is False
    assert validate_config(config) == []


def _state(config: ExperimentConfig) -> ProtocolState:
    return ProtocolState(
        run_id="run_test",
        config_hash=compute_config_hash(config),
        dataset=config.dataset.name.value,
    )


def test_state_round_trip_and_combination_buckets(tmp_path: Path) -> None:
    config = _factorial_config()
    state = _state(config)
    state.pending_combinations = ["small/latent_dim_2", "small/latent_dim_8"]
    state.current_combination = "small/latent_dim_2"
    state.mark_combination("small/latent_dim_2", succeeded=True)
    state.mark_combination("small/latent_dim_8", succeeded=False)
    state.add_warning("w")
    state.add_warning("w")
    state.phase = PHASE_GRID

    save_state(state, tmp_path)
    loaded = load_state(tmp_path)

    assert loaded.completed_combinations == ["small/latent_dim_2"]
    assert loaded.failed_combinations == ["small/latent_dim_8"]
    assert loaded.pending_combinations == []
    assert loaded.current_combination is None
    assert loaded.warnings == ["w"]
    assert loaded.phase == PHASE_GRID


def test_config_hash_distinguishes_the_legacy_and_the_default_grid() -> None:
    default = _factorial_config()
    legacy = _factorial_config(latent_dimensions=list(LEGACY_DIMENSIONS))

    assert compute_config_hash(default) != compute_config_hash(legacy)


def test_resume_rejects_a_legacy_twelve_combination_run_for_a_fifteen_combination_grid() -> None:
    legacy = _factorial_config(latent_dimensions=list(LEGACY_DIMENSIONS))
    legacy_state = _state(legacy)
    current = _factorial_config()

    with pytest.raises(ProtocolStateError, match="configuration differs"):
        validate_resume(legacy_state, current)
    validate_resume(legacy_state, legacy)


def test_config_hash_ignores_run_name_but_not_scientific_values() -> None:
    config = _factorial_config()
    renamed = _factorial_config()
    renamed.run_name = "something_else"
    changed = _factorial_config()
    changed.training.batch_size = config.training.batch_size + 1

    assert compute_config_hash(config) == compute_config_hash(renamed)
    assert compute_config_hash(config) != compute_config_hash(changed)


def test_resume_rejects_other_configuration_dataset_and_finished_runs() -> None:
    config = _factorial_config()
    state = _state(config)
    validate_resume(state, config)

    changed = _factorial_config()
    changed.optimizer.learning_rate = 0.5
    with pytest.raises(ProtocolStateError, match="configuration differs"):
        validate_resume(state, changed)

    state.dataset = "fashion_mnist"
    with pytest.raises(ProtocolStateError, match="dataset"):
        validate_resume(state, config)

    finished = _state(config)
    finished.status = STATUS_COMPLETED
    with pytest.raises(ProtocolStateError, match="already finished"):
        validate_resume(finished, config)


def test_resume_accepts_an_interrupted_or_failed_run() -> None:
    config = _factorial_config()
    state = _state(config)
    state.status = STATUS_FAILED

    validate_resume(state, config)


def test_load_state_reports_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ProtocolStateError, match=r"protocol_state\.json"):
        load_state(tmp_path)

    (tmp_path / "protocol_state.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ProtocolStateError, match="Cannot read"):
        load_state(tmp_path)

    write_json(tmp_path / "protocol_state.json", {"unrelated": 1})
    with pytest.raises(ProtocolStateError, match="does not look like"):
        load_state(tmp_path)


def test_json_io_turns_non_finite_numbers_into_null(tmp_path: Path) -> None:
    payload = {"value": float("nan"), "path": tmp_path, "items": (1, float("inf"))}

    converted = to_jsonable(payload)

    assert converted["value"] is None
    assert converted["items"] == [1, None]
    assert converted["path"] == tmp_path.as_posix()
