"""Integration tests for the factorial scientific protocol.

Everything runs on CPU with tiny synthetic data. No dataset is downloaded
(sockets are blocked) and no real MNIST or Fashion-MNIST training happens. UMAP
is replaced by a test double in most tests because numba compilation is slow;
``test_real_pca_umap_tsne_smoke`` runs the real implementations on a tiny sample.

All outputs are written below a ``synthetic_smoke_test`` directory, which is
required for synthetic runs and keeps them apart from real results.
"""

from __future__ import annotations

import copy
import json
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from protocol_support import (
    REDUCED_DIMENSIONS,
    REDUCED_TOPOLOGIES,
    FakeUMAPProjection,
    make_protocol_config,
    make_synthetic_splits,
)

from vae_project import cli
from vae_project.experiments import (
    factorial_protocol,
    heldout_evaluation,
    latent_evidence,
    single_experiment,
)
from vae_project.experiments.factorial_protocol import ProtocolResult, run_factorial_protocol
from vae_project.experiments.protocol_state import ProtocolStateError, load_state
from vae_project.utils import paths
from vae_project.utils.paths import ArtifactError

pytestmark = pytest.mark.integration

FIGURE_CATEGORIES = ("training", "reconstructions", "generated_samples", "latent_space")
LATENT_PLOT_PREFIXES = ("latent_space_pca_", "latent_space_umap_", "latent_space_tsne_")
DIRECT_STEM = "latent_space_direct_2d"
REQUIRED_RUN_FILES = (
    "checkpoints/best_model.pt",
    "checkpoints/checkpoint_metadata.json",
    "latent_data/latent_representations.csv",
    "latent_data/latent_representations.npz",
    "tables/training_history.csv",
    "tables/training_history.json",
    "tables/projection_metrics.csv",
    "config_resolved.yaml",
    "metrics.json",
    "run_summary.md",
    "artifact_manifest.json",
)
LATENT_COLUMNS = (
    "sample_index",
    "split",
    "label",
    "class_name",
    "topology_name",
    "latent_dimension",
    "seed",
    "experiment_type",
    "execution_id",
    "latent_mean_0",
)


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is not allowed in the tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@dataclass
class Recorder:
    """Observations made while a protocol runs (test doubles around real code)."""

    trainers: list[Any] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    test_measurements: list[dict[str, Any]] = field(default_factory=list)


def _install_recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Wrap the trainer, the seed and the test-evaluation entry points."""
    recorder = Recorder()
    real_trainer = single_experiment.VAETrainer
    real_seed = single_experiment.set_global_seed
    real_evaluate = factorial_protocol.evaluate_on_test
    real_measure = heldout_evaluation.measure_test_losses

    class RecordingTrainer(real_trainer):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            recorder.trainers.append(self)

    def recording_seed(seed: int, *args: object, **kwargs: object) -> None:
        recorder.seeds.append(seed)
        real_seed(seed, *args, **kwargs)

    def recording_evaluate(*args: object, **kwargs: object) -> Any:
        recorder.evaluations.append(dict(kwargs))
        return real_evaluate(*args, **kwargs)

    def recording_measure(model: Any, head: Any, run_config: Any, data: Any) -> Any:
        recorder.test_measurements.append(
            {
                "topology": run_config.model.topology.name,
                "latent_dimension": run_config.model.latent_dimension,
                "supervised": head is not None,
            }
        )
        return real_measure(model, head, run_config, data)

    monkeypatch.setattr(single_experiment, "VAETrainer", RecordingTrainer)
    monkeypatch.setattr(single_experiment, "set_global_seed", recording_seed)
    monkeypatch.setattr(factorial_protocol, "evaluate_on_test", recording_evaluate)
    monkeypatch.setattr(heldout_evaluation, "measure_test_losses", recording_measure)
    return recorder


@pytest.fixture(autouse=True)
def _fake_umap_and_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latent_evidence, "UMAPProjection", FakeUMAPProjection)
    _block_network(monkeypatch)


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """Base directory for protocol runs; the name carries the required marker."""
    return tmp_path / "synthetic_smoke_test"


@dataclass
class CompletedProtocol:
    """A finished reduced protocol and everything observed while it ran."""

    result: ProtocolResult
    recorder: Recorder
    directory: Path


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> CompletedProtocol:
    """Run the reduced protocol (2 topologies x 2 dimensions) once for read-only tests."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(latent_evidence, "UMAPProjection", FakeUMAPProjection)
    _block_network(monkeypatch)
    recorder = _install_recorder(monkeypatch)
    try:
        base = tmp_path_factory.mktemp("synthetic_smoke_test")
        result = run_factorial_protocol(
            make_protocol_config(), base, run_name="reduced", splits=make_synthetic_splits()
        )
    finally:
        monkeypatch.undo()
    return CompletedProtocol(result, recorder, result.protocol_directory)


def _grid_dir(protocol: CompletedProtocol, topology: str, dimension: int) -> Path:
    return protocol.directory / "grid" / topology / f"latent_dim_{dimension}"


def _run(base: Path, **kwargs: Any) -> ProtocolResult:
    config = kwargs.pop("config", None) or make_protocol_config(
        topologies=["small"], dimensions=[2, 8], supervised_extension=False
    )
    return run_factorial_protocol(config, base, splits=make_synthetic_splits(), **kwargs)


def _files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern))


# --------------------------------------------------------------------------- grid


class TestGridExecution:
    def test_protocol_completes_with_the_reduced_grid(self, completed: CompletedProtocol) -> None:
        assert completed.result.status == "completed"
        assert len(completed.result.records) == len(REDUCED_TOPOLOGIES) * len(REDUCED_DIMENSIONS)
        assert {record.key for record in completed.result.records} == {
            f"{topology}/latent_dim_{dimension}"
            for topology in REDUCED_TOPOLOGIES
            for dimension in REDUCED_DIMENSIONS
        }

    def test_every_combination_has_its_own_checkpoint_and_required_files(
        self, completed: CompletedProtocol
    ) -> None:
        checkpoints = set()
        for topology in REDUCED_TOPOLOGIES:
            for dimension in REDUCED_DIMENSIONS:
                run_dir = _grid_dir(completed, topology, dimension)
                for name in REQUIRED_RUN_FILES:
                    path = run_dir / name
                    assert path.is_file(), path
                    assert path.stat().st_size > 0, path
                checkpoints.add(run_dir / "checkpoints" / "best_model.pt")
        assert len(checkpoints) == 4

    def test_each_combination_trains_a_new_model_with_the_same_seed(
        self, completed: CompletedProtocol
    ) -> None:
        unsupervised = [t for t in completed.recorder.trainers if t.supervised_head is None]

        assert len(unsupervised) == 4
        for attribute in ("model", "optimizer", "early_stopping"):
            assert len({id(getattr(trainer, attribute)) for trainer in unsupervised}) == 4
        assert all(trainer.scheduler is None for trainer in unsupervised)
        assert {tuple(t.model.hidden_dimensions) for t in unsupervised} == {
            (256, 128),
            (512, 256, 128),
        }
        assert {t.model.latent_dimension for t in unsupervised} == set(REDUCED_DIMENSIONS)
        assert set(completed.recorder.seeds) == {42}
        assert len(completed.recorder.seeds) >= 4

    def test_grid_runs_are_unsupervised_without_classification(
        self, completed: CompletedProtocol
    ) -> None:
        for record in completed.result.records:
            assert record.experiment_type == "unsupervised"
            assert record.validation_classification_loss is None
            config = yaml.safe_load(
                (completed.directory / record.run_directory / "config_resolved.yaml").read_text(
                    encoding="utf-8"
                )
            )
            assert config["supervised_component"]["enabled"] is False
            assert config["reproducibility"]["global_seed"] == 42
            assert config["factorial_comparison"]["enabled"] is False

    def test_only_topology_and_latent_dimension_vary_between_combinations(
        self, completed: CompletedProtocol
    ) -> None:
        configs = []
        for record in completed.result.records:
            path = completed.directory / record.run_directory / "config_resolved.yaml"
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            config["model"].pop("latent_dimension")
            config["model"]["topology"].pop("name")
            config["model"]["topology"].pop("hidden_dimensions")
            configs.append(config)

        assert all(config == configs[0] for config in configs)

    def test_runs_use_the_cpu(self, completed: CompletedProtocol) -> None:
        text = (_grid_dir(completed, "small", 2) / "run_summary.md").read_text(encoding="utf-8")

        assert "**Device**: cpu" in text
        assert all(
            next(t.model.parameters()).device.type == "cpu" for t in completed.recorder.trainers
        )


class TestEvidenceCategories:
    @pytest.mark.parametrize("topology", REDUCED_TOPOLOGIES)
    @pytest.mark.parametrize("dimension", REDUCED_DIMENSIONS)
    def test_four_figure_categories_in_png_and_svg(
        self, completed: CompletedProtocol, topology: str, dimension: int
    ) -> None:
        figures = _grid_dir(completed, topology, dimension) / "figures"

        for category in FIGURE_CATEGORIES:
            for suffix in ("png", "svg"):
                found = _files(figures / category, f"*.{suffix}")
                assert found, (category, suffix)
                assert all(path.stat().st_size > 0 for path in found)
        for name in ("training_total_loss", "training_loss_components"):
            assert (figures / "training" / f"{name}.png").is_file()
        assert (figures / "reconstructions" / "reconstruction_comparison.png").is_file()
        assert (figures / "generated_samples" / "generated_samples.png").is_file()

    @pytest.mark.parametrize("topology", REDUCED_TOPOLOGIES)
    def test_2d_uses_the_direct_plot_and_no_projection(
        self, completed: CompletedProtocol, topology: str
    ) -> None:
        latent_dir = _grid_dir(completed, topology, 2) / "figures" / "latent_space"
        names = {path.name for path in latent_dir.iterdir()}

        assert names == {f"{DIRECT_STEM}.png", f"{DIRECT_STEM}.svg"}
        assert not (
            _grid_dir(completed, topology, 2) / "tables" / "pca_explained_variance.csv"
        ).exists()

    @pytest.mark.parametrize("topology", REDUCED_TOPOLOGIES)
    def test_8d_has_pca_umap_tsne_and_pca_variance(
        self, completed: CompletedProtocol, topology: str
    ) -> None:
        run_dir = _grid_dir(completed, topology, 8)
        latent_dir = run_dir / "figures" / "latent_space"

        for prefix in LATENT_PLOT_PREFIXES:
            for suffix in ("png", "svg"):
                assert (latent_dir / f"{prefix}8d.{suffix}").is_file()
        assert (latent_dir / "pca_explained_variance_8d.png").is_file()
        assert not (latent_dir / f"{DIRECT_STEM}.png").exists()
        variance = pd.read_csv(run_dir / "tables" / "pca_explained_variance.csv")
        assert {"topology_name", "latent_dimension"} <= set(variance.columns)
        assert variance["explained_variance_ratio"].between(0, 1).all()
        assert not [c for c in variance.columns if "umap" in c or "tsne" in c]

    def test_pca_variance_is_recorded_only_for_pca(self, completed: CompletedProtocol) -> None:
        for record in completed.result.records:
            if record.latent_dimension == 2:
                assert record.pca_explained_variance is None
            else:
                first, second, cumulative = record.pca_explained_variance  # type: ignore[misc]
                assert cumulative == pytest.approx(first + second)

    def test_projection_metrics_identify_space_and_scope(
        self, completed: CompletedProtocol
    ) -> None:
        table = pd.read_csv(_grid_dir(completed, "small", 8) / "tables" / "projection_metrics.csv")

        assert {"latent_original", "pca_2d", "umap_2d", "tsne_2d"} <= set(table["space_type"])
        assert {"scope", "silhouette_score", "calinski_harabasz_index"} <= set(table.columns)
        assert "unavailable_metrics" in table.columns

    def test_latent_representations_carry_provenance(self, completed: CompletedProtocol) -> None:
        run_dir = _grid_dir(completed, "deep", 8)
        table = pd.read_csv(run_dir / "latent_data" / "latent_representations.csv")

        assert set(LATENT_COLUMNS) <= set(table.columns)
        assert set(table["topology_name"]) == {"deep"}
        assert set(table["latent_dimension"]) == {8}
        assert set(table["seed"]) == {42}
        assert set(table["experiment_type"]) == {"unsupervised"}
        assert set(table["split"]) == {"val"}
        assert [c for c in table.columns if c.startswith("latent_mean_")] == [
            f"latent_mean_{i}" for i in range(8)
        ]
        indices = pd.read_csv(run_dir / "latent_data" / "projection_sample_indices.csv")
        assert len(indices) > 0

    def test_training_history_covers_the_real_run(self, completed: CompletedProtocol) -> None:
        history = pd.read_csv(_grid_dir(completed, "small", 2) / "tables" / "training_history.csv")

        assert {"train_total_loss", "val_total_loss", "val_reconstruction_loss"} <= set(
            history.columns
        )
        assert len(history) >= 1
        flagged = history[history["is_best_epoch"]]
        assert not flagged.empty
        assert flagged["val_total_loss"].iloc[-1] == history["val_total_loss"].min()


class TestSharedSamples:
    def test_shared_files_exist(self, completed: CompletedProtocol) -> None:
        shared = completed.directory / "shared"

        for name in (
            "environment.json",
            "projection_sample_indices.csv",
            "reconstruction_sample_indices.csv",
            "generation_seed.json",
        ):
            assert (shared / name).is_file(), name
        assert (
            json.loads((shared / "generation_seed.json").read_text("utf-8"))["generation_seed"]
            == 42
        )

    def test_projection_samples_are_identical_across_combinations(
        self, completed: CompletedProtocol
    ) -> None:
        shared = pd.read_csv(completed.directory / "shared" / "projection_sample_indices.csv")
        for topology in REDUCED_TOPOLOGIES:
            local = pd.read_csv(
                _grid_dir(completed, topology, 8) / "latent_data" / "projection_sample_indices.csv"
            )
            assert sorted(local["sample_index"]) == sorted(shared["sample_index"])

    def test_reconstruction_indices_are_shared(self, completed: CompletedProtocol) -> None:
        shared = pd.read_csv(completed.directory / "shared" / "reconstruction_sample_indices.csv")

        assert len(shared) > 0
        assert shared["sample_index"].is_unique


# ----------------------------------------------------------------------- selection


class TestSelectionAndFinalEvaluation:
    def test_factorial_table_has_one_row_per_combination(
        self, completed: CompletedProtocol
    ) -> None:
        table = pd.read_csv(completed.directory / "selection" / "factorial_comparison.csv")

        assert len(table) == 4
        assert table[["topology_name", "latent_dimension"]].drop_duplicates().shape[0] == 4
        assert set(table["execution_status"]) == {"completed"}
        assert table["validation_classification_loss"].isna().all()
        assert set(table["experiment_type"]) == {"unsupervised"}
        assert (
            table.loc[table["latent_dimension"] == 2, "pca_cumulative_explained_variance"]
            .isna()
            .all()
        )
        assert (
            table.loc[table["latent_dimension"] == 8, "pca_cumulative_explained_variance"]
            .notna()
            .all()
        )

    def test_winner_has_the_lowest_validation_total_loss(
        self, completed: CompletedProtocol
    ) -> None:
        table = pd.read_csv(completed.directory / "selection" / "factorial_comparison.csv")
        selected = yaml.safe_load(
            (completed.directory / "selection" / "selected_configuration.yaml").read_text("utf-8")
        )
        best = table.sort_values("validation_total_loss").iloc[0]

        assert selected["topology_name"] == best["topology_name"]
        assert selected["latent_dimension"] == best["latent_dimension"]
        assert selected["selected_value"] == pytest.approx(best["validation_total_loss"])
        assert selected["selected_metric"] == "validation_total_loss"
        assert (completed.directory / selected["checkpoint_path"]).is_file()
        assert (completed.directory / "selection" / "selection_summary.md").is_file()

    def test_only_the_winner_is_evaluated_on_the_test_split(
        self, completed: CompletedProtocol
    ) -> None:
        selected = yaml.safe_load(
            (completed.directory / "selection" / "selected_configuration.yaml").read_text("utf-8")
        )
        measurements = completed.recorder.test_measurements

        assert [m["supervised"] for m in measurements] == [False, True]
        assert {(m["topology"], m["latent_dimension"]) for m in measurements} == {
            (selected["topology_name"], selected["latent_dimension"])
        }
        losers = [r for r in completed.result.records if r.key != completed.result.selection.winner.key]  # type: ignore[union-attr]
        assert len(losers) == 3
        for loser in losers:
            assert not (
                completed.directory / loser.run_directory / "final_evaluation_results.json"
            ).exists()

    def test_final_evaluation_artifacts_mark_the_test_phase(
        self, completed: CompletedProtocol
    ) -> None:
        directory = completed.directory / "final_evaluation"
        results = json.loads((directory / "final_evaluation_results.json").read_text("utf-8"))
        metrics = pd.read_csv(directory / "tables" / "final_test_metrics.csv")

        assert results["split"] == "test"
        assert results["phase"] == "final evaluation"
        assert results["selected_configuration"] is True
        assert results["experiment_type"] == "unsupervised"
        assert metrics.loc[0, "n_samples"] == 50
        assert pd.isna(metrics.loc[0, "test_classification_loss"])
        assert pd.isna(metrics.loc[0, "test_accuracy"])
        for category in ("reconstructions", "generated_samples", "latent_space"):
            assert _files(directory / "figures" / category, "*.png")
        assert (directory / "latent_data" / "latent_representations.csv").is_file()
        manifest = json.loads((directory / "artifact_manifest.json").read_text("utf-8"))
        assert manifest["split"] == "test"
        assert manifest["selected_configuration"] is True
        latent = pd.read_csv(directory / "latent_data" / "latent_representations.csv")
        assert set(latent["split"]) == {"test"}

    def test_final_evaluation_does_not_retrain_or_change_the_checkpoint(
        self, completed: CompletedProtocol
    ) -> None:
        winner = completed.result.selection.winner  # type: ignore[union-attr]
        evaluation_call = completed.recorder.evaluations[0]
        checkpoint_dir = Path(evaluation_call["checkpoint_dir"])

        assert checkpoint_dir == (completed.directory / winner.checkpoint_path).parent  # type: ignore[arg-type]
        assert not (
            completed.directory / "final_evaluation" / "checkpoints" / "best_model.pt"
        ).exists()
        assert len(completed.recorder.trainers) == 5

    def test_evaluation_uses_the_best_checkpoint_of_the_winner(
        self, completed: CompletedProtocol
    ) -> None:
        winner = completed.result.selection.winner  # type: ignore[union-attr]
        metadata = json.loads(
            (
                completed.directory
                / winner.run_directory
                / "checkpoints"
                / "checkpoint_metadata.json"
            ).read_text("utf-8")
        )
        results = json.loads(
            (completed.directory / "final_evaluation" / "final_evaluation_results.json").read_text(
                "utf-8"
            )
        )

        assert results["best_epoch"] == metadata["epoch"]
        assert results["validation_total_loss"] == pytest.approx(metadata["val_total_loss"])


# ------------------------------------------------------------------- supervised


class TestSupervisedExtension:
    def test_generated_configuration_matches_the_winner(self, completed: CompletedProtocol) -> None:
        winner = completed.result.selection.winner  # type: ignore[union-attr]
        generated = yaml.safe_load(
            (
                completed.directory / "generated_configs" / "supervised_selected_configuration.yaml"
            ).read_text("utf-8")
        )

        assert generated["supervised_component"]["enabled"] is True
        assert generated["model"]["topology"]["name"] == winner.topology_name
        assert generated["model"]["topology"]["hidden_dimensions"] == winner.hidden_dimensions
        assert generated["model"]["latent_dimension"] == winner.latent_dimension
        assert generated["reproducibility"]["global_seed"] == 42

    def test_training_is_a_new_model_with_only_the_allowed_differences(
        self, completed: CompletedProtocol
    ) -> None:
        winner = completed.result.selection.winner  # type: ignore[union-attr]
        supervised_trainer = [
            t for t in completed.recorder.trainers if t.supervised_head is not None
        ]
        unsupervised_winner = [
            t
            for t in completed.recorder.trainers
            if t.supervised_head is None
            and t.model.latent_dimension == winner.latent_dimension
            and t.model.topology_name == winner.topology_name
        ]

        assert len(supervised_trainer) == 1
        assert supervised_trainer[0].model is not unsupervised_winner[0].model
        assert supervised_trainer[0].model.latent_dimension == winner.latent_dimension
        assert supervised_trainer[0].model.topology_name == winner.topology_name

        def load(path: Path) -> dict[str, Any]:
            return yaml.safe_load(path.read_text(encoding="utf-8"))

        unsupervised_config = load(
            completed.directory / winner.run_directory / "config_resolved.yaml"
        )
        supervised_config = load(
            completed.directory / "supervised_extension" / "training" / "config_resolved.yaml"
        )
        assert supervised_config["supervised_component"]["enabled"] is True
        assert supervised_config["supervised_component"]["classification_loss_weight"] == 1.0
        assert supervised_config["supervised_component"]["use_latent_mean"] is True
        unsupervised_config.pop("supervised_component")
        supervised_config.pop("supervised_component")
        assert supervised_config == unsupervised_config

    def test_supervised_training_and_test_results(self, completed: CompletedProtocol) -> None:
        base = completed.directory / "supervised_extension"
        metrics = json.loads((base / "training" / "metrics.json").read_text("utf-8"))
        results = json.loads(
            (base / "final_evaluation" / "final_evaluation_results.json").read_text("utf-8")
        )
        test = results["test_metrics"]

        assert metrics["experiment_type"] == "supervised"
        assert metrics["validation_classification_loss"] is not None
        assert results["split"] == "test"
        assert results["experiment_type"] == "supervised"
        assert test["test_classification_loss"] is not None
        assert 0.0 <= test["test_accuracy"] <= 1.0
        for key in ("test_precision_macro", "test_recall_macro", "test_f1_macro"):
            assert test[key] is not None
        confusion = pd.read_csv(base / "final_evaluation" / "tables" / "test_confusion_matrix.csv")
        assert confusion.shape[0] == 10

    def test_comparison_files(self, completed: CompletedProtocol) -> None:
        base = completed.directory / "supervised_extension"
        table = pd.read_csv(base / "comparison_with_unsupervised.csv")
        summary = (base / "comparison_summary.md").read_text("utf-8")

        assert list(table["experiment_type"]) == ["unsupervised", "supervised"]
        assert table["topology_name"].nunique() == 1
        assert table["latent_dimension"].nunique() == 1
        assert pd.isna(table.loc[0, "validation_classification_loss"])
        assert pd.isna(table.loc[0, "test_accuracy"])
        assert table.loc[1, "test_accuracy"] >= 0
        assert "not compared numerically" in summary
        assert "to be completed" in summary
        assert "Manual interpretation" in summary
        assert (base / "artifact_manifest.json").is_file()


# ------------------------------------------------------------- reports and state


class TestReportsManifestsAndState:
    def test_local_manifest_registers_every_file(self, completed: CompletedProtocol) -> None:
        run_dir = _grid_dir(completed, "small", 8)
        manifest = json.loads((run_dir / "artifact_manifest.json").read_text("utf-8"))
        registered = {entry["path"] for entry in manifest["artifacts"]}
        on_disk = {
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        }

        assert registered == on_disk
        assert manifest["execution_status"] == "completed"
        assert manifest["topology_name"] == "small"
        assert manifest["latent_dimension"] == 8
        assert all(entry["exists"] and entry["sha256"] for entry in manifest["artifacts"])

    def test_local_summary_lists_every_evidence_category(
        self, completed: CompletedProtocol
    ) -> None:
        text = (_grid_dir(completed, "deep", 8) / "run_summary.md").read_text("utf-8")

        for expected in (
            "training_total_loss.png",
            "reconstruction_comparison.png",
            "generated_samples.png",
            "latent_space_pca_8d.png",
            "latent_space_umap_8d.png",
            "latent_space_tsne_8d.png",
            "best_model.pt",
            "Best epoch",
            "Validation total loss",
            "Total duration",
            "## Warnings",
            "## Errors",
            "COMPLETED",
            "SYNTHETIC DATA",
        ):
            assert expected in text, expected

    def test_protocol_summary_and_evidence_index(self, completed: CompletedProtocol) -> None:
        summary = (completed.directory / "protocol_summary.md").read_text("utf-8")
        index = (completed.directory / "scientific_evidence_index.md").read_text("utf-8")

        for expected in (
            "Selected combination",
            "validation results only",
            "Limitations",
            "COMPLETED",
        ):
            assert expected in summary, expected
        for expected in (
            "Training curves",
            "Reconstructions",
            "Generated samples",
            "Direct latent space",
            "PCA explained variance",
            "UMAP",
            "t-SNE",
            "Comparison between modes",
            "Does supervision improve the separation?",
            "to be completed",
        ):
            assert expected in index, expected
        assert "SYNTHETIC DATA" in summary
        assert "SYNTHETIC DATA" in index

    def test_protocol_manifest_references_local_manifests(
        self, completed: CompletedProtocol
    ) -> None:
        manifest = json.loads((completed.directory / "protocol_manifest.json").read_text("utf-8"))

        assert manifest["dataset"] == "mnist"
        assert manifest["data_origin"] == "synthetic"
        assert manifest["status"] == "completed"
        assert len(manifest["combinations"]) == 4
        for entry in manifest["combinations"]:
            assert entry["manifest"]["exists"] is True
            assert entry["checkpoint"]["sha256"]
            assert "artifacts" not in entry
        assert manifest["selection"]["selected_configuration"]["exists"] is True
        assert manifest["final_evaluation"]["manifest"]["exists"] is True
        assert manifest["supervised_extension"]["comparison"]["exists"] is True
        assert manifest["reports"]["protocol_summary"]["exists"] is True
        assert manifest["environment"]["exists"] is True

    def test_state_file_describes_the_finished_run(self, completed: CompletedProtocol) -> None:
        state = load_state(completed.directory)

        assert state.status == "completed"
        assert state.phase == "done"
        assert len(state.completed_combinations) == 4
        assert state.failed_combinations == []
        assert state.pending_combinations == []
        assert state.final_evaluation_status == "completed"
        assert state.supervised_training_status == "completed"
        assert state.supervised_evaluation_status == "completed"
        assert state.selected_configuration is not None
        assert state.config_hash
        assert state.data_origin == "synthetic"

    def test_run_directory_is_new_and_marked_synthetic(self, completed: CompletedProtocol) -> None:
        assert completed.directory.name.startswith("run_")
        assert completed.directory.name.endswith("_reduced")
        assert any(part.startswith("synthetic_smoke_test") for part in completed.directory.parts)
        assert (completed.directory / "protocol_config_resolved.yaml").is_file()


# ---------------------------------------------------------------- failure handling


class TestFailureHandling:
    def test_failed_combination_is_excluded_and_the_run_continues(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _install_recorder(monkeypatch)
        real_trainer = single_experiment.VAETrainer

        class FlakyTrainer(real_trainer):  # type: ignore[misc, valid-type]
            def fit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
                if self.model.latent_dimension == 2:
                    raise RuntimeError("simulated training failure")
                return super().fit(*args, **kwargs)

        monkeypatch.setattr(single_experiment, "VAETrainer", FlakyTrainer)

        result = _run(base_dir)

        directory = result.protocol_directory
        table = pd.read_csv(directory / "selection" / "factorial_comparison.csv")
        status = dict(zip(table["latent_dimension"], table["execution_status"], strict=True))
        assert status == {2: "failed", 8: "completed"}
        assert result.selection.winner.key == "small/latent_dim_8"  # type: ignore[union-attr]
        assert result.status == "completed_with_warnings"
        state = load_state(directory)
        assert state.failed_combinations == ["small/latent_dim_2"]
        assert state.completed_combinations == ["small/latent_dim_8"]
        assert "simulated training failure" in "".join(state.errors)
        failed_dir = directory / "grid" / "small" / "latent_dim_2"
        assert (failed_dir / "artifact_manifest.json").is_file()
        assert (failed_dir / "logs" / "experiment.log").is_file()
        assert "simulated training failure" in (failed_dir / "run_summary.md").read_text("utf-8")
        assert [m["latent_dimension"] for m in recorder.test_measurements] == [8]

    def test_no_valid_combination_stops_before_test_and_supervised(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _install_recorder(monkeypatch)

        class BrokenTrainer(single_experiment.VAETrainer):  # type: ignore[misc, name-defined]
            def fit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
                raise RuntimeError("always fails")

        monkeypatch.setattr(single_experiment, "VAETrainer", BrokenTrainer)

        result = _run(base_dir, config=make_protocol_config(["small"], [2, 8]))

        directory = result.protocol_directory
        assert result.status == "failed"
        assert result.selection is not None
        assert result.selection.winner is None
        assert recorder.test_measurements == []
        assert recorder.evaluations == []
        assert not (directory / "final_evaluation").exists()
        assert not (directory / "supervised_extension").exists()
        assert not (directory / "selection" / "selected_configuration.yaml").exists()
        assert "No valid combination" in (
            directory / "selection" / "selection_summary.md"
        ).read_text("utf-8")
        assert (directory / "protocol_summary.md").is_file()
        assert (directory / "protocol_manifest.json").is_file()
        assert load_state(directory).status == "failed"

    def test_projection_failure_keeps_the_combination_eligible(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing_tsne(self: object, vectors: np.ndarray) -> None:
            raise RuntimeError("simulated t-SNE failure")

        monkeypatch.setattr(latent_evidence.TSNEProjection, "fit_transform", failing_tsne)

        result = _run(base_dir)

        directory = result.protocol_directory
        run_dir = directory / "grid" / "small" / "latent_dim_8"
        record = next(r for r in result.records if r.latent_dimension == 8)
        assert record.status == "partial"
        assert (run_dir / "checkpoints" / "best_model.pt").is_file()
        assert list((run_dir / "latent_data").glob("*.csv"))
        assert _files(run_dir / "figures" / "latent_space", "latent_space_pca_8d.png")
        assert not _files(run_dir / "figures" / "latent_space", "latent_space_tsne_8d.png")
        assert "simulated t-SNE failure" in (run_dir / "run_summary.md").read_text("utf-8")
        assert result.status == "completed_with_warnings"
        assert any("small/latent_dim_8" in warning for warning in load_state(directory).warnings)
        assert {r.key for r in result.selection.candidates} == {  # type: ignore[union-attr]
            "small/latent_dim_2",
            "small/latent_dim_8",
        }

    def test_partial_runs_are_excluded_when_the_policy_forbids_them(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing_tsne(self: object, vectors: np.ndarray) -> None:
            raise RuntimeError("simulated t-SNE failure")

        monkeypatch.setattr(latent_evidence.TSNEProjection, "fit_transform", failing_tsne)
        config = make_protocol_config(["small"], [2, 8], supervised_extension=False)
        config.protocol.continue_on_noncritical_evidence_error = False

        result = _run(base_dir, config=config)

        assert [r.key for r in result.selection.candidates] == ["small/latent_dim_2"]  # type: ignore[union-attr]
        assert result.selection.winner.key == "small/latent_dim_2"  # type: ignore[union-attr]

    def test_failed_final_evaluation_keeps_the_selection_and_can_be_resumed(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _install_recorder(monkeypatch)
        config = make_protocol_config(["small"], [2, 8], supervised_extension=True)
        with monkeypatch.context() as patch:
            patch.setattr(
                heldout_evaluation,
                "measure_test_losses",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError("simulated test failure")
                ),
            )
            first = _run(base_dir, config=copy.deepcopy(config))

        directory = first.protocol_directory
        selected_text = (directory / "selection" / "selected_configuration.yaml").read_text("utf-8")
        assert first.status == "failed"
        assert load_state(directory).final_evaluation_status == "failed"
        assert len(recorder.evaluations) == 1
        assert not (directory / "supervised_extension").exists()
        trainers_before_resume = len(recorder.trainers)

        resumed = run_factorial_protocol(
            copy.deepcopy(config), resume_dir=directory, splits=make_synthetic_splits()
        )

        assert resumed.status == "completed_with_warnings"
        assert (directory / "selection" / "selected_configuration.yaml").read_text(
            "utf-8"
        ) == selected_text
        state = load_state(directory)
        assert state.final_evaluation_status == "completed"
        assert state.supervised_evaluation_status == "completed"
        assert len(state.resumed_at) == 1
        assert len(recorder.trainers) == trainers_before_resume + 1  # only the supervised model
        assert len(recorder.evaluations) == 3
        assert list(directory.glob("final_evaluation_incomplete_*"))
        winner_dims = {m["latent_dimension"] for m in recorder.test_measurements}
        assert len(winner_dims) == 1


# --------------------------------------------------------------- resume and safety


class TestResumeAndOverwrite:
    def test_interrupted_run_resumes_only_the_missing_combination(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _install_recorder(monkeypatch)
        real_trainer = single_experiment.VAETrainer
        config = make_protocol_config(["small"], [2, 8], supervised_extension=False)

        class InterruptingTrainer(real_trainer):  # type: ignore[misc, valid-type]
            def fit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
                if self.model.latent_dimension == 8:
                    raise KeyboardInterrupt
                return super().fit(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(single_experiment, "VAETrainer", InterruptingTrainer)
            with pytest.raises(KeyboardInterrupt):
                _run(base_dir, config=copy.deepcopy(config))
        directory = next(base_dir.glob("run_*"))
        state = load_state(directory)
        assert state.status == "interrupted"
        assert state.completed_combinations == ["small/latent_dim_2"]
        finished_checkpoint = directory / "grid/small/latent_dim_2/checkpoints/best_model.pt"
        finished_bytes = finished_checkpoint.read_bytes()
        trainers_before = len(recorder.trainers)

        resumed = run_factorial_protocol(
            copy.deepcopy(config), resume_dir=directory, splits=make_synthetic_splits()
        )

        assert resumed.status == "completed_with_warnings" or resumed.status == "completed"
        assert len(recorder.trainers) == trainers_before + 1
        assert recorder.trainers[-1].model.latent_dimension == 8
        assert finished_checkpoint.read_bytes() == finished_bytes
        assert list((directory / "grid/small").glob("latent_dim_8_incomplete_*"))
        final_state = load_state(directory)
        assert final_state.completed_combinations == ["small/latent_dim_2", "small/latent_dim_8"]
        assert len(final_state.resumed_at) == 1
        assert len(next(iter(base_dir.glob("run_*"))).name) > 0
        assert len(list(base_dir.glob("run_*"))) == 1

    def test_directory_without_valid_artifacts_is_not_considered_complete(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _install_recorder(monkeypatch)
        config = make_protocol_config(["small"], [2], supervised_extension=False)
        first = _run(base_dir, config=copy.deepcopy(config))
        directory = first.protocol_directory
        (directory / "grid/small/latent_dim_2/artifact_manifest.json").unlink()
        state = load_state(directory)
        state.status = "interrupted"
        from vae_project.experiments.protocol_state import save_state

        save_state(state, directory)
        trainers_before = len(recorder.trainers)

        run_factorial_protocol(config, resume_dir=directory, splits=make_synthetic_splits())

        assert len(recorder.trainers) == trainers_before + 1
        assert (directory / "grid/small/latent_dim_2/artifact_manifest.json").is_file()

    def test_resume_rejects_an_incompatible_configuration(self, base_dir: Path) -> None:
        config = make_protocol_config(["small"], [2], supervised_extension=False)
        first = _run(base_dir, config=copy.deepcopy(config))
        state = load_state(first.protocol_directory)
        from vae_project.experiments.protocol_state import save_state

        state.status = "interrupted"
        save_state(state, first.protocol_directory)
        changed = copy.deepcopy(config)
        changed.optimizer.learning_rate = 0.5

        with pytest.raises(ProtocolStateError, match="configuration differs"):
            run_factorial_protocol(
                changed, resume_dir=first.protocol_directory, splits=make_synthetic_splits()
            )

    def test_resume_rejects_a_finished_run_and_a_missing_state(self, base_dir: Path) -> None:
        config = make_protocol_config(["small"], [2], supervised_extension=False)
        first = _run(base_dir, config=copy.deepcopy(config))

        with pytest.raises(ProtocolStateError, match="already finished"):
            run_factorial_protocol(
                config, resume_dir=first.protocol_directory, splits=make_synthetic_splits()
            )
        with pytest.raises(ProtocolStateError, match=r"protocol_state\.json"):
            run_factorial_protocol(
                config, resume_dir=base_dir / "does_not_exist", splits=make_synthetic_splits()
            )

    def test_existing_run_directory_is_never_overwritten(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(paths, "build_run_timestamp", lambda: "20260101_000000")
        existing = base_dir / "run_20260101_000000_fixed"
        existing.mkdir(parents=True)
        (existing / "keep.txt").write_text("precious", encoding="utf-8")

        with pytest.raises(ArtifactError, match="already exists"):
            _run(base_dir, run_name="fixed")

        assert (existing / "keep.txt").read_text("utf-8") == "precious"
        assert [path.name for path in existing.iterdir()] == ["keep.txt"]

    def test_each_execution_creates_a_new_run_directory(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        names = {paths.build_comparison_run_dir(base_dir).name for _ in range(5)}

        assert len(names) == 5
        assert all(name.startswith("run_") for name in names)

    def test_unmarked_directory_is_rejected_for_fake_data(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactError, match="smoke_test"):
            _run(tmp_path / "artifacts" / "mnist")
        assert not (tmp_path / "artifacts").exists() or not list(
            (tmp_path / "artifacts").rglob("run_*/*")
        )

    def test_disabled_protocol_is_rejected(self, base_dir: Path) -> None:
        from vae_project.experiments.protocol_models import ProtocolError

        config = make_protocol_config()
        config.factorial_comparison.enabled = False

        with pytest.raises(ProtocolError, match="enabled is false"):
            _run(base_dir, config=config)


class TestManualSupervisedMode:
    def test_manual_mode_writes_the_configuration_and_does_not_train(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _install_recorder(monkeypatch)
        config = make_protocol_config(["small"], [2, 8], True, execution_mode="manual")

        result = _run(base_dir, config=config)

        directory = result.protocol_directory
        assert result.status == "awaiting_manual_supervised_execution"
        assert result.manual_command is not None
        assert "run-pipeline" in result.manual_command
        assert "supervised_selected_configuration.yaml" in result.manual_command
        generated = yaml.safe_load(
            (directory / "generated_configs" / "supervised_selected_configuration.yaml").read_text(
                "utf-8"
            )
        )
        assert generated["supervised_component"]["enabled"] is True
        assert not (directory / "supervised_extension").exists()
        assert [t.supervised_head for t in recorder.trainers if t.supervised_head is not None] == []
        state = load_state(directory)
        assert state.supervised_training_status == "awaiting_manual_supervised_execution"
        assert len(recorder.test_measurements) == 1


# ------------------------------------------------------------------ CLI and real PCA/UMAP/t-SNE


class TestCommandLine:
    def test_command_is_registered_and_accepts_resume(self) -> None:
        parser = cli._build_parser()

        args = parser.parse_args(
            [
                "run-factorial-protocol",
                "--config",
                "c.yaml",
                "--resume",
                "some/dir",
                "--run-name",
                "x",
            ]
        )

        assert args.command == "run-factorial-protocol"
        assert args.resume == Path("some/dir")
        assert args.run_name == "x"

    def test_existing_commands_are_preserved(self) -> None:
        parser = cli._build_parser()

        for command in ("train", "run-pipeline", "select-topology", "compare-latent-dims"):
            assert parser.parse_args([command, "--config", "c.yaml"]).command == command
        assert (
            parser.parse_args(["validate-config", "--config", "c.yaml"]).command
            == "validate-config"
        )
        assert parser.parse_args(["smoke-test"]).command == "smoke-test"

    def test_command_reports_a_missing_configuration(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = cli._build_parser().parse_args(
            ["run-factorial-protocol", "--config", str(tmp_path / "missing.yaml")]
        )

        assert cli._cmd_run_factorial_protocol(args) == 1
        assert "configuration" in capsys.readouterr().err

    def test_command_runs_the_protocol_service(
        self,
        tmp_path: Path,
        base_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        recorded: dict[str, Any] = {}

        def fake_run(config: Any, **kwargs: Any) -> ProtocolResult:
            recorded.update(kwargs)
            state = load_state_stub()
            return ProtocolResult(protocol_directory=base_dir, status="completed", state=state)

        def load_state_stub() -> Any:
            from vae_project.experiments.protocol_state import ProtocolState

            return ProtocolState(run_id="r", config_hash="h", dataset="mnist")

        monkeypatch.setattr(factorial_protocol, "run_factorial_protocol", fake_run)
        project_root = Path(__file__).resolve().parents[2]
        args = cli._build_parser().parse_args(
            [
                "run-factorial-protocol",
                "--config",
                str(project_root / "configs" / "factorial_protocol_mnist.yaml"),
                "--output-dir",
                str(tmp_path),
            ]
        )

        assert cli._cmd_run_factorial_protocol(args) == 0
        assert recorded["base_output_dir"] == tmp_path
        assert recorded["resume_dir"] is None
        assert "Protocol status: completed" in capsys.readouterr().out


@pytest.mark.slow
def test_real_pca_umap_tsne_smoke(base_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real PCA, UMAP and t-SNE on a tiny synthetic sample (no test doubles)."""
    from vae_project.projections.umap_projection import UMAPProjection

    monkeypatch.setattr(latent_evidence, "UMAPProjection", UMAPProjection)
    config = make_protocol_config(["small"], [8], supervised_extension=False)
    config.projections.umap.n_neighbors = 5
    config.projections.sampling.max_samples = 40

    result = _run(base_dir, config=config)

    run_dir = result.protocol_directory / "grid" / "small" / "latent_dim_8"
    assert result.status == "completed"
    for prefix in LATENT_PLOT_PREFIXES:
        assert (run_dir / "figures" / "latent_space" / f"{prefix}8d.png").is_file()
    metrics = pd.read_csv(run_dir / "tables" / "projection_metrics.csv")
    assert {"pca_2d", "umap_2d", "tsne_2d"} <= set(metrics["space_type"])
