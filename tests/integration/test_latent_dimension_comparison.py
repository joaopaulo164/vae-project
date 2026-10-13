"""Integration tests for the latent dimension comparison command.

Everything runs on CPU with small synthetic data: no downloads and no real
MNIST training. UMAP is replaced by a test double (numba compilation is slow);
PCA and t-SNE run for real on tiny inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from vae_project import cli
from vae_project.config.models import (
    DatasetName,
    DevicePolicy,
    EarlyStoppingConfig,
    ExperimentConfig,
    ModelConfig,
    ProjectionsConfig,
    RuntimeConfig,
    SamplingConfig,
    TopologyConfig,
    TrainingConfig,
    TSNEConfig,
)
from vae_project.data.splits import DataSplits
from vae_project.experiments import latent_dimension_comparison as comparison
from vae_project.experiments import latent_evidence
from vae_project.experiments.latent_dimension_comparison import (
    LatentDimensionComparisonError,
    run_latent_dimension_comparison,
)
from vae_project.projections.umap_projection import UMAPResult
from vae_project.reporting.comparison_report import (
    COMPARISON_CSV_COLUMNS,
    DimensionRunRecord,
    write_dimension_run_summary,
)
from vae_project.utils import paths
from vae_project.utils.paths import (
    ArtifactError,
    build_comparison_run_dir,
    resolve_comparison_run_dir,
)

pytestmark = pytest.mark.integration

DIMENSIONS = [2, 8, 16]
N_CLASSES = 10


class FakeUMAPProjection:
    """Test double for UMAP (avoids numba compilation in the test suite)."""

    def __init__(self, config: object) -> None:
        self.config = config

    def fit_transform(self, latent_vectors: np.ndarray) -> UMAPResult:
        rng = np.random.default_rng(0)
        n_samples, latent_dim = latent_vectors.shape
        return UMAPResult(
            projection_2d=rng.standard_normal((n_samples, 2)),
            n_samples=n_samples,
            latent_dimension=latent_dim,
            n_neighbors=5,
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )


def _make_config(dimensions: list[int]) -> ExperimentConfig:
    config = ExperimentConfig()
    config.dataset.name = DatasetName.MNIST
    config.dataset.download = False
    config.model = ModelConfig(
        latent_dimension=4,
        topology=TopologyConfig(name="medium", hidden_dimensions=[32, 16]),
    )
    config.training = TrainingConfig(max_epochs=2, batch_size=32)
    config.early_stopping = EarlyStoppingConfig(enabled=True, patience=2)
    config.runtime = RuntimeConfig(device=DevicePolicy.CPU)
    config.reproducibility.global_seed = 42
    config.reproducibility.deterministic_mode = True
    config.latent_dimension_comparison.dimensions = list(dimensions)
    config.latent_dimension_comparison.fixed_topology = "medium"
    config.projections = ProjectionsConfig(
        methods=["pca", "umap", "tsne"],
        tsne=TSNEConfig(perplexity=5.0, max_iter=250),
        sampling=SamplingConfig(enabled=True, max_samples=60, stratified=True, seed=42),
    )
    config.metrics.trustworthiness_n_neighbors = 3
    config.plotting.save_svg = True
    config.plotting.save_png = True
    return config


def _make_splits() -> DataSplits:
    generator = torch.Generator().manual_seed(7)

    def dataset(n: int) -> TensorDataset:
        images = torch.rand(n, 1, 28, 28, generator=generator)
        labels = torch.arange(n) % N_CLASSES
        return TensorDataset(images, labels)

    return DataSplits(
        train_dataset=dataset(120),
        val_dataset=dataset(100),
        test_dataset=dataset(50),
        train_indices=list(range(120)),
        val_indices=list(range(100)),
        class_names=[str(i) for i in range(N_CLASSES)],
        num_classes=N_CLASSES,
        dataset_name=DatasetName.MNIST,
        is_quick_mode=False,
    )


@pytest.fixture(autouse=True)
def _fake_umap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latent_evidence, "UMAPProjection", FakeUMAPProjection)


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """Run directory for synthetic data; the name carries the required marker."""
    return tmp_path / "synthetic_smoke_test"


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pd.DataFrame]:
    """Run the comparison once and share the artifacts across read-only tests."""
    mp = pytest.MonkeyPatch()
    mp.setattr(latent_evidence, "UMAPProjection", FakeUMAPProjection)
    try:
        output_dir = tmp_path_factory.mktemp("synthetic_comparison")
        table = run_latent_dimension_comparison(
            _make_config(DIMENSIONS), output_dir, splits=_make_splits()
        )
    finally:
        mp.undo()
    return output_dir, table


class TestPerDimensionArtifacts:
    """Every dimension gets its own directory with the full evidence set."""

    def test_one_directory_per_dimension(self, completed_run: tuple[Path, pd.DataFrame]) -> None:
        output_dir, _ = completed_run
        for dim in DIMENSIONS:
            assert (output_dir / f"latent_dim_{dim}").is_dir()

    def test_dimension_list_gives_one_row_each(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        _, table = completed_run
        assert list(table["latent_dimension"]) == DIMENSIONS
        assert list(table["execution_status"]) == ["completed"] * len(DIMENSIONS)

    def test_best_checkpoint_has_matching_dimension(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        output_dir, _ = completed_run
        for dim in DIMENSIONS:
            ckpt_dir = output_dir / f"latent_dim_{dim}" / "checkpoints"
            assert (ckpt_dir / "best_model.pt").exists()
            metadata = json.loads((ckpt_dir / "checkpoint_metadata.json").read_text("utf-8"))
            assert metadata["latent_dimension"] == dim
            assert metadata["seed"] == 42
            assert metadata["experiment_type"] == "unsupervised"

    def test_models_are_independent_between_dimensions(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        output_dir, table = completed_run
        assert table["parameter_count"].nunique() == len(DIMENSIONS)
        state_2 = torch.load(
            output_dir / "latent_dim_2" / "checkpoints" / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        state_8 = torch.load(
            output_dir / "latent_dim_8" / "checkpoints" / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert str(state_2) != str(state_8)

    def test_direct_2d_has_no_projection_substitute(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        output_dir, _ = completed_run
        dim_dir = output_dir / "latent_dim_2"
        figures = sorted(p.name for p in (dim_dir / "figures" / "latent_space").iterdir())
        assert any(name.endswith(".png") for name in figures)
        assert any(name.endswith(".svg") for name in figures)
        for forbidden in ("pca", "umap", "tsne"):
            assert not any(forbidden in name for name in figures)
        assert not (dim_dir / "tables" / "pca_explained_variance.csv").exists()
        assert not (dim_dir / "latent_data" / "projection_sample_indices.csv").exists()

    @pytest.mark.parametrize("dim", [8, 16])
    def test_projections_for_higher_dimensions(
        self, completed_run: tuple[Path, pd.DataFrame], dim: int
    ) -> None:
        output_dir, _ = completed_run
        figures_dir = output_dir / f"latent_dim_{dim}" / "figures" / "latent_space"
        names = [p.name for p in figures_dir.iterdir()]
        for method in ("pca", "umap", "tsne"):
            assert any(method in n and n.endswith(".png") for n in names), method
            assert any(method in n and n.endswith(".svg") for n in names), method
        assert any("explained_variance" in n for n in names)

    def test_dimension_is_in_projection_file_names(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        output_dir, _ = completed_run
        for dim in (8, 16):
            figures_dir = output_dir / f"latent_dim_{dim}" / "figures" / "latent_space"
            projection_files = [p.name for p in figures_dir.iterdir() if "explained" not in p.name]
            assert projection_files
            assert all(f"{dim}" in name for name in projection_files)

    def test_pca_explained_variance_table(self, completed_run: tuple[Path, pd.DataFrame]) -> None:
        output_dir, _ = completed_run
        df = pd.read_csv(output_dir / "latent_dim_8" / "tables" / "pca_explained_variance.csv")
        for column in (
            "latent_dimension",
            "component",
            "explained_variance_ratio",
            "cumulative_explained_variance_ratio",
        ):
            assert column in df.columns
        assert (df["latent_dimension"] == 8).all()
        assert df["cumulative_explained_variance_ratio"].is_monotonic_increasing

    def test_projection_metrics_distinguish_spaces_and_scope(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        output_dir, _ = completed_run
        df = pd.read_csv(output_dir / "latent_dim_8" / "tables" / "projection_metrics.csv")
        assert {"latent_original", "pca_2d", "umap_2d", "tsne_2d"} <= set(df["space_type"])
        assert set(df["scope"]) == {"full_split", "sample"}
        assert (df["seed"] == 42).all()
        assert (df["experiment_type"] == "unsupervised").all()
        assert (df["latent_dimension"] == 8).all()

    def test_sample_indices_csv_is_stratified_and_shared(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        output_dir, _ = completed_run
        indices = {
            dim: pd.read_csv(
                output_dir / f"latent_dim_{dim}" / "latent_data" / "projection_sample_indices.csv"
            )
            for dim in (8, 16)
        }
        assert len(indices[8]) == 60
        assert indices[8]["label"].nunique() == N_CLASSES
        assert list(indices[8]["sample_index"]) == list(indices[16]["sample_index"])

    def test_latent_representations_saved(self, completed_run: tuple[Path, pd.DataFrame]) -> None:
        output_dir, _ = completed_run
        latent_dir = output_dir / "latent_dim_8" / "latent_data"
        csv_files = [p for p in latent_dir.glob("*.csv") if "indices" not in p.name]
        assert csv_files
        df = pd.read_csv(csv_files[0])
        assert "latent_mean_7" in df.columns
        assert "latent_mean_8" not in df.columns

    def test_manifest_summary_and_config_per_dimension(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        output_dir, _ = completed_run
        for dim in DIMENSIONS:
            dim_dir = output_dir / f"latent_dim_{dim}"
            manifest = json.loads((dim_dir / "artifact_manifest.json").read_text("utf-8"))
            assert manifest["execution_status"] == "completed"
            assert manifest["latent_dimension"] == dim
            assert manifest["experiment_type"] == "unsupervised"
            assert (dim_dir / "run_summary.md").read_text("utf-8").strip()
            assert (dim_dir / "config_resolved.yaml").exists()
            assert (dim_dir / "metrics.json").exists()
            assert (dim_dir / "tables" / "training_history.csv").exists()

    def test_consolidated_csv_and_summary(self, completed_run: tuple[Path, pd.DataFrame]) -> None:
        output_dir, table = completed_run
        saved = pd.read_csv(output_dir / "latent_dimension_comparison.csv")
        assert list(saved.columns) == list(COMPARISON_CSV_COLUMNS)
        assert list(table.columns) == list(COMPARISON_CSV_COLUMNS)
        summary = (output_dir / "latent_dimension_comparison_summary.md").read_text("utf-8")
        assert summary.strip()

    def test_not_applicable_values_are_missing_not_zero(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        _, table = completed_run
        row_2 = table[table["latent_dimension"] == 2].iloc[0]
        assert pd.isna(row_2["pca_plot_path"])
        assert pd.isna(row_2["pca_cumulative_explained_variance"])
        assert not pd.isna(row_2["direct_plot_path"])
        row_8 = table[table["latent_dimension"] == 8].iloc[0]
        assert pd.isna(row_8["direct_plot_path"])

    def test_same_seed_topology_and_unsupervised_losses(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        _, table = completed_run
        assert (table["seed"] == 42).all()
        assert (table["topology_name"] == "medium").all()
        assert (table["experiment_type"] == "unsupervised").all()
        assert (table["validation_classification_loss"].fillna(0.0) == 0.0).all()

    def test_runs_on_cpu_without_download(self, completed_run: tuple[Path, pd.DataFrame]) -> None:
        output_dir, _ = completed_run
        metrics = json.loads((output_dir / "latent_dim_2" / "metrics.json").read_text("utf-8"))
        assert metrics["execution_status"] == "completed"
        summary = (output_dir / "latent_dim_2" / "run_summary.md").read_text("utf-8")
        assert "cpu" in summary.lower()


class TestFailureHandling:
    """Failures are recorded, never hidden, and do not stop other dimensions."""

    def test_projection_failure_marks_partial_and_keeps_artifacts(
        self, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing_tsne(self: object, vectors: np.ndarray) -> None:
            raise RuntimeError("simulated t-SNE failure")

        monkeypatch.setattr(latent_evidence.TSNEProjection, "fit_transform", failing_tsne)
        table = run_latent_dimension_comparison(_make_config([8]), out_dir, splits=_make_splits())
        assert table.iloc[0]["execution_status"] == "partial"
        assert "simulated t-SNE failure" in str(table.iloc[0]["error_summary"])
        dim_dir = out_dir / "latent_dim_8"
        assert (dim_dir / "checkpoints" / "best_model.pt").exists()
        assert list((dim_dir / "latent_data").glob("*.csv"))
        manifest = json.loads((dim_dir / "artifact_manifest.json").read_text("utf-8"))
        assert manifest["execution_status"] == "partial"
        assert manifest["errors"]

    def test_training_failure_continues_with_next_dimension(
        self, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_trainer = comparison.VAETrainer

        class FlakyTrainer(real_trainer):  # type: ignore[misc, valid-type]
            def fit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
                if self.model.latent_dimension == 8:
                    raise RuntimeError("simulated training failure")
                return super().fit(*args, **kwargs)

        monkeypatch.setattr(comparison, "VAETrainer", FlakyTrainer)
        table = run_latent_dimension_comparison(
            _make_config([2, 8, 16]), out_dir, splits=_make_splits()
        )
        status = dict(zip(table["latent_dimension"], table["execution_status"], strict=True))
        assert status == {2: "completed", 8: "failed", 16: "completed"}
        summary = (out_dir / "latent_dimension_comparison_summary.md").read_text("utf-8")
        assert "simulated training failure" in summary
        assert (out_dir / "latent_dim_8" / "artifact_manifest.json").exists()

    def test_all_dimensions_failing_raises(
        self, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class BrokenTrainer(comparison.VAETrainer):  # type: ignore[misc, name-defined]
            def fit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
                raise RuntimeError("always fails")

        monkeypatch.setattr(comparison, "VAETrainer", BrokenTrainer)
        with pytest.raises(LatentDimensionComparisonError):
            run_latent_dimension_comparison(_make_config([2, 8]), out_dir, splits=_make_splits())


class TestComparisonGuards:
    """Invalid comparison inputs are rejected before any training."""

    @pytest.mark.parametrize("dimensions", [[], [1, 8], [8, 8], [2, 8.5]])
    def test_invalid_dimension_lists_rejected(self, out_dir: Path, dimensions: list) -> None:
        with pytest.raises(LatentDimensionComparisonError):
            run_latent_dimension_comparison(
                _make_config([2]), out_dir, dimensions=dimensions, splits=_make_splits()
            )

    def test_topology_mismatch_rejected(self, out_dir: Path) -> None:
        config = _make_config([2])
        config.latent_dimension_comparison.fixed_topology = "deep"
        with pytest.raises(LatentDimensionComparisonError):
            run_latent_dimension_comparison(config, out_dir, splits=_make_splits())

    def test_disabled_comparison_rejected(self, out_dir: Path) -> None:
        config = _make_config([2])
        config.latent_dimension_comparison.enabled = False
        with pytest.raises(LatentDimensionComparisonError):
            run_latent_dimension_comparison(config, out_dir, splits=_make_splits())

    def test_model_latent_dimension_is_not_mutated(self, out_dir: Path) -> None:
        config = _make_config([2])
        original = copy.deepcopy(config.model.latent_dimension)
        run_latent_dimension_comparison(config, out_dir, splits=_make_splits())
        assert config.model.latent_dimension == original
        assert isinstance(config.model.latent_dimension, int)

    def test_rerun_into_new_directory_does_not_touch_previous(self, out_dir: Path) -> None:
        first = out_dir / "run_a"
        second = out_dir / "run_b"
        run_latent_dimension_comparison(_make_config([2]), first, splits=_make_splits())
        marker = first / "latent_dim_2" / "metrics.json"
        before = marker.read_text("utf-8")
        run_latent_dimension_comparison(_make_config([2]), second, splits=_make_splits())
        assert marker.read_text("utf-8") == before


class TestBestCheckpointAndIndependence:
    """Evidence uses the reloaded best checkpoint, and every dimension is fresh."""

    def test_analysis_uses_best_checkpoint_not_last_epoch(
        self, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trained_models: list[torch.nn.Module] = []
        analyzed_models: list[torch.nn.Module] = []
        real_trainer = comparison.VAETrainer
        real_evidence = comparison.generate_latent_evidence

        class DegradingTrainer(real_trainer):  # type: ignore[misc, valid-type]
            """Corrupts the final weights so the last epoch is worse than the best."""

            def fit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
                history = super().fit(*args, **kwargs)
                with torch.no_grad():
                    for parameter in self.model.parameters():
                        parameter.add_(5.0)
                trained_models.append(self.model)
                return history

        def recording_evidence(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            analyzed_models.append(kwargs["model"])
            return real_evidence(*args, **kwargs)

        monkeypatch.setattr(comparison, "VAETrainer", DegradingTrainer)
        monkeypatch.setattr(comparison, "generate_latent_evidence", recording_evidence)
        run_latent_dimension_comparison(_make_config([4]), out_dir, splits=_make_splits())

        checkpoint = torch.load(
            out_dir / "latent_dim_4" / "checkpoints" / "best_model.pt",
            map_location="cpu",
            weights_only=True,
        )["model_state_dict"]
        analyzed = analyzed_models[0]
        assert analyzed is not trained_models[0]
        assert not analyzed.training
        for name, tensor in analyzed.state_dict().items():
            assert torch.equal(tensor, checkpoint[name]), name
        degraded = trained_models[0].state_dict()
        assert any(not torch.equal(degraded[name], checkpoint[name]) for name in checkpoint)

    def test_each_dimension_gets_fresh_training_components_and_seed(
        self, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainers: list[object] = []
        seeds: list[int] = []
        real_trainer = comparison.VAETrainer
        real_seed = comparison.set_global_seed

        class RecordingTrainer(real_trainer):  # type: ignore[misc, valid-type]
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                trainers.append(self)

        def recording_seed(seed: int, *args: object, **kwargs: object) -> None:
            seeds.append(seed)
            real_seed(seed, *args, **kwargs)

        monkeypatch.setattr(comparison, "VAETrainer", RecordingTrainer)
        monkeypatch.setattr(comparison, "set_global_seed", recording_seed)
        run_latent_dimension_comparison(_make_config([2, 8, 16]), out_dir, splits=_make_splits())

        assert seeds == [42, 42, 42]
        assert [trainer.model.latent_dimension for trainer in trainers] == [2, 8, 16]
        # The scheduler is disabled by default (None); it is checked only when present.
        assert all(trainer.scheduler is None for trainer in trainers)
        for attribute in ("model", "optimizer", "early_stopping"):
            identities = {id(getattr(trainer, attribute)) for trainer in trainers}
            assert len(identities) == len(trainers), attribute
        topologies = {tuple(trainer.model.hidden_dimensions) for trainer in trainers}
        assert len(topologies) == 1


# Figure categories produced by the comparison, with the dimensions that must have them.
FIGURE_CATEGORIES: dict[str, tuple[str, list[int]]] = {
    "direct_2d": ("latent_space_direct_2d", [2]),
    "pca": ("latent_space_pca_", [8, 16]),
    "umap": ("latent_space_umap_", [8, 16]),
    "tsne": ("latent_space_tsne_", [8, 16]),
    "pca_explained_variance": ("pca_explained_variance_", [8, 16]),
}
FIGURE_COLUMN = {
    "direct_2d": "direct_plot_path",
    "pca": "pca_plot_path",
    "umap": "umap_plot_path",
    "tsne": "tsne_plot_path",
    "pca_explained_variance": "pca_explained_variance_plot_path",
}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class TestFigureFiles:
    """Mechanical checks of the generated files. They say nothing about scientific quality."""

    @pytest.mark.parametrize("category", sorted(FIGURE_CATEGORIES))
    @pytest.mark.parametrize("extension", [".png", ".svg"])
    def test_figure_exists_is_not_empty_and_has_expected_extension(
        self, completed_run: tuple[Path, pd.DataFrame], category: str, extension: str
    ) -> None:
        output_dir, _ = completed_run
        prefix, dimensions = FIGURE_CATEGORIES[category]
        for dim in dimensions:
            figures_dir = output_dir / f"latent_dim_{dim}" / "figures" / "latent_space"
            matches = [
                p
                for p in figures_dir.iterdir()
                if p.name.startswith(prefix) and p.suffix == extension
            ]
            assert len(matches) == 1, (category, dim, extension)
            assert matches[0].stat().st_size > 0
            if extension == ".png":
                assert matches[0].read_bytes()[:8] == PNG_MAGIC
            else:
                assert b"<svg" in matches[0].read_bytes()[:2000]

    @pytest.mark.parametrize("category", sorted(FIGURE_CATEGORIES))
    def test_figure_belongs_to_its_dimension(
        self, completed_run: tuple[Path, pd.DataFrame], category: str
    ) -> None:
        """The file sits in the right latent_dim_N directory and, for N>2, names N."""
        output_dir, _ = completed_run
        prefix, dimensions = FIGURE_CATEGORIES[category]
        for dim in dimensions:
            figures_dir = output_dir / f"latent_dim_{dim}" / "figures" / "latent_space"
            for path in figures_dir.glob(f"{prefix}*"):
                if category != "direct_2d":
                    assert path.stem.endswith(f"_{dim}d"), path.name
        other_dims = [d for d in DIMENSIONS if d not in dimensions]
        for dim in other_dims:
            figures_dir = output_dir / f"latent_dim_{dim}" / "figures" / "latent_space"
            assert not list(figures_dir.glob(f"{prefix}*")), (category, dim)

    def test_projection_input_data_is_not_empty(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        output_dir, _ = completed_run
        for dim in DIMENSIONS:
            latent = pd.read_csv(
                output_dir / f"latent_dim_{dim}" / "latent_data" / "latent_representations.csv"
            )
            assert len(latent) > 0
            latent_columns = [c for c in latent.columns if c.startswith("latent_mean_")]
            assert len(latent_columns) == dim
            assert latent[latent_columns].notna().all().all()
        for dim in (8, 16):
            indices = pd.read_csv(
                output_dir / f"latent_dim_{dim}" / "latent_data" / "projection_sample_indices.csv"
            )
            assert len(indices) > 0

    def test_every_figure_is_registered_in_the_manifest(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        output_dir, _ = completed_run
        for dim in DIMENSIONS:
            dim_dir = output_dir / f"latent_dim_{dim}"
            manifest = json.loads((dim_dir / "artifact_manifest.json").read_text("utf-8"))
            entries = {entry["path"]: entry for entry in manifest["artifacts"]}
            figures = sorted((dim_dir / "figures").rglob("*.*"))
            assert figures
            for figure in figures:
                relative = figure.relative_to(dim_dir).as_posix()
                assert relative in entries, relative
                assert entries[relative]["size_bytes"] == figure.stat().st_size > 0
                assert entries[relative]["sha256"]

    def test_consolidated_table_points_to_existing_figures(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        output_dir, table = completed_run
        for category, column in FIGURE_COLUMN.items():
            _, dimensions = FIGURE_CATEGORIES[category]
            for _, row in table.iterrows():
                dim = int(row["latent_dimension"])
                if dim in dimensions:
                    path = output_dir / row[column]
                    assert path.is_file()
                    assert path.stat().st_size > 0
                    assert f"latent_dim_{dim}/" in str(row[column])
                else:
                    assert pd.isna(row[column])

    def test_figures_are_not_overwritten_by_a_second_run(self, out_dir: Path) -> None:
        first = out_dir / "first"
        second = out_dir / "second"
        run_latent_dimension_comparison(_make_config([2]), first, splits=_make_splits())
        figure = first / "latent_dim_2" / "figures" / "latent_space" / "latent_space_direct_2d.png"
        before = (figure.stat().st_mtime_ns, figure.read_bytes())
        run_latent_dimension_comparison(_make_config([2]), second, splits=_make_splits())
        assert (figure.stat().st_mtime_ns, figure.read_bytes()) == before
        assert (second / "latent_dim_2" / "figures" / "latent_space").is_dir()


class TestOverwritePrevention:
    """A finished run is never replaced silently."""

    def test_existing_run_directory_is_rejected_without_overwrite(self, out_dir: Path) -> None:
        run_dir = out_dir / "run_20260101_000000_abc123"
        run_latent_dimension_comparison(_make_config([2]), run_dir, splits=_make_splits())
        metrics = run_dir / "latent_dim_2" / "metrics.json"
        before = metrics.read_bytes()
        with pytest.raises(ArtifactError, match="Conflict"):
            run_latent_dimension_comparison(_make_config([2]), run_dir, splits=_make_splits())
        assert metrics.read_bytes() == before

    def test_explicit_overwrite_replaces_files_and_warns(
        self, out_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        run_dir = out_dir / "reused"
        run_latent_dimension_comparison(_make_config([2]), run_dir, splits=_make_splits())
        with caplog.at_level("WARNING"):
            run_latent_dimension_comparison(
                _make_config([2]), run_dir, splits=_make_splits(), overwrite=True
            )
        assert any("Overwrite enabled" in record.getMessage() for record in caplog.records)
        assert (run_dir / "latent_dim_2" / "metrics.json").exists()

    def test_empty_existing_directory_is_accepted(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True)
        run_latent_dimension_comparison(_make_config([2]), out_dir, splits=_make_splits())
        assert (out_dir / "latent_dim_2").is_dir()

    def test_two_consecutive_cli_style_runs_use_distinct_directories(self, out_dir: Path) -> None:
        """Using --output-dir as a base never reuses a previous run directory."""
        base = out_dir / "base"
        run_dirs = []
        for _ in range(2):
            run_dir = resolve_comparison_run_dir(base, base)
            run_latent_dimension_comparison(_make_config([2]), run_dir, splits=_make_splits())
            run_dirs.append(run_dir)
        assert run_dirs[0] != run_dirs[1]
        assert all((d / "latent_dimension_comparison.csv").exists() for d in run_dirs)
        assert {p.name for p in base.iterdir()} == {d.name for d in run_dirs}

    def test_two_runs_with_same_timestamp_do_not_collide(
        self, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(paths, "build_run_timestamp", lambda: "20260101_000000")
        first = build_comparison_run_dir(out_dir)
        first.mkdir(parents=True)
        second = build_comparison_run_dir(out_dir)
        assert first != second

    def test_run_directory_name_is_windows_safe(self, out_dir: Path) -> None:
        run_dir = build_comparison_run_dir(out_dir, run_name='my:run*?"<>|name')
        assert re.fullmatch(r"run_\d{8}_\d{6}_[A-Za-z0-9_\-]+", run_dir.name)
        auto = build_comparison_run_dir(out_dir)
        assert re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{6}", auto.name)

    def test_cli_rejects_existing_run_directory_with_clear_message(
        self, out_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = out_dir / "run_20260101_000000_abc123"
        run_dir.mkdir(parents=True)
        (run_dir / "latent_dimension_comparison.csv").write_text("x\n", "utf-8")
        config_path = out_dir / "config.yaml"
        config_path.write_text(
            "dataset:\n  name: mnist\nlatent_dimension_comparison:\n  enabled: true\n"
            "  dimensions: [2, 8]\n  fixed_topology: medium\n",
            "utf-8",
        )
        args = argparse.Namespace(
            config=config_path, output_dir=run_dir, overwrite=False, run_name=None
        )
        assert cli._cmd_compare_latent_dims(args) == 1
        error = capsys.readouterr().err
        assert "output conflict" in error
        assert "--overwrite" in error
        assert (run_dir / "latent_dimension_comparison.csv").read_text("utf-8") == "x\n"

    def test_cli_accepts_overwrite_flag(self) -> None:
        parser = cli._build_parser()
        args = parser.parse_args(
            ["compare-latent-dims", "--config", "c.yaml", "--output-dir", "x", "--overwrite"]
        )
        assert args.overwrite is True
        default = parser.parse_args(["compare-latent-dims", "--config", "c.yaml"])
        assert default.overwrite is False


class TestSyntheticLabelling:
    """Synthetic data is never presented as a scientific result."""

    def test_unmarked_directory_is_rejected_for_injected_data(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactError, match="smoke_test"):
            run_latent_dimension_comparison(
                _make_config([2]), tmp_path / "plain_run", splits=_make_splits()
            )
        assert not (tmp_path / "plain_run").exists()

    def test_summaries_manifest_metrics_and_csv_state_synthetic(
        self, completed_run: tuple[Path, pd.DataFrame]
    ) -> None:
        output_dir, table = completed_run
        assert "synthetic" in output_dir.name
        comparison_summary = (output_dir / "latent_dimension_comparison_summary.md").read_text(
            "utf-8"
        )
        assert "SYNTHETIC DATA" in comparison_summary
        assert "NOT scientific results" in comparison_summary
        assert (table["data_origin"] == "synthetic").all()
        for dim in DIMENSIONS:
            dim_dir = output_dir / f"latent_dim_{dim}"
            assert "SYNTHETIC DATA" in (dim_dir / "run_summary.md").read_text("utf-8")
            manifest = json.loads((dim_dir / "artifact_manifest.json").read_text("utf-8"))
            assert manifest["data_origin"] == "synthetic"
            metrics = json.loads((dim_dir / "metrics.json").read_text("utf-8"))
            assert metrics["data_origin"] == "synthetic"

    def test_real_data_summary_has_no_synthetic_notice(self, tmp_path: Path) -> None:
        record = DimensionRunRecord(
            dataset="mnist",
            experiment_type="unsupervised",
            topology_name="medium",
            hidden_dimensions=[32, 16],
            latent_dimension=2,
            seed=42,
            status="completed",
        )
        path = write_dimension_run_summary(record, tmp_path)
        text = path.read_text("utf-8")
        assert "SYNTHETIC DATA" not in text
        assert "Data origin**: real" in text
