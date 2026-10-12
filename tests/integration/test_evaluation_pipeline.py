"""Integration test: evaluation pipeline with synthetic data."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from vae_project.config.models import ModelConfig, TopologyConfig
from vae_project.evaluation.latent_extraction import extract_latent_representations
from vae_project.models.vae import VariationalAutoencoder
from vae_project.projections.pca import PCAProjection
from vae_project.reporting.artifact_manifest import ArtifactManifest

pytestmark = pytest.mark.integration


def _make_model_and_loader() -> tuple[VariationalAutoencoder, DataLoader]:  # type: ignore[type-arg]
    """Create a small VAE and a synthetic DataLoader."""
    config = ModelConfig(
        latent_dimension=8,
        topology=TopologyConfig(hidden_dimensions=[32]),
    )
    model = VariationalAutoencoder(config)

    torch.manual_seed(42)
    images = torch.rand(40, 1, 28, 28)
    labels = torch.randint(0, 10, (40,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16)

    return model, loader


class TestEvaluationPipeline:
    """Integration tests for the evaluation pipeline."""

    def test_latent_extraction_produces_correct_shapes(self) -> None:
        """Latent extraction returns arrays with correct shapes."""
        model, loader = _make_model_and_loader()
        device = torch.device("cpu")

        reps = extract_latent_representations(
            model=model,
            data_loader=loader,
            device=device,
            split_name="test",
            class_names=[str(i) for i in range(10)],
            dataset_name="mnist",
        )

        assert reps.latent_means.shape == (40, 8)
        assert reps.latent_log_variances.shape == (40, 8)
        assert reps.latent_samples.shape == (40, 8)
        assert len(reps.labels) == 40

    def test_latent_to_dataframe(self) -> None:
        """LatentRepresentations converts to DataFrame correctly."""
        model, loader = _make_model_and_loader()
        device = torch.device("cpu")

        reps = extract_latent_representations(
            model=model,
            data_loader=loader,
            device=device,
            split_name="val",
            class_names=[str(i) for i in range(10)],
            dataset_name="mnist",
        )
        df = reps.to_dataframe()
        assert "label" in df.columns
        assert "split" in df.columns
        assert "latent_mean_0" in df.columns
        assert len(df) == 40

    def test_latent_save_creates_files(self, tmp_path: Path) -> None:
        """save() creates CSV and NPZ files."""
        model, loader = _make_model_and_loader()
        device = torch.device("cpu")

        reps = extract_latent_representations(
            model=model,
            data_loader=loader,
            device=device,
            split_name="val",
            class_names=[str(i) for i in range(10)],
            dataset_name="mnist",
        )
        saved_paths = reps.save(output_dir=tmp_path, save_npz=True)
        assert any(p.suffix == ".csv" for p in saved_paths)
        assert any(p.suffix == ".npz" for p in saved_paths)

    def test_pca_projection_integration(self) -> None:
        """PCA projection runs on latent representations without errors."""
        model, loader = _make_model_and_loader()
        device = torch.device("cpu")

        reps = extract_latent_representations(
            model=model,
            data_loader=loader,
            device=device,
            split_name="val",
            class_names=[str(i) for i in range(10)],
            dataset_name="mnist",
        )
        pca = PCAProjection(n_components=2)
        result = pca.fit_transform(reps.latent_means)
        assert result.projection_2d.shape == (40, 2)
        assert result.explained_variance_ratio is not None

    def test_artifact_manifest_scan(self, tmp_path: Path) -> None:
        """ArtifactManifest scans a directory and saves JSON."""
        # Create some dummy files
        (tmp_path / "figure.png").write_bytes(b"fake png")
        (tmp_path / "table.csv").write_text("a,b\n1,2")

        manifest = ArtifactManifest(run_dir=tmp_path)
        manifest.register_run_dir_scan()
        manifest_path = manifest.save()

        assert manifest_path.exists()
        assert len(manifest.get_entries()) >= 2
