"""Unit tests for model checkpointing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vae_project.config.models import ModelConfig, TopologyConfig
from vae_project.models.vae import VariationalAutoencoder
from vae_project.training.checkpointing import (
    CheckpointError,
    CheckpointMetadata,
    ModelCheckpointer,
)

pytestmark = pytest.mark.unit


def _make_model() -> VariationalAutoencoder:
    """Create a minimal VAE for checkpoint tests."""
    config = ModelConfig(
        latent_dimension=4,
        topology=TopologyConfig(hidden_dimensions=[32, 16]),
    )
    return VariationalAutoencoder(config)


def _make_metadata(epoch: int = 5, val_loss: float = 0.5) -> CheckpointMetadata:
    """Create metadata for a test checkpoint."""
    return CheckpointMetadata(
        epoch=epoch,
        val_total_loss=val_loss,
        val_reconstruction_loss=0.4,
        val_kl_divergence_loss=0.1,
        topology_name="test",
        hidden_dimensions=[32, 16],
        latent_dimension=4,
        dataset_name="mnist",
    )


class TestModelCheckpointer:
    """Tests for checkpoint save/load functionality."""

    def test_save_if_best_saves_on_improvement(self, tmp_path: Path) -> None:
        """Checkpoint is saved when val_loss improves."""
        model = _make_model()
        checkpointer = ModelCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        saved = checkpointer.save_if_best(
            model=model, val_loss=0.5, metadata=_make_metadata(epoch=0, val_loss=0.5)
        )
        assert saved
        assert checkpointer.best_checkpoint_path.exists()

    def test_save_if_best_does_not_save_on_regression(self, tmp_path: Path) -> None:
        """Checkpoint is NOT saved when val_loss does not improve."""
        model = _make_model()
        checkpointer = ModelCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        checkpointer.save_if_best(model, 0.5, _make_metadata(0, 0.5))
        saved = checkpointer.save_if_best(model, 0.6, _make_metadata(1, 0.6))
        assert not saved

    def test_load_best_restores_weights(self, tmp_path: Path) -> None:
        """Loading best checkpoint restores model weights."""
        model = _make_model()
        for param in model.parameters():
            param.data.fill_(1.0)

        checkpointer = ModelCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        checkpointer.save_if_best(model, 0.3, _make_metadata())

        new_model = _make_model()
        for param in new_model.parameters():
            param.data.fill_(0.0)

        checkpointer.load_best(new_model, device=torch.device("cpu"))

        for old, new in zip(model.parameters(), new_model.parameters(), strict=True):
            assert torch.allclose(old, new)

    def test_load_best_raises_when_no_checkpoint(self, tmp_path: Path) -> None:
        """CheckpointError raised when no checkpoint exists."""
        model = _make_model()
        checkpointer = ModelCheckpointer(checkpoint_dir=tmp_path / "empty_ckpt")
        with pytest.raises(CheckpointError):
            checkpointer.load_best(model, device=torch.device("cpu"))

    def test_metadata_is_saved_and_loadable(self, tmp_path: Path) -> None:
        """Checkpoint metadata is correctly saved and loaded."""
        model = _make_model()
        metadata = _make_metadata(epoch=7, val_loss=0.25)
        checkpointer = ModelCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        checkpointer.save_if_best(model, 0.25, metadata)
        loaded = checkpointer.get_metadata()
        assert loaded is not None
        assert loaded.epoch == 7
        assert abs(loaded.val_total_loss - 0.25) < 1e-6

    def test_load_best_rejects_latent_dimension_mismatch(self, tmp_path: Path) -> None:
        """A checkpoint of another latent dimension is never loaded."""
        checkpointer = ModelCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        checkpointer.save_if_best(_make_model(), 0.3, _make_metadata())
        other = VariationalAutoencoder(
            ModelConfig(latent_dimension=8, topology=TopologyConfig(hidden_dimensions=[32, 16]))
        )
        with pytest.raises(CheckpointError, match="latent"):
            checkpointer.load_best(other, device=torch.device("cpu"))

    def test_metadata_preserves_seed_and_experiment_type(self, tmp_path: Path) -> None:
        """Seed and experiment type are stored in the checkpoint metadata."""
        metadata = _make_metadata()
        metadata.seed = 42
        metadata.experiment_type = "unsupervised"
        checkpointer = ModelCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        checkpointer.save_if_best(_make_model(), 0.3, metadata)
        loaded = checkpointer.get_metadata()
        assert loaded is not None
        assert loaded.seed == 42
        assert loaded.experiment_type == "unsupervised"
        assert loaded.latent_dimension == 4
