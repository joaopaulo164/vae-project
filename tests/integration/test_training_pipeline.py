"""Integration test: training pipeline with synthetic data.

Tests the complete training loop without requiring internet access or
real dataset downloads. Uses TensorDataset fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from vae_project.config.models import (
    DevicePolicy,
    EarlyStoppingConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TopologyConfig,
    TrainingConfig,
)
from vae_project.models.vae import VariationalAutoencoder
from vae_project.training.checkpointing import ModelCheckpointer
from vae_project.training.history import TrainingHistory
from vae_project.training.trainer import VAETrainer

pytestmark = pytest.mark.integration


def _make_minimal_config() -> ExperimentConfig:
    """Create a minimal ExperimentConfig for integration tests."""
    config = ExperimentConfig()
    config.model = ModelConfig(
        latent_dimension=4,
        topology=TopologyConfig(name="test", hidden_dimensions=[32, 16]),
    )
    config.training = TrainingConfig(max_epochs=3, batch_size=16)
    config.early_stopping = EarlyStoppingConfig(enabled=True, patience=2)
    config.optimizer = OptimizerConfig(learning_rate=1e-3)
    config.loss = LossConfig()
    config.runtime = RuntimeConfig(device=DevicePolicy.CPU)
    return config


def _make_synthetic_loaders() -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    """Create synthetic train and val loaders."""
    torch.manual_seed(42)
    train_images = torch.rand(80, 1, 28, 28)
    train_labels = torch.randint(0, 10, (80,))
    val_images = torch.rand(20, 1, 28, 28)
    val_labels = torch.randint(0, 10, (20,))

    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=16)
    val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=16)
    return train_loader, val_loader


class TestTrainingPipeline:
    """Integration tests for training pipeline."""

    def test_training_runs_to_completion(self, tmp_path: Path) -> None:
        """Training runs for all epochs and returns history."""
        config = _make_minimal_config()
        model = VariationalAutoencoder(config.model)
        device = torch.device("cpu")

        checkpointer = ModelCheckpointer(tmp_path / "ckpt")
        history = TrainingHistory(tmp_path / "tables")
        train_loader, val_loader = _make_synthetic_loaders()

        trainer = VAETrainer(
            model=model,
            config=config,
            device=device,
            checkpointer=checkpointer,
            history=history,
        )
        result_history = trainer.fit(train_loader, val_loader, "mnist")

        assert result_history.num_epochs_completed > 0
        assert result_history.num_epochs_completed <= config.training.max_epochs

    def test_best_checkpoint_is_saved(self, tmp_path: Path) -> None:
        """Best model checkpoint is saved during training."""
        config = _make_minimal_config()
        model = VariationalAutoencoder(config.model)
        device = torch.device("cpu")

        checkpointer = ModelCheckpointer(tmp_path / "ckpt")
        history = TrainingHistory(tmp_path / "tables")
        train_loader, val_loader = _make_synthetic_loaders()

        trainer = VAETrainer(
            model=model,
            config=config,
            device=device,
            checkpointer=checkpointer,
            history=history,
        )
        trainer.fit(train_loader, val_loader, "mnist")

        assert checkpointer.best_checkpoint_path.exists()

    def test_checkpoint_loads_correctly(self, tmp_path: Path) -> None:
        """Saved checkpoint can be loaded into a new model."""
        config = _make_minimal_config()
        model = VariationalAutoencoder(config.model)
        device = torch.device("cpu")

        checkpointer = ModelCheckpointer(tmp_path / "ckpt")
        history = TrainingHistory(tmp_path / "tables")
        train_loader, val_loader = _make_synthetic_loaders()

        trainer = VAETrainer(
            model=model,
            config=config,
            device=device,
            checkpointer=checkpointer,
            history=history,
        )
        trainer.fit(train_loader, val_loader, "mnist")

        # Load into a fresh model
        new_model = VariationalAutoencoder(config.model)
        checkpointer.load_best(new_model, device)

        # Both models should produce the same output
        new_model.eval()
        model.eval()
        test_input = torch.rand(4, 1, 28, 28)
        with torch.no_grad():
            out_original = model(test_input).reconstruction
            out_loaded = new_model(test_input).reconstruction
        assert torch.allclose(out_original, out_loaded, atol=1e-5)

    def test_training_history_is_saved(self, tmp_path: Path) -> None:
        """Training history CSV and JSON are created."""
        config = _make_minimal_config()
        model = VariationalAutoencoder(config.model)
        device = torch.device("cpu")

        checkpointer = ModelCheckpointer(tmp_path / "ckpt")
        history = TrainingHistory(tmp_path / "tables")
        train_loader, val_loader = _make_synthetic_loaders()

        trainer = VAETrainer(
            model=model,
            config=config,
            device=device,
            checkpointer=checkpointer,
            history=history,
        )
        trainer.fit(train_loader, val_loader, "mnist")
        history.save()

        assert (tmp_path / "tables" / "training_history.csv").exists()
        assert (tmp_path / "tables" / "training_history.json").exists()
