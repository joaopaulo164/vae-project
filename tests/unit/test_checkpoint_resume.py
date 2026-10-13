"""Unit tests for the full-state checkpoint: saving, inference loading and resume."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from vae_project.config.loader import config_to_dict
from vae_project.config.models import (
    EarlyStoppingConfig,
    ExperimentConfig,
    ModelConfig,
    SchedulerConfig,
    TopologyConfig,
)
from vae_project.models.vae import VariationalAutoencoder
from vae_project.training.checkpointing import (
    CheckpointError,
    CheckpointMetadata,
    ModelCheckpointer,
)
from vae_project.training.early_stopping import EarlyStopping
from vae_project.training.trainer import _build_optimizer, _build_scheduler

pytestmark = pytest.mark.unit

CPU = torch.device("cpu")


def _make_model(
    latent_dimension: int = 4, hidden: list[int] | None = None
) -> VariationalAutoencoder:
    """Create a minimal VAE."""
    return VariationalAutoencoder(
        ModelConfig(
            latent_dimension=latent_dimension,
            topology=TopologyConfig(hidden_dimensions=hidden or [32, 16]),
        )
    )


def _make_metadata(epoch: int = 3, latent_dimension: int = 4) -> CheckpointMetadata:
    """Create metadata for a test checkpoint."""
    return CheckpointMetadata(
        epoch=epoch,
        val_total_loss=0.5,
        val_reconstruction_loss=0.4,
        val_kl_divergence_loss=0.1,
        val_classification_loss=0.0,
        topology_name="medium",
        hidden_dimensions=[32, 16],
        latent_dimension=latent_dimension,
        dataset_name="synthetic",
        seed=42,
        experiment_type="unsupervised",
    )


def _train_one_step(model: VariationalAutoencoder, optimizer: torch.optim.Optimizer) -> None:
    """Run one optimisation step so that the optimizer holds non-trivial state."""
    optimizer.zero_grad()
    output = model(torch.rand(8, 1, 28, 28))
    output.reconstruction.sum().backward()
    optimizer.step()


def _components() -> (
    tuple[VariationalAutoencoder, torch.optim.Optimizer, Any, EarlyStopping, ExperimentConfig]
):
    """Build model, optimizer, enabled scheduler and early stopping with real state."""
    config = ExperimentConfig()
    config.scheduler = SchedulerConfig(enabled=True, patience=1, factor=0.5)
    config.early_stopping = EarlyStoppingConfig(enabled=True, patience=3)
    model = _make_model()
    optimizer = _build_optimizer(list(model.parameters()), config.optimizer)
    scheduler = _build_scheduler(optimizer, config.scheduler)
    early_stopping = EarlyStopping(config.early_stopping)
    _train_one_step(model, optimizer)
    early_stopping.step(1.0, 0)
    early_stopping.step(1.1, 1)
    assert scheduler is not None
    scheduler.step(1.0)
    scheduler.step(1.1)
    scheduler.step(1.2)
    return model, optimizer, scheduler, early_stopping, config


def _load_raw(path: Path) -> dict[str, Any]:
    """Load a checkpoint file with ``weights_only=True``."""
    loaded: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return loaded


class TestCheckpointContents:
    """The checkpoint stores the weights and the full training state."""

    def test_model_state_dict_is_saved(self, tmp_path: Path) -> None:
        model = _make_model()
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(model, 0.5, _make_metadata())
        payload = _load_raw(checkpointer.best_checkpoint_path)
        assert set(payload["model_state_dict"]) == set(model.state_dict())

    def test_optimizer_state_dict_is_saved(self, tmp_path: Path) -> None:
        model, optimizer, _, _, _ = _components()
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(model, 0.5, _make_metadata(), optimizer=optimizer)
        payload = _load_raw(checkpointer.best_checkpoint_path)
        assert payload["optimizer_state_dict"]["state"]
        assert payload["optimizer_state_dict"]["param_groups"]

    def test_scheduler_state_saved_only_when_scheduler_exists(self, tmp_path: Path) -> None:
        model, optimizer, scheduler, _, _ = _components()
        with_scheduler = ModelCheckpointer(tmp_path / "with")
        with_scheduler.save_if_best(
            model, 0.5, _make_metadata(), optimizer=optimizer, scheduler=scheduler
        )
        without_scheduler = ModelCheckpointer(tmp_path / "without")
        without_scheduler.save_if_best(model, 0.5, _make_metadata(), optimizer=optimizer)
        assert "scheduler_state_dict" in _load_raw(with_scheduler.best_checkpoint_path)
        assert "scheduler_state_dict" not in _load_raw(without_scheduler.best_checkpoint_path)

    def test_scaler_state_saved_when_amp_enabled(self, tmp_path: Path) -> None:
        model, optimizer, _, _, _ = _components()
        scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
        checkpointer = ModelCheckpointer(tmp_path / "amp")
        checkpointer.save_if_best(model, 0.5, _make_metadata(), optimizer=optimizer, scaler=scaler)
        payload = _load_raw(checkpointer.best_checkpoint_path)
        assert payload["scaler_state_dict"]["scale"] == 1024.0

        plain = ModelCheckpointer(tmp_path / "no_amp")
        plain.save_if_best(model, 0.5, _make_metadata(), optimizer=optimizer)
        assert "scaler_state_dict" not in _load_raw(plain.best_checkpoint_path)

    def test_metadata_fields_are_stored_in_checkpoint(self, tmp_path: Path) -> None:
        model, optimizer, _, early_stopping, config = _components()
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(
            model,
            0.5,
            _make_metadata(epoch=3),
            optimizer=optimizer,
            early_stopping=early_stopping,
            resolved_config=config_to_dict(config),
        )
        payload = _load_raw(checkpointer.best_checkpoint_path)
        assert payload["epoch"] == 3
        assert payload["best_epoch"] == 3
        assert payload["best_val_total_loss"] == pytest.approx(0.5)
        assert payload["val_reconstruction_loss"] == pytest.approx(0.4)
        assert payload["val_kl_divergence_loss"] == pytest.approx(0.1)
        assert payload["val_classification_loss"] == 0.0
        assert payload["latent_dimension"] == 4
        assert payload["topology_name"] == "medium"
        assert payload["hidden_dimensions"] == [32, 16]
        assert payload["global_seed"] == 42
        assert payload["dataset_name"] == "synthetic"
        assert payload["experiment_type"] == "unsupervised"
        assert payload["early_stopping_state"]["patience_counter"] == 1
        assert payload["resolved_config"]["reproducibility"]["global_seed"] == 42


class TestInferenceLoading:
    """Inference only needs weights and metadata."""

    def test_weights_only_checkpoint_loads_for_inference(self, tmp_path: Path) -> None:
        model = _make_model()
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(model, 0.5, _make_metadata())  # no optimizer state
        fresh = _make_model()
        metadata = checkpointer.load_for_inference(fresh, CPU)
        assert metadata.latent_dimension == 4
        for expected, actual in zip(model.parameters(), fresh.parameters(), strict=True):
            assert torch.equal(expected, actual)

    def test_legacy_raw_state_dict_loads_for_inference(self, tmp_path: Path) -> None:
        """Old checkpoints (raw state dict + JSON metadata) keep working."""
        model = _make_model()
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(model, 0.5, _make_metadata())
        torch.save(model.state_dict(), checkpointer.best_checkpoint_path)  # legacy layout
        fresh = _make_model()
        metadata = checkpointer.load_best(fresh, CPU)
        assert metadata.epoch == 3
        for expected, actual in zip(model.parameters(), fresh.parameters(), strict=True):
            assert torch.equal(expected, actual)

    def test_legacy_checkpoint_without_metadata_json_fails_clearly(self, tmp_path: Path) -> None:
        model = _make_model()
        checkpointer = ModelCheckpointer(tmp_path)
        torch.save(model.state_dict(), checkpointer.best_checkpoint_path)
        with pytest.raises(CheckpointError, match="metadata"):
            checkpointer.load_for_inference(_make_model(), CPU)

    def test_latent_dimension_mismatch_is_rejected(self, tmp_path: Path) -> None:
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(_make_model(4), 0.5, _make_metadata())
        with pytest.raises(CheckpointError, match=r"latent_dimension=4.*latent_dimension=8"):
            checkpointer.load_for_inference(_make_model(8), CPU)

    def test_topology_mismatch_is_rejected(self, tmp_path: Path) -> None:
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(_make_model(4), 0.5, _make_metadata())
        with pytest.raises(CheckpointError, match="topology mismatch"):
            checkpointer.load_for_inference(_make_model(4, hidden=[64, 32]), CPU)

    def test_legacy_checkpoint_latent_mismatch_is_rejected(self, tmp_path: Path) -> None:
        model = _make_model(4)
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(model, 0.5, _make_metadata())
        torch.save(model.state_dict(), checkpointer.best_checkpoint_path)
        with pytest.raises(CheckpointError, match="latent_dimension"):
            checkpointer.load_for_inference(_make_model(8), CPU)


class TestResume:
    """Resume restores weights, optimizer, scheduler, scaler and early stopping."""

    def test_resume_restores_all_states(self, tmp_path: Path) -> None:
        model, optimizer, scheduler, early_stopping, config = _components()
        scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=512.0)
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(
            model,
            0.5,
            _make_metadata(epoch=3),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
        )

        new_model = _make_model()
        new_optimizer = _build_optimizer(list(new_model.parameters()), config.optimizer)
        new_scheduler = _build_scheduler(new_optimizer, config.scheduler)
        new_scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=2.0)
        new_early_stopping = EarlyStopping(config.early_stopping)
        assert new_scheduler is not None

        state = checkpointer.load_for_resume(
            new_model,
            new_optimizer,
            CPU,
            scheduler=new_scheduler,
            scaler=new_scaler,
            early_stopping=new_early_stopping,
        )

        assert state.next_epoch == 4
        assert state.best_val_total_loss == pytest.approx(0.5)
        for expected, actual in zip(model.parameters(), new_model.parameters(), strict=True):
            assert torch.equal(expected, actual)
        assert new_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
        for index, original in optimizer.state_dict()["state"].items():
            restored = new_optimizer.state_dict()["state"][index]
            assert torch.equal(original["exp_avg"], restored["exp_avg"])
            assert torch.equal(original["exp_avg_sq"], restored["exp_avg_sq"])
        assert new_scheduler.state_dict() == scheduler.state_dict()
        assert new_scaler.get_scale() == 512.0
        assert new_early_stopping.state_dict() == early_stopping.state_dict()
        assert new_early_stopping.patience_counter == 1

    def test_resume_makes_checkpointer_keep_the_restored_best_loss(self, tmp_path: Path) -> None:
        model, optimizer, _, _, config = _components()
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(model, 0.5, _make_metadata(), optimizer=optimizer)

        resumed = ModelCheckpointer(tmp_path)
        new_model = _make_model()
        new_optimizer = _build_optimizer(list(new_model.parameters()), config.optimizer)
        resumed.load_for_resume(new_model, new_optimizer, CPU)
        assert not resumed.save_if_best(new_model, 0.9, _make_metadata(epoch=4))
        assert resumed.save_if_best(new_model, 0.1, _make_metadata(epoch=5))

    def test_resume_from_legacy_checkpoint_gives_clear_message(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = _make_model()
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(model, 0.5, _make_metadata())
        torch.save(model.state_dict(), checkpointer.best_checkpoint_path)  # legacy layout
        optimizer = _build_optimizer(list(model.parameters()), ExperimentConfig().optimizer)

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(CheckpointError, match="cannot be used to resume") as error,
        ):
            checkpointer.load_for_resume(_make_model(), optimizer, CPU)
        assert "optimizer_state_dict" in str(error.value)
        assert "inference" in str(error.value)
        assert any("cannot be used to resume" in record.message for record in caplog.records)

    def test_resume_without_scheduler_state_is_rejected(self, tmp_path: Path) -> None:
        model, optimizer, _, _, config = _components()
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(model, 0.5, _make_metadata(), optimizer=optimizer)

        new_model = _make_model()
        new_optimizer = _build_optimizer(list(new_model.parameters()), config.optimizer)
        new_scheduler = _build_scheduler(new_optimizer, config.scheduler)
        with pytest.raises(CheckpointError, match="scheduler_state_dict"):
            checkpointer.load_for_resume(new_model, new_optimizer, CPU, scheduler=new_scheduler)

    def test_resume_validates_latent_dimension(self, tmp_path: Path) -> None:
        model, optimizer, _, _, config = _components()
        checkpointer = ModelCheckpointer(tmp_path)
        checkpointer.save_if_best(model, 0.5, _make_metadata(), optimizer=optimizer)
        other = _make_model(8)
        other_optimizer = _build_optimizer(list(other.parameters()), config.optimizer)
        with pytest.raises(CheckpointError, match="latent_dimension"):
            checkpointer.load_for_resume(other, other_optimizer, CPU)
