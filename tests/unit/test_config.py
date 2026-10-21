"""Unit tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from vae_project.config.loader import (
    ConfigurationError,
    load_config,
    load_config_from_dict,
)
from vae_project.config.models import (
    DatasetName,
    DevicePolicy,
    ReconstructionLossType,
)

pytestmark = pytest.mark.unit


class TestConfigLoading:
    """Tests for YAML config loading."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Config loads successfully from a valid YAML file."""
        import yaml

        config_data = {
            "dataset": {"name": "mnist"},
            "model": {
                "latent_dimension": 16,
                "topology": {"name": "small", "hidden_dimensions": [128, 64]},
            },
        }
        config_file = tmp_path / "test_config.yaml"
        with config_file.open("w") as f:
            yaml.safe_dump(config_data, f)

        config = load_config(config_file)
        assert config.dataset.name == DatasetName.MNIST
        assert config.model.latent_dimension == 16

    def test_load_nonexistent_file_raises(self) -> None:
        """FileNotFoundError raised for missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent_config.yaml"))

    def test_load_from_dict_defaults(self) -> None:
        """Loading empty dict produces default ExperimentConfig."""
        config = load_config_from_dict({})
        assert config.dataset.name == DatasetName.MNIST
        assert config.model.latent_dimension == 16
        assert config.loss.beta == 1.0

    def test_dataset_name_enum_conversion(self) -> None:
        """String dataset names are converted to DatasetName enums."""
        config = load_config_from_dict({"dataset": {"name": "fashion_mnist"}})
        assert config.dataset.name == DatasetName.FASHION_MNIST

    def test_device_enum_conversion(self) -> None:
        """String device policy is converted to DevicePolicy enum."""
        config = load_config_from_dict({"runtime": {"device": "cpu"}})
        assert config.runtime.device == DevicePolicy.CPU

    def test_loss_type_enum_conversion(self) -> None:
        """String loss type is converted to ReconstructionLossType enum."""
        config = load_config_from_dict({"loss": {"reconstruction_loss_type": "mse"}})
        assert config.loss.reconstruction_loss_type == ReconstructionLossType.MSE


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_invalid_latent_dimension_raises(self) -> None:
        """Validation error for latent_dimension < 1."""
        with pytest.raises(ConfigurationError, match="latent_dimension"):
            load_config_from_dict(
                {
                    "model": {
                        "latent_dimension": 0,
                        "topology": {"hidden_dimensions": [128]},
                    }
                }
            )

    def test_empty_hidden_dimensions_raises(self) -> None:
        """Validation error for empty hidden_dimensions."""
        with pytest.raises(ConfigurationError, match="hidden_dimensions"):
            load_config_from_dict(
                {
                    "model": {
                        "latent_dimension": 8,
                        "topology": {"hidden_dimensions": []},
                    }
                }
            )

    def test_invalid_validation_fraction_raises(self) -> None:
        """Validation error for val fraction outside (0, 1)."""
        with pytest.raises(ConfigurationError, match="validation_fraction"):
            load_config_from_dict({"splits": {"validation_fraction": 0.0}})

    def test_invalid_beta_raises(self) -> None:
        """Validation error for non-positive beta."""
        with pytest.raises(ConfigurationError, match="beta"):
            load_config_from_dict({"loss": {"beta": -1.0}})

    def test_invalid_reduction_raises(self) -> None:
        """Validation error for unknown reduction type."""
        with pytest.raises(ConfigurationError, match="reduction"):
            load_config_from_dict({"loss": {"reduction": "invalid_reduction"}})

    def test_non_standard_beta_is_valid_and_warned_by_trainer(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """beta != 1 is not a config error; the trainer logs the deviation instead."""
        import logging

        import torch

        from vae_project.models.vae import VariationalAutoencoder
        from vae_project.training.checkpointing import ModelCheckpointer
        from vae_project.training.history import TrainingHistory
        from vae_project.training.trainer import VAETrainer

        config = load_config_from_dict({"loss": {"beta": 4.0}})
        assert config.loss.beta == 4.0
        model = VariationalAutoencoder(config.model)
        with caplog.at_level(logging.WARNING, logger="vae_project.training.trainer"):
            VAETrainer(
                model=model,
                config=config,
                device=torch.device("cpu"),
                checkpointer=ModelCheckpointer(tmp_path / "beta_test"),
                history=TrainingHistory(tmp_path / "beta_test"),
            )
        assert any("beta=4.0000" in record.getMessage() for record in caplog.records)

    def test_cuda_policy_is_valid_and_checked_when_device_is_selected(self) -> None:
        """runtime.device=cuda passes validation; availability is checked by get_device."""
        from vae_project.utils.device import DeviceSetupError, get_device

        config = load_config_from_dict({"runtime": {"device": "cuda"}})
        assert config.runtime.device == DevicePolicy.CUDA
        import torch

        if not torch.cuda.is_available():
            with pytest.raises(DeviceSetupError, match="CUDA was requested"):
                get_device(config.runtime.device, config.runtime.cuda_fallback_to_cpu)

    def test_valid_config_no_errors(self) -> None:
        """Valid config with explicit values passes validation."""
        config = load_config_from_dict(
            {
                "dataset": {"name": "mnist"},
                "model": {
                    "latent_dimension": 8,
                    "topology": {"name": "small", "hidden_dimensions": [128, 64]},
                },
                "loss": {"beta": 1.0, "reduction": "sum"},
            }
        )
        assert config is not None

    def test_supervised_component_weight_validation(self) -> None:
        """Validation error when supervised weight <= 0 and component is enabled."""
        with pytest.raises(ConfigurationError, match="classification_loss_weight"):
            load_config_from_dict(
                {
                    "supervised_component": {
                        "enabled": True,
                        "classification_loss_weight": -1.0,
                        "hidden_dimensions": [64],
                    }
                }
            )


class TestLatentDimensionComparisonValidation:
    """Tests for the latent_dimension_comparison block."""

    @staticmethod
    def _config(dimensions: object) -> dict:
        return {"latent_dimension_comparison": {"dimensions": dimensions}}

    def test_list_of_dimensions_is_accepted(self) -> None:
        config = load_config_from_dict(self._config([2, 8, 16, 32]))
        assert config.latent_dimension_comparison.dimensions == [2, 8, 16, 32]
        assert config.model.latent_dimension == 16

    def test_empty_list_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="dimensions"):
            load_config_from_dict(self._config([]))

    def test_dimension_below_two_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="dimensions"):
            load_config_from_dict(self._config([1, 8]))

    def test_duplicate_dimensions_are_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="duplicate"):
            load_config_from_dict(self._config([2, 8, 8]))

    def test_non_integer_dimensions_are_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="dimensions"):
            load_config_from_dict(self._config([2, 8.5]))
        with pytest.raises(ConfigurationError, match="dimensions"):
            load_config_from_dict(self._config([2, "8"]))

    def test_model_latent_dimension_rejects_list(self) -> None:
        with pytest.raises(ConfigurationError, match="latent_dimension"):
            load_config_from_dict(
                {
                    "model": {
                        "latent_dimension": [2, 8],
                        "topology": {"hidden_dimensions": [32]},
                    }
                }
            )

    def test_default_experiment_is_unsupervised(self) -> None:
        config = load_config_from_dict({})
        assert config.experiment_type == "unsupervised"
        assert not config.supervised_component.enabled

    def test_project_comparison_configs_are_unsupervised_single_seed(self) -> None:
        for name in ("latent_dimensions_mnist", "latent_dimensions_fashion_mnist"):
            config = load_config(Path("configs") / f"{name}.yaml")
            assert config.latent_dimension_comparison.dimensions == [2, 8, 16, 32]
            assert config.latent_dimension_comparison.fixed_topology == "medium"
            assert config.experiment_type == "unsupervised"
            assert config.reproducibility.global_seed == 42
            assert config.reproducibility.deterministic_mode is True
            assert config.projections.umap.random_state == 42
            assert config.projections.tsne.random_state == 42
