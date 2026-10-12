"""Shared pytest fixtures and configuration.

All test fixtures use synthetic data and do NOT require internet access
or dataset downloads. Real datasets are only used in production runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from vae_project.config.models import (
    LossConfig,
    ModelConfig,
    TopologyConfig,
)
from vae_project.models.vae import VariationalAutoencoder

# ===========================================================================
# Synthetic dataset fixtures (no internet access required)
# ===========================================================================


@pytest.fixture
def batch_size() -> int:
    """Standard batch size for tests."""
    return 32


@pytest.fixture
def latent_dim() -> int:
    """Standard latent dimension for tests."""
    return 8


@pytest.fixture
def input_dim() -> int:
    """MNIST-like flattened input dimension."""
    return 784


@pytest.fixture
def num_classes() -> int:
    """Number of synthetic classes."""
    return 10


@pytest.fixture
def synthetic_batch(batch_size: int, input_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A synthetic batch of MNIST-like images with random labels.

    Returns:
        Tuple of (images_4d, labels) tensors.
    """
    torch.manual_seed(42)
    images = torch.rand(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


@pytest.fixture
def small_model_config(latent_dim: int, input_dim: int) -> ModelConfig:
    """A small VAE config for fast unit tests."""
    return ModelConfig(
        latent_dimension=latent_dim,
        topology=TopologyConfig(name="test_small", hidden_dimensions=[64, 32]),
        input_dimension=input_dim,
    )


@pytest.fixture
def vae_model(small_model_config: ModelConfig) -> VariationalAutoencoder:
    """Instantiated VAE model for testing."""
    return VariationalAutoencoder(small_model_config)


@pytest.fixture
def loss_config() -> LossConfig:
    """Default loss configuration for testing."""
    return LossConfig()


@pytest.fixture
def synthetic_train_loader() -> DataLoader:  # type: ignore[type-arg]
    """Synthetic training DataLoader (no download required)."""
    torch.manual_seed(42)
    images = torch.rand(100, 1, 28, 28)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=32, shuffle=False)


@pytest.fixture
def synthetic_val_loader() -> DataLoader:  # type: ignore[type-arg]
    """Synthetic validation DataLoader."""
    torch.manual_seed(99)
    images = torch.rand(20, 1, 28, 28)
    labels = torch.randint(0, 10, (20,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=32, shuffle=False)


@pytest.fixture
def synthetic_latent_vectors(latent_dim: int) -> np.ndarray:
    """Synthetic latent vectors for projection tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, latent_dim)).astype(np.float32)


@pytest.fixture
def synthetic_labels(num_classes: int) -> np.ndarray:
    """Synthetic balanced integer labels."""
    n_per_class = 20
    return np.repeat(np.arange(num_classes), n_per_class)
