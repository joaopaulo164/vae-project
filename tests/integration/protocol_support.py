"""Shared helpers for the factorial protocol tests (synthetic data only).

Everything runs on CPU with tiny synthetic tensors: no downloads and no real
MNIST or Fashion-MNIST training. UMAP can be replaced by ``FakeUMAPProjection``
because numba compilation is slow; PCA and t-SNE run for real on tiny inputs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from vae_project.config.models import (
    DatasetName,
    DevicePolicy,
    EarlyStoppingConfig,
    ExperimentConfig,
    ProjectionsConfig,
    RuntimeConfig,
    SamplingConfig,
    TrainingConfig,
    TSNEConfig,
)
from vae_project.data.splits import DataSplits
from vae_project.projections.umap_projection import UMAPResult

NUM_CLASSES = 10
TRAIN_SIZE = 120
VAL_SIZE = 100
TEST_SIZE = 50
SEED = 42
REDUCED_TOPOLOGIES = ["small", "deep"]
REDUCED_DIMENSIONS = [2, 8]


class FakeUMAPProjection:
    """Test double for UMAP (avoids numba compilation in the test suite)."""

    def __init__(self, config: object) -> None:
        self.config = config

    def fit_transform(self, latent_vectors: np.ndarray) -> UMAPResult:
        """Return a deterministic random 2D embedding."""
        rng = np.random.default_rng(0)
        n_samples, latent_dim = latent_vectors.shape
        return UMAPResult(
            projection_2d=rng.standard_normal((n_samples, 2)),
            n_samples=n_samples,
            latent_dimension=latent_dim,
            n_neighbors=5,
            min_dist=0.1,
            metric="euclidean",
            random_state=SEED,
        )


def make_protocol_config(
    topologies: list[str] | None = None,
    dimensions: list[int] | None = None,
    supervised_extension: bool = True,
    execution_mode: str = "automatic",
) -> ExperimentConfig:
    """Build a small, valid factorial protocol configuration."""
    config = ExperimentConfig()
    config.dataset.name = DatasetName.MNIST
    config.dataset.download = False
    config.training = TrainingConfig(max_epochs=2, batch_size=32)
    config.early_stopping = EarlyStoppingConfig(enabled=True, patience=2)
    config.runtime = RuntimeConfig(device=DevicePolicy.CPU)
    config.reproducibility.global_seed = SEED
    config.reproducibility.deterministic_mode = True
    config.projections = ProjectionsConfig(
        methods=["pca", "umap", "tsne"],
        tsne=TSNEConfig(perplexity=5.0, max_iter=250),
        sampling=SamplingConfig(enabled=True, max_samples=60, stratified=True, seed=SEED),
    )
    config.metrics.trustworthiness_n_neighbors = 3
    config.plotting.save_svg = True
    config.plotting.save_png = True
    config.supervised_component.enabled = False
    config.factorial_comparison.enabled = True
    config.factorial_comparison.topologies = list(topologies or REDUCED_TOPOLOGIES)
    config.factorial_comparison.latent_dimensions = list(dimensions or REDUCED_DIMENSIONS)
    config.supervised_extension.enabled = supervised_extension
    config.supervised_extension.execution_mode = execution_mode
    return config


def make_synthetic_splits() -> DataSplits:
    """Return tiny MNIST-shaped synthetic splits (28x28, 10 classes)."""
    generator = torch.Generator().manual_seed(7)

    def dataset(size: int) -> TensorDataset:
        images = torch.rand(size, 1, 28, 28, generator=generator)
        labels = torch.arange(size) % NUM_CLASSES
        return TensorDataset(images, labels)

    return DataSplits(
        train_dataset=dataset(TRAIN_SIZE),
        val_dataset=dataset(VAL_SIZE),
        test_dataset=dataset(TEST_SIZE),
        train_indices=list(range(TRAIN_SIZE)),
        val_indices=list(range(VAL_SIZE)),
        class_names=[str(index) for index in range(NUM_CLASSES)],
        num_classes=NUM_CLASSES,
        dataset_name=DatasetName.MNIST,
        is_quick_mode=False,
    )
