"""Unit tests for data splitting."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from vae_project.config.models import DatasetName, SplitsConfig
from vae_project.data.splits import create_data_splits

pytestmark = pytest.mark.unit


def _make_synthetic_dataset(n: int, n_classes: int = 10) -> TensorDataset:
    """Create a synthetic TensorDataset with labels."""
    torch.manual_seed(42)
    images = torch.rand(n, 1, 28, 28)
    labels = torch.randint(0, n_classes, (n,))
    return TensorDataset(images, labels)


class TestDataSplits:
    """Tests for reproducible data splitting."""

    def test_split_sizes_sum_correctly(self) -> None:
        """Train + val sizes sum to the full training set size."""
        n_total = 500
        val_fraction = 0.1
        train_full = _make_synthetic_dataset(n_total)
        test_ds = _make_synthetic_dataset(100)

        splits = create_data_splits(
            train_full_dataset=train_full,
            test_dataset=test_ds,
            dataset_name=DatasetName.MNIST,
            splits_config=SplitsConfig(validation_fraction=val_fraction),
        )

        total = len(splits.train_indices) + len(splits.val_indices)
        assert total == n_total

    def test_split_is_reproducible(self) -> None:
        """Same seed produces identical splits."""
        train_full = _make_synthetic_dataset(200)
        test_ds = _make_synthetic_dataset(50)
        config = SplitsConfig(seed=42)

        splits1 = create_data_splits(train_full, test_ds, DatasetName.MNIST, config)
        splits2 = create_data_splits(train_full, test_ds, DatasetName.MNIST, config)

        assert splits1.train_indices == splits2.train_indices
        assert splits1.val_indices == splits2.val_indices

    def test_different_seeds_produce_different_splits(self) -> None:
        """Different seeds produce different splits."""
        train_full = _make_synthetic_dataset(200)
        test_ds = _make_synthetic_dataset(50)

        splits1 = create_data_splits(train_full, test_ds, DatasetName.MNIST, SplitsConfig(seed=42))
        splits2 = create_data_splits(train_full, test_ds, DatasetName.MNIST, SplitsConfig(seed=99))
        assert splits1.train_indices != splits2.train_indices

    def test_no_overlap_between_train_and_val(self) -> None:
        """Train and validation indices do not overlap."""
        train_full = _make_synthetic_dataset(200)
        test_ds = _make_synthetic_dataset(50)

        splits = create_data_splits(train_full, test_ds, DatasetName.MNIST, SplitsConfig())

        train_set = set(splits.train_indices)
        val_set = set(splits.val_indices)
        assert len(train_set & val_set) == 0

    def test_quick_mode_uses_smaller_subset(self) -> None:
        """Quick mode uses a smaller training subset."""
        train_full = _make_synthetic_dataset(1000)
        test_ds = _make_synthetic_dataset(200)
        config = SplitsConfig(
            quick_mode_train_size=100,
            quick_mode_val_size=20,
        )

        full_splits = create_data_splits(train_full, test_ds, DatasetName.MNIST, config)
        quick_splits = create_data_splits(
            train_full, test_ds, DatasetName.MNIST, config, quick_mode=True
        )

        assert len(quick_splits.train_indices) < len(full_splits.train_indices)
        assert quick_splits.is_quick_mode

    def test_class_distribution_has_all_classes(self) -> None:
        """Class distribution records all 10 classes."""
        train_full = _make_synthetic_dataset(500)
        test_ds = _make_synthetic_dataset(100)

        splits = create_data_splits(train_full, test_ds, DatasetName.MNIST, SplitsConfig())
        classes = splits.class_distribution.get("classes", [])
        assert len(classes) == 10

    def test_fashion_mnist_class_names(self) -> None:
        """Fashion-MNIST splits have correct class names."""
        train_full = _make_synthetic_dataset(100)
        test_ds = _make_synthetic_dataset(20)

        splits = create_data_splits(train_full, test_ds, DatasetName.FASHION_MNIST, SplitsConfig())
        assert "T-shirt/top" in splits.class_names
        assert "Ankle boot" in splits.class_names
