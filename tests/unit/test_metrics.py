"""Unit tests for subsidiary metrics."""

from __future__ import annotations

import numpy as np
import pytest

from vae_project.evaluation.clustering_metrics import compute_clustering_metrics

pytestmark = pytest.mark.unit


class TestClusteringMetrics:
    """Tests for clustering quality metrics."""

    def test_metrics_return_values(
        self,
        synthetic_latent_vectors: np.ndarray,
        synthetic_labels: np.ndarray,
        num_classes: int,
    ) -> None:
        """Clustering metrics return numeric values."""
        result = compute_clustering_metrics(
            latent_vectors=synthetic_latent_vectors,
            labels=synthetic_labels,
            space_name="test_space",
            num_classes=num_classes,
            max_samples_silhouette=100,
        )
        assert result.silhouette_score is not None
        assert result.calinski_harabasz_index is not None
        assert result.davies_bouldin_index is not None

    def test_silhouette_in_valid_range(
        self,
        synthetic_latent_vectors: np.ndarray,
        synthetic_labels: np.ndarray,
        num_classes: int,
    ) -> None:
        """Silhouette score is in [-1, 1]."""
        result = compute_clustering_metrics(
            latent_vectors=synthetic_latent_vectors,
            labels=synthetic_labels,
            space_name="test_space",
            num_classes=num_classes,
            max_samples_silhouette=100,
        )
        assert result.silhouette_score is not None
        assert -1.0 <= result.silhouette_score <= 1.0

    def test_as_dict_contains_space_name(
        self,
        synthetic_latent_vectors: np.ndarray,
        synthetic_labels: np.ndarray,
        num_classes: int,
    ) -> None:
        """as_dict() includes the space name."""
        result = compute_clustering_metrics(
            latent_vectors=synthetic_latent_vectors,
            labels=synthetic_labels,
            space_name="my_space",
            num_classes=num_classes,
        )
        result_dict = result.as_dict()
        assert result_dict["space_name"] == "my_space"

    def test_single_class_returns_none_values(self) -> None:
        """Single class input returns None for metrics requiring 2+ classes."""
        vectors = np.random.randn(50, 8).astype(np.float32)
        labels = np.zeros(50, dtype=np.int64)  # All same class
        result = compute_clustering_metrics(
            latent_vectors=vectors,
            labels=labels,
            space_name="single_class",
            num_classes=1,
        )
        assert result.silhouette_score is None
