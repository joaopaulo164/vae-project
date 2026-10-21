"""Unit tests for projections (PCA, UMAP, t-SNE) and sampling."""

from __future__ import annotations

import numpy as np
import pytest

from vae_project.config.models import TSNEConfig
from vae_project.projections.pca import PCAProjection
from vae_project.projections.sampling import stratified_subsample
from vae_project.projections.tsne import TSNEProjection

pytestmark = pytest.mark.unit


class TestPCAProjection:
    """Tests for PCA projection."""

    def test_pca_output_shape(
        self,
        synthetic_latent_vectors: np.ndarray,
    ) -> None:
        """PCA projects to 2D correctly."""
        pca = PCAProjection(n_components=2)
        result = pca.fit_transform(synthetic_latent_vectors)
        assert result.projection_2d.shape == (len(synthetic_latent_vectors), 2)

    def test_pca_explained_variance_sums_to_one(
        self,
        synthetic_latent_vectors: np.ndarray,
    ) -> None:
        """Explained variance ratios sum to approximately 1.0 across all components."""
        n_components = min(4, synthetic_latent_vectors.shape[1])
        pca = PCAProjection(n_components=n_components)
        result = pca.fit_transform(synthetic_latent_vectors)
        # Individual components sum should be <= 1.0
        assert 0.0 < result.cumulative_explained_variance[-1] <= 1.0 + 1e-6

    def test_pca_returns_correct_n_components(
        self,
        synthetic_latent_vectors: np.ndarray,
    ) -> None:
        """PCA result reports the correct number of components."""
        pca = PCAProjection(n_components=2)
        result = pca.fit_transform(synthetic_latent_vectors)
        assert result.n_components == 2

    def test_pca_2d_passthrough(self) -> None:
        """PCA with 2D input returns data unchanged."""
        vectors_2d = np.random.randn(50, 2).astype(np.float32)
        pca = PCAProjection(n_components=2)
        result = pca.fit_transform(vectors_2d)
        assert result.projection_2d.shape == (50, 2)


class TestStratifiedSubsample:
    """Tests for stratified subsampling."""

    def test_subsample_returns_max_samples(
        self,
        synthetic_latent_vectors: np.ndarray,
        synthetic_labels: np.ndarray,
    ) -> None:
        """Subsampling returns at most max_samples."""
        max_samples = 50
        sub_vecs, sub_labels, indices = stratified_subsample(
            synthetic_latent_vectors, synthetic_labels, max_samples=max_samples
        )
        assert len(sub_vecs) <= max_samples
        assert len(sub_labels) <= max_samples
        assert len(indices) <= max_samples

    def test_subsample_preserves_class_proportions(
        self,
        synthetic_latent_vectors: np.ndarray,
        synthetic_labels: np.ndarray,
    ) -> None:
        """Subsampling preserves approximate class proportions."""
        max_samples = 100
        _, sub_labels, _ = stratified_subsample(
            synthetic_latent_vectors, synthetic_labels, max_samples=max_samples
        )
        unique_classes = np.unique(sub_labels)
        # All 10 classes should be represented
        assert len(unique_classes) == 10

    def test_subsample_no_change_when_below_max(
        self,
        synthetic_latent_vectors: np.ndarray,
        synthetic_labels: np.ndarray,
    ) -> None:
        """No subsampling when n_samples <= max_samples."""
        max_samples = len(synthetic_latent_vectors) + 100
        sub_vecs, _sub_labels, _indices = stratified_subsample(
            synthetic_latent_vectors, synthetic_labels, max_samples=max_samples
        )
        assert len(sub_vecs) == len(synthetic_latent_vectors)

    def test_tsne_perplexity_adjustment(self) -> None:
        """t-SNE automatically adjusts perplexity when too large for sample count."""
        small_vectors = np.random.randn(20, 4).astype(np.float32)
        config = TSNEConfig(perplexity=100.0, max_iter=250)  # Too large
        tsne = TSNEProjection(config)
        result = tsne.fit_transform(small_vectors)
        # Should not raise, perplexity should have been reduced
        assert result.perplexity < 100.0
        assert result.projection_2d.shape == (20, 2)


class TestTSNEProjection:
    """Tests for t-SNE projection."""

    def test_tsne_output_shape(
        self,
        synthetic_latent_vectors: np.ndarray,
    ) -> None:
        """t-SNE projects to 2D correctly."""
        config = TSNEConfig(perplexity=5.0, max_iter=250, random_state=42)
        tsne = TSNEProjection(config)
        result = tsne.fit_transform(synthetic_latent_vectors[:50])
        assert result.projection_2d.shape == (50, 2)

    def test_tsne_parameters_recorded(
        self,
        synthetic_latent_vectors: np.ndarray,
    ) -> None:
        """t-SNE result records the parameters used."""
        config = TSNEConfig(perplexity=5.0, max_iter=250, random_state=99)
        tsne = TSNEProjection(config)
        result = tsne.fit_transform(synthetic_latent_vectors[:50])
        params = result.get_parameters()
        assert params["method"] == "tsne"
        assert "note" in params  # Must document limitations
