"""UMAP projection for latent space visualization.

UMAP (Uniform Manifold Approximation and Projection) is a non-linear
dimensionality reduction method that preserves local neighborhood structure.

IMPORTANT methodological notes:
- UMAP does NOT provide explained variance ratios (unlike PCA).
- UMAP preserves local structure well but global distances may be distorted.
- The appearance of clusters in UMAP may exaggerate separation; always
  cross-check with PCA and t-SNE.
- Parameters (n_neighbors, min_dist) significantly affect the visual result.
- All UMAP parameters must be documented with each figure.

Reference:
    McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold
    Approximation and Projection for Dimension Reduction.
    https://arxiv.org/abs/1802.03426
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from vae_project.config.models import UMAPConfig

logger = logging.getLogger(__name__)


@dataclass
class UMAPResult:
    """Results from UMAP projection.

    Attributes:
        projection_2d: 2D UMAP coordinates, shape (N, 2).
        n_samples: Number of samples projected.
        latent_dimension: Input latent space dimensionality.
        n_neighbors: UMAP n_neighbors parameter used.
        min_dist: UMAP min_dist parameter used.
        metric: UMAP metric used.
        random_state: Random state used.
    """

    projection_2d: np.ndarray
    n_samples: int
    latent_dimension: int
    n_neighbors: int
    min_dist: float
    metric: str
    random_state: int

    def get_parameters(self) -> dict[str, object]:
        """Return UMAP hyperparameters as a dictionary."""
        return {
            "method": "umap",
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "random_state": self.random_state,
            "n_samples": self.n_samples,
            "latent_dimension": self.latent_dimension,
            "note": (
                "UMAP does not produce explained variance. "
                "Preserves local structure; global distances may be distorted."
            ),
        }


class UMAPProjection:
    """Fits and applies UMAP projection to latent representations.

    Args:
        config: UMAP configuration parameters.
    """

    def __init__(self, config: UMAPConfig) -> None:
        self.config = config

    def fit_transform(
        self,
        latent_vectors: np.ndarray,
    ) -> UMAPResult:
        """Fit UMAP to latent vectors and transform them to 2D.

        Args:
            latent_vectors: Input feature matrix, shape (N, D).

        Returns:
            UMAPResult with 2D projection and parameter metadata.

        Raises:
            ImportError: If umap-learn is not installed.
        """
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "umap-learn is required for UMAP projections. "
                "Install it with: pip install umap-learn"
            ) from exc

        n_samples, latent_dim = latent_vectors.shape

        logger.info(
            "Running UMAP: n_samples=%d, latent_dim=%d, "
            "n_neighbors=%d, min_dist=%.3f, metric=%s",
            n_samples,
            latent_dim,
            self.config.n_neighbors,
            self.config.min_dist,
            self.config.metric,
        )

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            random_state=self.config.random_state,
            low_memory=True,  # Safer for CPU with large datasets
        )

        projection = reducer.fit_transform(latent_vectors)

        logger.info("UMAP projection complete: output shape %s", projection.shape)

        return UMAPResult(
            projection_2d=projection,
            n_samples=n_samples,
            latent_dimension=latent_dim,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            random_state=self.config.random_state,
        )
