"""t-SNE projection for latent space visualization.

t-SNE (t-distributed Stochastic Neighbor Embedding) is a non-linear
technique particularly effective at revealing cluster structure in
high-dimensional data.

IMPORTANT methodological notes:
- t-SNE does NOT provide explained variance ratios (unlike PCA).
- Global distances and cluster sizes in t-SNE plots are NOT directly
  interpretable as true distances in the original space.
- t-SNE preserves local neighborhood structure only.
- The perplexity parameter must be much smaller than the number of samples.
- Different random states can produce different layouts — always report
  the random_state used.
- Do not interpret cluster sizes as indicating cluster density.

Reference:
    van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE.
    Journal of Machine Learning Research, 9, 2579-2605.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from vae_project.config.models import TSNEConfig

logger = logging.getLogger(__name__)


@dataclass
class TSNEResult:
    """Results from t-SNE projection.

    Attributes:
        projection_2d: 2D t-SNE coordinates, shape (N, 2).
        n_samples: Number of samples projected.
        latent_dimension: Input latent space dimensionality.
        perplexity: Perplexity parameter used.
        learning_rate: Learning rate used.
        max_iter: Number of iterations.
        init: Initialization method.
        random_state: Random state used.
    """

    projection_2d: np.ndarray
    n_samples: int
    latent_dimension: int
    perplexity: float
    learning_rate: float
    max_iter: int
    init: str
    random_state: int

    def get_parameters(self) -> dict[str, object]:
        """Return t-SNE hyperparameters as a dictionary."""
        return {
            "method": "tsne",
            "perplexity": self.perplexity,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "init": self.init,
            "random_state": self.random_state,
            "n_samples": self.n_samples,
            "latent_dimension": self.latent_dimension,
            "note": (
                "t-SNE does not produce explained variance. "
                "Global distances and cluster sizes are NOT directly interpretable."
            ),
        }


class TSNEProjection:
    """Fits and applies t-SNE projection to latent representations.

    Args:
        config: t-SNE configuration parameters.
    """

    def __init__(self, config: TSNEConfig) -> None:
        self.config = config

    def fit_transform(
        self,
        latent_vectors: np.ndarray,
    ) -> TSNEResult:
        """Fit t-SNE to latent vectors and transform them to 2D.

        Validates that perplexity is appropriate for the sample size.
        t-SNE requires perplexity < n_samples / 3 (approximately).

        Args:
            latent_vectors: Input feature matrix, shape (N, D).

        Returns:
            TSNEResult with 2D projection and parameter metadata.

        Raises:
            ValueError: If perplexity is too large for the number of samples.
        """
        from sklearn.manifold import TSNE

        n_samples, latent_dim = latent_vectors.shape

        # Validate perplexity
        effective_perplexity = self.config.perplexity

        if effective_perplexity >= n_samples:
            effective_perplexity = max(5.0, float(n_samples // 3))
            logger.warning(
                "t-SNE perplexity=%.1f >= n_samples=%d. "
                "Reducing to %.1f for numerical stability.",
                self.config.perplexity,
                n_samples,
                effective_perplexity,
            )

        logger.info(
            "Running t-SNE: n_samples=%d, latent_dim=%d, " "perplexity=%.1f, max_iter=%d, init=%s",
            n_samples,
            latent_dim,
            effective_perplexity,
            self.config.max_iter,
            self.config.init,
        )

        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            init=self.config.init,
            random_state=self.config.random_state,
            n_jobs=1,  # Single thread for reproducibility on CPU
        )

        projection = tsne.fit_transform(latent_vectors)

        logger.info("t-SNE projection complete: output shape %s", projection.shape)

        return TSNEResult(
            projection_2d=projection,
            n_samples=n_samples,
            latent_dimension=latent_dim,
            perplexity=effective_perplexity,
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            init=self.config.init,
            random_state=self.config.random_state,
        )
