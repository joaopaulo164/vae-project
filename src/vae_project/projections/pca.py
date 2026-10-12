"""PCA projection for latent space dimensionality reduction.

PCA provides a linear projection that maximizes explained variance.
The explained variance ratio is a meaningful quantity for PCA only —
it measures what fraction of the total linear variance is captured
by the top principal components.

IMPORTANT methodological note:
- Explained variance ratio refers to linear variance in the latent space.
- It does NOT directly measure class separability.
- High explained variance does not imply good cluster separation.
- Low explained variance does not mean the projection is useless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class PCAResult:
    """Results from PCA projection.

    Attributes:
        projection_2d: 2D projected coordinates, shape (N, 2).
        explained_variance_ratio: Fraction of variance per component.
        cumulative_explained_variance: Cumulative sum of explained_variance_ratio.
        n_components: Number of components computed.
        n_samples: Number of samples projected.
        latent_dimension: Dimensionality of the input latent space.
    """

    projection_2d: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    n_components: int
    n_samples: int
    latent_dimension: int

    def save_explained_variance_csv(self, output_path: Path) -> None:
        """Save explained variance table to CSV.

        Args:
            output_path: Output path for the CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for comp_idx in range(self.n_components):
            rows.append(
                {
                    "component": comp_idx + 1,
                    "explained_variance_ratio": float(self.explained_variance_ratio[comp_idx]),
                    "cumulative_explained_variance": float(
                        self.cumulative_explained_variance[comp_idx]
                    ),
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info("PCA explained variance saved: %s", output_path)


class PCAProjection:
    """Fits and applies PCA projection to latent representations.

    Args:
        n_components: Number of principal components to compute.
        random_state: Random state for reproducibility.
    """

    def __init__(self, n_components: int = 2, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self._pca: PCA | None = None

    def fit_transform(
        self,
        latent_vectors: np.ndarray,
    ) -> PCAResult:
        """Fit PCA to latent vectors and transform them.

        Args:
            latent_vectors: Input feature matrix, shape (N, D).
                If D == 2, PCA is skipped and data is returned as-is.

        Returns:
            PCAResult with projection and explained variance.
        """
        n_samples, latent_dim = latent_vectors.shape

        if latent_dim <= self.n_components:
            logger.info(
                "PCA: latent_dim=%d <= n_components=%d. Returning original data.",
                latent_dim,
                self.n_components,
            )
            n_comp = latent_dim
            ones = np.ones(n_comp)
            return PCAResult(
                projection_2d=latent_vectors[:, : self.n_components],
                explained_variance_ratio=ones / n_comp,
                cumulative_explained_variance=np.cumsum(ones / n_comp),
                n_components=n_comp,
                n_samples=n_samples,
                latent_dimension=latent_dim,
            )

        self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
        projection = self._pca.fit_transform(latent_vectors)

        logger.info(
            "PCA projection: latent_dim=%d -> %dD, "
            "explained_variance_per_component=%s, total=%.4f",
            latent_dim,
            self.n_components,
            [f"{v:.4f}" for v in self._pca.explained_variance_ratio_],
            sum(self._pca.explained_variance_ratio_),
        )

        return PCAResult(
            projection_2d=projection,
            explained_variance_ratio=self._pca.explained_variance_ratio_,
            cumulative_explained_variance=np.cumsum(self._pca.explained_variance_ratio_),
            n_components=self.n_components,
            n_samples=n_samples,
            latent_dimension=latent_dim,
        )
