"""Subsidiary clustering quality metrics for latent space analysis.

These metrics are SUBSIDIARY evidence to support (but not replace)
visual analysis of the latent space. They must not be used as the
primary or sole evidence for conclusions about cluster structure.

Key methodological notes:
- Silhouette Score, Calinski-Harabasz, and Davies-Bouldin Index use
  ground-truth labels as cluster assignments, NOT discovered clusters.
- Trustworthiness measures local neighborhood preservation in projections.
- ARI and NMI compare K-Means cluster assignments to ground-truth labels
  and should be treated as exploratory tools, not validation.
- No single metric provides definitive evidence of natural clustering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusteringMetricsResult:
    """Container for clustering quality metric values.

    All metrics are computed with ground-truth labels as cluster
    assignments (not discovered clusters), unless noted otherwise.

    Attributes:
        silhouette_score: Range [-1, 1]. Higher = more compact, separated clusters.
            Computed in the specified latent/projection space.
        calinski_harabasz_index: Higher = better defined clusters.
        davies_bouldin_index: Lower = better separated clusters.
        trustworthiness: [0, 1]. Higher = better local neighborhood preservation
            in the projection. Only meaningful for projections, not the latent space.
        kmeans_ari: Adjusted Rand Index between K-Means clusters and ground truth.
            Only computed when K-Means is requested.
        kmeans_nmi: Normalized Mutual Information between K-Means and ground truth.
        space_name: Identifier for the space where metrics were computed.
        n_samples: Number of samples used for computation.
        n_samples_total: Total samples (before subsampling).
        was_subsampled: Whether a subsample was used.
    """

    silhouette_score: float | None
    calinski_harabasz_index: float | None
    davies_bouldin_index: float | None
    trustworthiness: float | None
    kmeans_ari: float | None
    kmeans_nmi: float | None
    space_name: str
    n_samples: int
    n_samples_total: int
    was_subsampled: bool

    def as_dict(self) -> dict[str, object]:
        """Return metrics as a flat dictionary for logging and CSV export."""
        return {
            "space_name": self.space_name,
            "silhouette_score": self.silhouette_score,
            "calinski_harabasz_index": self.calinski_harabasz_index,
            "davies_bouldin_index": self.davies_bouldin_index,
            "trustworthiness": self.trustworthiness,
            "kmeans_ari": self.kmeans_ari,
            "kmeans_nmi": self.kmeans_nmi,
            "n_samples": self.n_samples,
            "n_samples_total": self.n_samples_total,
            "was_subsampled": self.was_subsampled,
        }


def compute_clustering_metrics(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    space_name: str,
    num_classes: int,
    max_samples_silhouette: int | None = 5000,
    compute_kmeans: bool = False,
    kmeans_random_state: int = 42,
    trustworthiness_n_neighbors: int = 5,
    original_space: np.ndarray | None = None,
) -> ClusteringMetricsResult:
    """Compute subsidiary clustering quality metrics.

    Args:
        latent_vectors: Feature matrix, shape (N, D).
        labels: Ground-truth integer labels, shape (N,).
        space_name: Identifier for this space (e.g. 'latent_original', 'pca_2d').
        num_classes: Expected number of classes.
        max_samples_silhouette: Maximum samples for silhouette computation.
            None = use all samples (may be slow for large datasets).
        compute_kmeans: Whether to run K-Means and compute ARI/NMI.
            The number of clusters is set to num_classes.
        kmeans_random_state: Random state for K-Means reproducibility.
        trustworthiness_n_neighbors: Number of neighbors for trustworthiness.
        original_space: Original high-dimensional space for trustworthiness
            computation. If None, trustworthiness is not computed.

    Returns:
        ClusteringMetricsResult with all computed metric values.
    """
    n_total = len(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        logger.warning(
            "Cannot compute clustering metrics: only %d unique class(es) in labels.",
            len(unique_labels),
        )
        return ClusteringMetricsResult(
            silhouette_score=None,
            calinski_harabasz_index=None,
            davies_bouldin_index=None,
            trustworthiness=None,
            kmeans_ari=None,
            kmeans_nmi=None,
            space_name=space_name,
            n_samples=n_total,
            n_samples_total=n_total,
            was_subsampled=False,
        )

    # Subsample for silhouette if needed (expensive O(N^2))
    was_subsampled = False
    silhouette_indices = np.arange(n_total)
    if max_samples_silhouette is not None and n_total > max_samples_silhouette:
        rng = np.random.default_rng(42)
        silhouette_indices = rng.choice(n_total, size=max_samples_silhouette, replace=False)
        was_subsampled = True
        logger.info(
            "Silhouette computation: subsampled %d/%d samples for space '%s'",
            max_samples_silhouette,
            n_total,
            space_name,
        )

    sub_vectors = latent_vectors[silhouette_indices]
    sub_labels = labels[silhouette_indices]

    # Silhouette Score
    sil_score: float | None = None
    try:
        sil_score = float(silhouette_score(sub_vectors, sub_labels))
    except Exception as exc:
        logger.warning("Silhouette score computation failed: %s", exc)

    # Calinski-Harabasz Index (uses all samples)
    ch_index: float | None = None
    try:
        ch_index = float(calinski_harabasz_score(latent_vectors, labels))
    except Exception as exc:
        logger.warning("Calinski-Harabasz computation failed: %s", exc)

    # Davies-Bouldin Index (uses all samples)
    db_index: float | None = None
    try:
        db_index = float(davies_bouldin_score(latent_vectors, labels))
    except Exception as exc:
        logger.warning("Davies-Bouldin computation failed: %s", exc)

    # Trustworthiness (only when original space is provided)
    tw_score: float | None = None
    if original_space is not None:
        try:
            tw_score = float(
                trustworthiness(
                    original_space,
                    latent_vectors,
                    n_neighbors=trustworthiness_n_neighbors,
                )
            )
        except Exception as exc:
            logger.warning("Trustworthiness computation failed: %s", exc)

    # K-Means ARI and NMI (optional)
    kmeans_ari: float | None = None
    kmeans_nmi: float | None = None

    if compute_kmeans:
        try:
            kmeans = KMeans(
                n_clusters=num_classes,
                n_init="auto",
                random_state=kmeans_random_state,
            )
            cluster_assignments = kmeans.fit_predict(latent_vectors)
            kmeans_ari = float(adjusted_rand_score(labels, cluster_assignments))
            kmeans_nmi = float(normalized_mutual_info_score(labels, cluster_assignments))
            logger.info(
                "K-Means (n_clusters=%d, random_state=%d): ARI=%.4f, NMI=%.4f",
                num_classes,
                kmeans_random_state,
                kmeans_ari,
                kmeans_nmi,
            )
        except Exception as exc:
            logger.warning("K-Means clustering failed: %s", exc)

    n_used = len(silhouette_indices)

    logger.info(
        "Clustering metrics (%s): silhouette=%.4f, CH=%.2f, DB=%.4f, " "trustworthiness=%s",
        space_name,
        sil_score if sil_score is not None else float("nan"),
        ch_index if ch_index is not None else float("nan"),
        db_index if db_index is not None else float("nan"),
        f"{tw_score:.4f}" if tw_score is not None else "N/A",
    )

    return ClusteringMetricsResult(
        silhouette_score=sil_score,
        calinski_harabasz_index=ch_index,
        davies_bouldin_index=db_index,
        trustworthiness=tw_score,
        kmeans_ari=kmeans_ari,
        kmeans_nmi=kmeans_nmi,
        space_name=space_name,
        n_samples=n_used,
        n_samples_total=n_total,
        was_subsampled=was_subsampled,
    )
