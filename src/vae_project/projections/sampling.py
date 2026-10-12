"""Stratified subsampling for projection visualization.

When the latent space contains more samples than can be efficiently
projected by UMAP or t-SNE, a stratified subsample is drawn to
maintain class proportions.

IMPORTANT: Any analysis on a subsample must clearly report:
- That a subsample was used
- The subsample size
- The total sample count
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def stratified_subsample(
    vectors: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a stratified subsample of latent vectors.

    Ensures each class is represented proportionally in the subsample.

    Args:
        vectors: Feature matrix, shape (N, D).
        labels: Integer labels, shape (N,).
        max_samples: Maximum number of samples to include.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (subsampled_vectors, subsampled_labels, selected_indices).
    """
    n_total = len(labels)

    if n_total <= max_samples:
        return vectors, labels, np.arange(n_total)

    rng = np.random.default_rng(seed)
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    per_class = max_samples // n_classes

    selected_indices_list: list[np.ndarray] = []

    for class_label in unique_classes:
        class_indices = np.where(labels == class_label)[0]
        n_select = min(per_class, len(class_indices))
        chosen = rng.choice(class_indices, size=n_select, replace=False)
        selected_indices_list.append(chosen)

    selected_indices = np.concatenate(selected_indices_list)
    selected_indices = np.sort(selected_indices)

    logger.info(
        "Stratified subsample: %d/%d samples selected (max_samples=%d, n_classes=%d)",
        len(selected_indices),
        n_total,
        max_samples,
        n_classes,
    )

    return vectors[selected_indices], labels[selected_indices], selected_indices
