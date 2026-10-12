"""Projections package for dimensionality reduction."""

from vae_project.projections.pca import PCAProjection, PCAResult
from vae_project.projections.sampling import stratified_subsample
from vae_project.projections.tsne import TSNEProjection, TSNEResult
from vae_project.projections.umap_projection import UMAPProjection, UMAPResult

__all__ = [
    "PCAProjection",
    "PCAResult",
    "TSNEProjection",
    "TSNEResult",
    "UMAPProjection",
    "UMAPResult",
    "stratified_subsample",
]
