"""Latent representation extraction from a trained VAE.

Extracts latent_mean, latent_log_variance, and latent_sample vectors
for all samples in a dataset split. Saves them to CSV and optionally NPZ.

The latent_mean is used for deterministic projections (PCA, UMAP, t-SNE)
because it represents the most likely encoding for each sample, without
the stochastic noise introduced by the reparameterization trick.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vae_project.models.supervised_head import SupervisedClassificationHead
from vae_project.models.vae import VariationalAutoencoder

logger = logging.getLogger(__name__)


@dataclass
class LatentRepresentations:
    """Container for extracted latent representations and metadata.

    Attributes:
        sample_indices: Array of original dataset indices, shape (N,).
        labels: Integer class labels, shape (N,).
        class_names: Mapping from label int to class name string.
        split_name: 'train', 'val', or 'test'.
        latent_means: Posterior means, shape (N, latent_dim).
        latent_log_variances: Posterior log variances, shape (N, latent_dim).
        latent_samples: Sampled latent vectors, shape (N, latent_dim).
        classifier_predictions: Optional predicted class indices, shape (N,).
        latent_dimension: Dimensionality of the latent space.
        dataset_name: Dataset identifier.
        is_quick_mode: Whether this was a quick smoke test run.
        metadata: Constant provenance values (e.g. run_id, seed,
            experiment_type) written as extra columns by ``to_dataframe``.
    """

    sample_indices: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    split_name: str
    latent_means: np.ndarray
    latent_log_variances: np.ndarray
    latent_samples: np.ndarray
    latent_dimension: int
    dataset_name: str
    classifier_predictions: np.ndarray | None = None
    is_quick_mode: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame for analysis and export.

        Returns:
            DataFrame with one row per sample and columns for:
            sample_index, split, label, class_name, [metadata columns],
            latent_mean_0..N, latent_log_var_0..N, latent_sample_0..N,
            [classifier_pred].
        """
        n_rows = len(self.labels)
        rows: dict[str, list] = {
            "sample_index": self.sample_indices.tolist(),
            "split": [self.split_name] * n_rows,
            "label": self.labels.tolist(),
            "class_name": [self.class_names[lbl] for lbl in self.labels],
            "dataset": [self.dataset_name] * n_rows,
            "latent_dimension": [self.latent_dimension] * n_rows,
        }
        for column_name, column_value in self.metadata.items():
            rows[column_name] = [column_value] * n_rows

        for dim_idx in range(self.latent_dimension):
            rows[f"latent_mean_{dim_idx}"] = self.latent_means[:, dim_idx].tolist()

        for dim_idx in range(self.latent_dimension):
            rows[f"latent_log_var_{dim_idx}"] = self.latent_log_variances[:, dim_idx].tolist()

        for dim_idx in range(self.latent_dimension):
            rows[f"latent_sample_{dim_idx}"] = self.latent_samples[:, dim_idx].tolist()

        if self.classifier_predictions is not None:
            rows["classifier_prediction"] = self.classifier_predictions.tolist()

        return pd.DataFrame(rows)

    def save(self, output_dir: Path, save_npz: bool = True) -> list[Path]:
        """Save representations to CSV and optionally NPZ.

        Args:
            output_dir: Directory where files will be written.
            save_npz: If True, also save a compressed NPZ file.

        Returns:
            List of paths to saved files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        csv_path = output_dir / "latent_representations.csv"
        df = self.to_dataframe()
        df.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)
        logger.info("Latent representations saved: %s (%d samples)", csv_path, len(df))

        if save_npz:
            npz_path = output_dir / "latent_representations.npz"
            save_data: dict[str, Any] = {
                "sample_indices": self.sample_indices,
                "labels": self.labels,
                "latent_means": self.latent_means,
                "latent_log_variances": self.latent_log_variances,
                "latent_samples": self.latent_samples,
            }
            if self.classifier_predictions is not None:
                save_data["classifier_predictions"] = self.classifier_predictions
            np.savez_compressed(npz_path, **save_data)  # type: ignore[arg-type]
            saved_paths.append(npz_path)

        return saved_paths


@torch.no_grad()
def extract_latent_representations(
    model: VariationalAutoencoder,
    data_loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    split_name: str,
    class_names: list[str],
    dataset_name: str,
    supervised_head: SupervisedClassificationHead | None = None,
    use_latent_mean_for_classifier: bool = True,
    is_quick_mode: bool = False,
    metadata: dict[str, object] | None = None,
) -> LatentRepresentations:
    """Extract latent representations for all samples in a data loader.

    The model is set to eval() mode. The reparameterization trick uses
    the mean (deterministic) during eval, but we also record the stochastic
    samples by temporarily enabling train mode for sampling.

    Args:
        model: Trained VAE model.
        data_loader: DataLoader for the split to extract.
        device: Compute device.
        split_name: 'train', 'val', or 'test'.
        class_names: List of class name strings.
        dataset_name: Dataset identifier.
        supervised_head: Optional trained classification head.
        use_latent_mean_for_classifier: Whether to use mu for classification.
        is_quick_mode: Whether this is a quick smoke test run.
        metadata: Optional constant provenance columns (run_id, seed, ...).

    Returns:
        LatentRepresentations with all extracted vectors and metadata.
    """
    model.eval()
    if supervised_head is not None:
        supervised_head.eval()

    all_indices: list[int] = []
    all_labels: list[int] = []
    all_means: list[np.ndarray] = []
    all_log_vars: list[np.ndarray] = []
    all_samples: list[np.ndarray] = []
    all_predictions: list[int] = []

    sample_counter = 0

    for batch_images, batch_labels in data_loader:
        batch_images = batch_images.to(device)
        batch_size = batch_images.size(0)

        vae_output = model(batch_images)

        # Extract mean and log variance (deterministic)
        means = vae_output.latent_mean.cpu().numpy()
        log_vars = vae_output.latent_log_variance.cpu().numpy()

        # Generate stochastic samples for this batch
        std_dev = torch.exp(0.5 * vae_output.latent_log_variance)
        epsilon = torch.randn_like(std_dev)
        stochastic_samples = (vae_output.latent_mean + epsilon * std_dev).cpu().numpy()

        all_means.append(means)
        all_log_vars.append(log_vars)
        all_samples.append(stochastic_samples)
        all_labels.extend(batch_labels.tolist())
        all_indices.extend(range(sample_counter, sample_counter + batch_size))
        sample_counter += batch_size

        if supervised_head is not None:
            latent_input = (
                vae_output.latent_mean
                if use_latent_mean_for_classifier
                else vae_output.latent_sample
            )
            logits = supervised_head(latent_input)
            predictions = logits.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions.tolist())

    latent_means_array = np.concatenate(all_means, axis=0)
    latent_log_vars_array = np.concatenate(all_log_vars, axis=0)
    latent_samples_array = np.concatenate(all_samples, axis=0)
    labels_array = np.array(all_labels, dtype=np.int64)
    indices_array = np.array(all_indices, dtype=np.int64)

    classifier_predictions = np.array(all_predictions, dtype=np.int64) if all_predictions else None

    logger.info(
        "Extracted latent representations: split=%s, n_samples=%d, latent_dim=%d",
        split_name,
        len(labels_array),
        model.latent_dimension,
    )

    return LatentRepresentations(
        sample_indices=indices_array,
        labels=labels_array,
        class_names=class_names,
        split_name=split_name,
        latent_means=latent_means_array,
        latent_log_variances=latent_log_vars_array,
        latent_samples=latent_samples_array,
        latent_dimension=model.latent_dimension,
        dataset_name=dataset_name,
        classifier_predictions=classifier_predictions,
        is_quick_mode=is_quick_mode,
        metadata=dict(metadata) if metadata else {},
    )
