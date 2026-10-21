"""Main evaluator that orchestrates all evaluation steps.

Coordinates latent extraction, projection, metrics, and visualization
for a trained VAE model. Separates validation evaluation (for model
selection) from final test evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from vae_project.config.models import ExperimentConfig
from vae_project.data.loaders import DataLoaders
from vae_project.data.splits import DataSplits
from vae_project.evaluation.classification_metrics import (
    train_linear_classifier,
)
from vae_project.evaluation.clustering_metrics import (
    compute_clustering_metrics,
)
from vae_project.evaluation.latent_extraction import (
    extract_latent_representations,
)
from vae_project.evaluation.reconstruction_metrics import compute_reconstruction_metrics
from vae_project.models.supervised_head import SupervisedClassificationHead
from vae_project.models.vae import VariationalAutoencoder

logger = logging.getLogger(__name__)


class VAEEvaluator:
    """Orchestrates evaluation of a trained VAE model.

    Args:
        model: Trained VAE model (best checkpoint loaded).
        config: Experiment configuration.
        device: Compute device.
        run_dirs: Dictionary of run subdirectory paths.
        splits: Data splits with class names and metadata.
        supervised_head: Optional supervised head (if trained).
    """

    def __init__(
        self,
        model: VariationalAutoencoder,
        config: ExperimentConfig,
        device: torch.device,
        run_dirs: dict[str, Path],
        splits: DataSplits,
        supervised_head: SupervisedClassificationHead | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.run_dirs = run_dirs
        self.splits = splits
        self.supervised_head = supervised_head

    def evaluate_validation(
        self,
        loaders: DataLoaders,
    ) -> dict[str, object]:
        """Run evaluation on the validation set (for model selection support).

        This method is safe to call during the model selection phase.
        It must NOT use the test loader.

        Args:
            loaders: DataLoaders (only train and val are used here).

        Returns:
            Dictionary with validation metrics and artifact paths.
        """
        logger.info("Running validation evaluation (no test set access).")
        return self._run_evaluation(loaders, include_test=False)

    def evaluate_final(
        self,
        loaders: DataLoaders,
    ) -> dict[str, object]:
        """Run final evaluation including the test set.

        MUST only be called once, after all model/config decisions are
        finalized. Results are clearly labeled as 'test' metrics.

        Args:
            loaders: DataLoaders (all three splits used).

        Returns:
            Dictionary with final metrics and artifact paths.
        """
        logger.info("Running FINAL evaluation (includes test set).")
        logger.warning(
            "Final test evaluation is being run. This should only happen ONCE "
            "after all model selection decisions have been made."
        )
        return self._run_evaluation(loaders, include_test=True)

    def _run_evaluation(
        self,
        loaders: DataLoaders,
        include_test: bool,
    ) -> dict[str, object]:
        """Internal evaluation orchestration.

        Args:
            loaders: DataLoaders for all splits.
            include_test: Whether to include test set evaluation.

        Returns:
            Dictionary with metrics and artifact paths.
        """
        results: dict[str, object] = {}

        # Extract latent representations
        train_representations = extract_latent_representations(
            model=self.model,
            data_loader=loaders.train,
            device=self.device,
            split_name="train",
            class_names=self.splits.class_names,
            dataset_name=self.splits.dataset_name.value,
            supervised_head=self.supervised_head,
            is_quick_mode=self.splits.is_quick_mode,
        )

        val_representations = extract_latent_representations(
            model=self.model,
            data_loader=loaders.val,
            device=self.device,
            split_name="val",
            class_names=self.splits.class_names,
            dataset_name=self.splits.dataset_name.value,
            supervised_head=self.supervised_head,
            is_quick_mode=self.splits.is_quick_mode,
        )

        # Save validation representations
        val_rep_paths = val_representations.save(
            output_dir=self.run_dirs["latent_data"],
            save_npz=self.config.artifacts.save_latent_npz,
        )
        results["val_latent_paths"] = [str(p) for p in val_rep_paths]

        # Reconstruction metrics
        recon_metrics = compute_reconstruction_metrics(
            model=self.model,
            data_loader=loaders.val,
            device=self.device,
        )
        results["val_reconstruction_metrics"] = recon_metrics

        # Clustering metrics on latent space
        metrics_config = self.config.metrics
        clustering_metrics = compute_clustering_metrics(
            latent_vectors=val_representations.latent_means,
            labels=val_representations.labels,
            space_name="val_latent_original",
            num_classes=self.splits.num_classes,
            max_samples_silhouette=metrics_config.silhouette_sample_size,
            compute_kmeans=False,
            kmeans_random_state=metrics_config.kmeans_random_state,
        )
        results["val_clustering_metrics"] = clustering_metrics.as_dict()

        # Linear classifier (subsidiary)
        if metrics_config.compute_linear_classifier:
            classifier_result = train_linear_classifier(
                train_latents=train_representations.latent_means,
                train_labels=train_representations.labels,
                val_latents=val_representations.latent_means,
                val_labels=val_representations.labels,
                class_names=self.splits.class_names,
                space_name="latent_means_val",
            )
            results["linear_classifier_val"] = classifier_result.as_dict()

        if include_test:
            test_representations = extract_latent_representations(
                model=self.model,
                data_loader=loaders.test,
                device=self.device,
                split_name="test",
                class_names=self.splits.class_names,
                dataset_name=self.splits.dataset_name.value,
                supervised_head=self.supervised_head,
                is_quick_mode=self.splits.is_quick_mode,
            )

            test_rep_paths = test_representations.save(
                output_dir=self.run_dirs["latent_data"],
                save_npz=self.config.artifacts.save_latent_npz,
            )
            results["test_latent_paths"] = [str(p) for p in test_rep_paths]

            test_recon = compute_reconstruction_metrics(
                model=self.model,
                data_loader=loaders.test,
                device=self.device,
            )
            results["test_reconstruction_metrics"] = test_recon
            results["test_representations"] = test_representations

        results["val_representations"] = val_representations
        results["train_representations"] = train_representations

        # Save metrics JSON
        metrics_path = self.run_dirs.get("tables", Path(".")) / "metrics.json"
        self._save_metrics_json(results, metrics_path, include_test)

        return results

    def _save_metrics_json(
        self,
        results: dict[str, object],
        output_path: Path,
        is_final: bool,
    ) -> None:
        """Save scalar metrics to JSON (excluding large array objects).

        Args:
            results: Evaluation results dictionary.
            output_path: Output JSON path.
            is_final: Whether this includes final test metrics.
        """
        serializable: dict[str, object] = {"is_final_evaluation": is_final}

        for key, value in results.items():
            if isinstance(value, dict):
                serializable[key] = value
            elif isinstance(value, (int, float, str, bool)):
                serializable[key] = value
            # Skip non-serializable objects like LatentRepresentations

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(serializable, json_file, indent=2, default=str)
        logger.info("Metrics saved: %s", output_path)
