"""Final evaluation experiment.

Loads the best checkpoint and runs a single evaluation on the test set.
This must only be called after all model selection decisions are finalized.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vae_project.config.models import ExperimentConfig
from vae_project.data.datasets import get_raw_datasets
from vae_project.data.loaders import create_data_loaders
from vae_project.data.splits import create_data_splits
from vae_project.evaluation.clustering_metrics import compute_clustering_metrics
from vae_project.evaluation.latent_extraction import extract_latent_representations
from vae_project.evaluation.reconstruction_metrics import compute_reconstruction_metrics
from vae_project.models.vae import VariationalAutoencoder
from vae_project.training.checkpointing import ModelCheckpointer
from vae_project.utils.device import get_device
from vae_project.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


def run_final_evaluation(
    config: ExperimentConfig,
    checkpoint_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run the final test-set evaluation.

    Loads the best checkpoint and evaluates on the test set once.
    All results are clearly labeled as 'final_test' metrics.

    Args:
        config: Experiment configuration.
        checkpoint_dir: Directory containing the best checkpoint.
        output_dir: Directory for final evaluation artifacts.

    Returns:
        Dictionary with final test metrics.
    """
    ensure_dir(output_dir)

    device = get_device(config.runtime.device, config.runtime.cuda_fallback_to_cpu)

    train_full, test_dataset = get_raw_datasets(
        dataset_name=config.dataset.name,
        data_dir=config.paths.data_dir,
        download=config.dataset.download,
    )
    splits = create_data_splits(
        train_full_dataset=train_full,
        test_dataset=test_dataset,
        dataset_name=config.dataset.name,
        splits_config=config.splits,
        quick_mode=config.project.quick_mode,
    )
    loaders = create_data_loaders(
        splits=splits,
        training_config=config.training,
        dataset_config=config.dataset,
    )

    model = VariationalAutoencoder(config.model).to(device)
    checkpointer = ModelCheckpointer(checkpoint_dir=checkpoint_dir)
    best_metadata = checkpointer.load_best(model=model, device=device)

    logger.warning(
        "FINAL TEST EVALUATION: Running single evaluation on test set. " "Epoch=%d, Val_loss=%.6f",
        best_metadata.epoch,
        best_metadata.val_total_loss,
    )

    # Reconstruction metrics on test set
    test_recon = compute_reconstruction_metrics(
        model=model,
        data_loader=loaders.test,
        device=device,
    )

    # Latent extraction on test set
    test_reps = extract_latent_representations(
        model=model,
        data_loader=loaders.test,
        device=device,
        split_name="test",
        class_names=splits.class_names,
        dataset_name=config.dataset.name.value,
        is_quick_mode=config.project.quick_mode,
    )
    test_reps.save(output_dir=output_dir / "latent_data")

    # Clustering metrics
    clustering = compute_clustering_metrics(
        latent_vectors=test_reps.latent_means,
        labels=test_reps.labels,
        space_name="final_test_latent",
        num_classes=splits.num_classes,
    )

    final_results: dict[str, object] = {
        "evaluation_type": "final_test",
        "checkpoint_epoch": best_metadata.epoch,
        "val_total_loss_at_checkpoint": best_metadata.val_total_loss,
        "test_reconstruction_metrics": test_recon,
        "test_clustering_metrics": clustering.as_dict(),
        "is_quick_mode": config.project.quick_mode,
    }

    # Save results
    results_path = output_dir / "final_evaluation_results.json"
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(final_results, results_file, indent=2, default=str)
    logger.info(
        "Final test results saved: %s [THESE ARE TEST METRICS]",
        results_path,
    )

    return final_results
