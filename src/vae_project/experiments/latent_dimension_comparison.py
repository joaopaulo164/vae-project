"""Latent dimension comparison experiment.

Trains the selected topology at multiple latent dimensions and generates
a consolidated comparison table. All other parameters are held constant.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from vae_project.config.models import ExperimentConfig, ModelConfig
from vae_project.data.datasets import get_raw_datasets
from vae_project.data.loaders import create_data_loaders
from vae_project.data.splits import create_data_splits
from vae_project.models.vae import VariationalAutoencoder
from vae_project.training.checkpointing import ModelCheckpointer
from vae_project.training.history import TrainingHistory
from vae_project.training.trainer import VAETrainer
from vae_project.utils.device import get_device
from vae_project.utils.paths import ensure_dir
from vae_project.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)


def run_latent_dimension_comparison(
    config: ExperimentConfig,
    output_dir: Path,
    dimensions: list[int] | None = None,
) -> pd.DataFrame:
    """Compare the same topology across multiple latent dimensions.

    Holds constant: dataset, topology, seed, lr, batch_size, max_epochs,
    early stopping, loss definition, data splits.

    Args:
        config: Base experiment configuration.
        output_dir: Directory for comparison artifacts.
        dimensions: List of latent dimensions to compare.
            Defaults to config.latent_dimension_comparison.dimensions.

    Returns:
        DataFrame with comparison results sorted by latent_dimension.
    """
    ensure_dir(output_dir)

    compare_dims = dimensions or config.latent_dimension_comparison.dimensions

    device = get_device(config.runtime.device, config.runtime.cuda_fallback_to_cpu)

    # Load data once (same splits for all dimensions)
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

    results_rows = []

    for latent_dim in compare_dims:
        logger.info(
            "Training latent_dim=%d (topology=%s)",
            latent_dim,
            config.model.topology.name,
        )

        model_config = ModelConfig(
            latent_dimension=latent_dim,
            topology=config.model.topology,
        )
        model = VariationalAutoencoder(model_config).to(device)
        param_count = model.count_parameters()

        dim_dir = output_dir / f"latent_dim_{latent_dim}"
        ensure_dir(dim_dir)

        checkpointer = ModelCheckpointer(
            checkpoint_dir=dim_dir / "checkpoints",
            run_id=f"latent_{latent_dim}",
        )
        history = TrainingHistory(output_dir=dim_dir)

        # Same seed for each dimension to isolate the effect
        set_global_seed(config.reproducibility.global_seed)

        trainer = VAETrainer(
            model=model,
            config=config,
            device=device,
            checkpointer=checkpointer,
            history=history,
        )
        trainer.fit(
            train_loader=loaders.train,
            val_loader=loaders.val,
            dataset_name=config.dataset.name.value,
            is_quick_mode=config.project.quick_mode,
        )

        best_record = history.get_best_epoch()
        if best_record is None:
            continue

        results_rows.append(
            {
                "latent_dimension": latent_dim,
                "topology": config.model.topology.name,
                "hidden_dimensions": str(config.model.topology.hidden_dimensions),
                "total_params": param_count["total"],
                "best_epoch": best_record.epoch,
                "best_val_total_loss": best_record.val_total_loss,
                "best_val_reconstruction_loss": best_record.val_reconstruction_loss,
                "best_val_kl_divergence_loss": best_record.val_kl_divergence_loss,
                "epochs_trained": history.num_epochs_completed,
            }
        )

        history.save()

    results_df = pd.DataFrame(results_rows)
    results_df = results_df.sort_values("latent_dimension")

    output_csv = output_dir / "latent_dimension_comparison.csv"
    results_df.to_csv(output_csv, index=False)
    logger.info("Latent dimension comparison saved: %s", output_csv)

    return results_df
