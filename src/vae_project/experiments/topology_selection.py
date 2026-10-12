"""Topology selection experiment.

Trains each candidate topology and selects the one with the lowest
validation total loss. Uses only training and validation sets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from vae_project.config.models import ExperimentConfig, ModelConfig, TopologyConfig
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

# Standard topology candidates
TOPOLOGY_CANDIDATES: dict[str, list[int]] = {
    "small": [256, 128],
    "medium": [512, 256],
    "deep": [512, 256, 128],
}


def run_topology_selection(
    config: ExperimentConfig,
    output_dir: Path,
    topology_candidates: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Run topology selection experiment.

    Trains each candidate topology and returns a comparison table.
    The best topology is the one with the lowest validation total loss.

    Args:
        config: Base experiment configuration (topology will be overridden).
        output_dir: Directory for topology selection artifacts.
        topology_candidates: Dictionary of topology_name -> hidden_dimensions.
            Defaults to TOPOLOGY_CANDIDATES.

    Returns:
        DataFrame with topology comparison results, sorted by val_total_loss.
    """
    ensure_dir(output_dir)
    candidates = topology_candidates or TOPOLOGY_CANDIDATES

    device = get_device(config.runtime.device, config.runtime.cuda_fallback_to_cpu)
    set_global_seed(config.reproducibility.global_seed)

    # Load data once
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

    for topology_name, hidden_dims in candidates.items():
        logger.info(
            "Evaluating topology: %s = %s",
            topology_name,
            hidden_dims,
        )

        # Override topology in config
        topology_config = TopologyConfig(
            name=topology_name,
            hidden_dimensions=hidden_dims,
        )
        model_config = ModelConfig(
            latent_dimension=config.model.latent_dimension,
            topology=topology_config,
        )

        model = VariationalAutoencoder(model_config).to(device)
        param_count = model.count_parameters()

        topology_dir = output_dir / f"topology_{topology_name}"
        ensure_dir(topology_dir)

        checkpointer = ModelCheckpointer(
            checkpoint_dir=topology_dir / "checkpoints",
            run_id=topology_name,
        )
        history = TrainingHistory(output_dir=topology_dir)

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

        best_epoch_record = history.get_best_epoch()
        if best_epoch_record is None:
            logger.warning("No training records for topology: %s", topology_name)
            continue

        results_rows.append(
            {
                "topology_name": topology_name,
                "hidden_dimensions": str(hidden_dims),
                "total_params": param_count["total"],
                "best_epoch": best_epoch_record.epoch,
                "best_val_total_loss": best_epoch_record.val_total_loss,
                "best_val_reconstruction_loss": best_epoch_record.val_reconstruction_loss,
                "best_val_kl_divergence_loss": best_epoch_record.val_kl_divergence_loss,
                "epochs_trained": history.num_epochs_completed,
            }
        )

        history.save()

    results_df = pd.DataFrame(results_rows)
    results_df = results_df.sort_values("best_val_total_loss")

    output_csv = output_dir / "topology_selection_results.csv"
    results_df.to_csv(output_csv, index=False)
    logger.info("Topology selection results saved: %s", output_csv)

    if not results_df.empty:
        best_topology = results_df.iloc[0]["topology_name"]
        logger.info(
            "Best topology: %s (val_total_loss=%.6f)",
            best_topology,
            results_df.iloc[0]["best_val_total_loss"],
        )

    return results_df
