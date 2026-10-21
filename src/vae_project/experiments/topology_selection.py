"""Topology selection experiment (specialized pipeline).

Trains each candidate topology at the configured latent dimension and selects
the one with the lowest validation total loss. Uses only training and
validation sets. Each candidate is one call of the common service
:func:`vae_project.experiments.single_experiment.execute_single_experiment`
with the ``TRAINING_ONLY`` evidence profile (training curves, no latent
analysis).

This pipeline remains available for isolated analyses; the factorial protocol
does not depend on it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from vae_project.config.models import STANDARD_TOPOLOGIES, ExperimentConfig, TopologyConfig
from vae_project.data.splits import DataSplits
from vae_project.experiments.experiment_result import (
    STATUS_FAILED,
    EvidenceProfile,
    SingleExperimentResult,
)
from vae_project.experiments.single_experiment import (
    SingleExperimentOptions,
    execute_single_experiment,
    prepare_experiment_data,
)
from vae_project.utils.paths import ensure_dir

logger = logging.getLogger(__name__)

# Standard topology candidates (shared with the factorial protocol).
TOPOLOGY_CANDIDATES: dict[str, list[int]] = {
    name: list(hidden) for name, hidden in STANDARD_TOPOLOGIES.items()
}

RESULTS_FILENAME = "topology_selection_results.csv"


def _result_row(result: SingleExperimentResult) -> dict[str, object]:
    """Build one row of the topology selection table from a service result."""
    return {
        "topology_name": result.topology_name,
        "hidden_dimensions": str(result.hidden_dimensions),
        "total_params": result.parameter_count,
        "best_epoch": result.best_epoch,
        "best_val_total_loss": result.validation_total_loss,
        "best_val_reconstruction_loss": result.validation_reconstruction_loss,
        "best_val_kl_divergence_loss": result.validation_kl_divergence,
        "epochs_trained": result.epochs_trained,
    }


def run_topology_selection(
    config: ExperimentConfig,
    output_dir: Path,
    topology_candidates: dict[str, list[int]] | None = None,
    splits: DataSplits | None = None,
) -> pd.DataFrame:
    """Run the topology selection experiment.

    Trains each candidate topology independently and returns a comparison
    table. The best topology is the one with the lowest validation total loss.
    A candidate that fails is logged and left out of the table.

    Args:
        config: Base experiment configuration (topology will be overridden).
        output_dir: Directory for topology selection artifacts.
        topology_candidates: Dictionary of topology_name -> hidden_dimensions.
            Defaults to TOPOLOGY_CANDIDATES.
        splits: Optional pre-built synthetic splits (tests and smoke tests).

    Returns:
        DataFrame with topology comparison results, sorted by val_total_loss.
    """
    ensure_dir(output_dir)
    candidates = topology_candidates or TOPOLOGY_CANDIDATES
    data = prepare_experiment_data(config, splits)

    results_rows: list[dict[str, object]] = []
    for topology_name, hidden_dims in candidates.items():
        logger.info("Evaluating topology: %s = %s", topology_name, hidden_dims)
        result = execute_single_experiment(
            config=config,
            topology=TopologyConfig(name=topology_name, hidden_dimensions=list(hidden_dims)),
            latent_dimension=config.model.latent_dimension,
            supervised=False,
            evidence_profile=EvidenceProfile.TRAINING_ONLY,
            output_directory=output_dir / f"topology_{topology_name}",
            data=data,
            options=SingleExperimentOptions(run_id=topology_name),
        )
        if result.status == STATUS_FAILED or result.validation_total_loss is None:
            logger.warning("No valid training result for topology: %s", topology_name)
            continue
        results_rows.append(_result_row(result))

    results_df = pd.DataFrame(results_rows)
    if not results_df.empty:
        results_df = results_df.sort_values("best_val_total_loss")

    output_csv = output_dir / RESULTS_FILENAME
    results_df.to_csv(output_csv, index=False)
    logger.info("Topology selection results saved: %s", output_csv)

    if not results_df.empty:
        logger.info(
            "Best topology: %s (val_total_loss=%.6f)",
            results_df.iloc[0]["topology_name"],
            results_df.iloc[0]["best_val_total_loss"],
        )

    return results_df
