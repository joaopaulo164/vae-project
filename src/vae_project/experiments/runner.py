"""Individual experiment runner: one topology, one latent dimension, one mode.

The runner is the technical execution service behind ``run-pipeline`` and
``train``. It resolves the run directory of the configured experiment and
delegates the whole flow (training, best-checkpoint evidence, tables, summary
and manifest) to
:func:`vae_project.experiments.single_experiment.execute_single_experiment`,
the same service used by the specialized comparisons and by the factorial
protocol.

Usage::

    runner = ExperimentRunner(config)
    run_directory = runner.run()
"""

from __future__ import annotations

import logging
from pathlib import Path

from vae_project.config.models import ExperimentConfig, TopologyConfig
from vae_project.experiments.experiment_result import EvidenceProfile, SingleExperimentResult
from vae_project.experiments.single_experiment import (
    SingleExperimentOptions,
    execute_single_experiment,
    prepare_experiment_data,
)
from vae_project.utils.logging import setup_logging
from vae_project.utils.paths import build_run_dir

logger = logging.getLogger(__name__)

LATENT_TRAVERSAL_DIMENSION = 2


class ExperimentRunner:
    """Run exactly one configured experiment through the common service.

    Args:
        config: Fully validated ExperimentConfig.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.result: SingleExperimentResult | None = None

    def build_output_directory(self) -> Path:
        """Return the new run directory of the configured experiment."""
        return build_run_dir(
            artifacts_dir=self.config.paths.artifacts_dir,
            dataset_name=self.config.dataset.name.value,
            experiment_type=self.config.experiment_type,
            latent_dimension=self.config.model.latent_dimension,
            run_name=self.config.run_name,
        )

    def run(self) -> Path:
        """Execute the full experiment pipeline.

        Returns:
            Path to the run directory containing all artifacts.

        Raises:
            Exception: The original training or evidence error, after the
                reports of the failed execution were written.
        """
        setup_logging(self.config.logging)
        output_directory = self.build_output_directory()
        logger.info("Experiment started: %s", output_directory)

        topology = self.config.model.topology
        latent_dimension = self.config.model.latent_dimension
        data = prepare_experiment_data(self.config)
        self.result = execute_single_experiment(
            config=self.config,
            topology=TopologyConfig(
                name=topology.name, hidden_dimensions=list(topology.hidden_dimensions)
            ),
            latent_dimension=latent_dimension,
            supervised=self.config.supervised_component.enabled,
            evidence_profile=EvidenceProfile.FULL,
            output_directory=output_directory,
            data=data,
            options=SingleExperimentOptions(
                write_environment=True,
                include_latent_traversal=latent_dimension == LATENT_TRAVERSAL_DIMENSION,
                raise_on_failure=True,
            ),
        )
        logger.info("Artifacts at: %s", output_directory)
        return output_directory
