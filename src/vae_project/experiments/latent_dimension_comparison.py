"""Latent dimension comparison experiment.

Trains the same, previously selected topology at several latent dimensions and
generates the scientific evidence of each dimension right after its training:

    configuration -> VAE -> training -> validation -> best checkpoint
    -> checkpoint reload -> latent extraction (latent_mean) -> visualizations
    -> subsidiary metrics -> tables -> manifest -> run summary
    -> consolidation across dimensions

Every dimension is an independent training run: a new model, optimizer,
scheduler and early stopping are created and the seed is applied again. The
latent dimension is the only experimental variable; data splits, seed,
hyperparameters, topology, projection parameters and sampling are shared.

The topology is not searched here (use ``select-topology`` first). The
supervised component stays disabled by default; labels are used only after
training to color plots and compute subsidiary metrics.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import pandas as pd
import torch

from vae_project.config.loader import save_resolved_config
from vae_project.config.models import ExperimentConfig, ModelConfig, TopologyConfig
from vae_project.config.validation import validate_comparison_dimensions
from vae_project.data.datasets import get_raw_datasets
from vae_project.data.loaders import DataLoaders, create_data_loaders
from vae_project.data.splits import DataSplits, create_data_splits
from vae_project.experiments.latent_evidence import (
    EvidenceDirs,
    EvidenceResult,
    generate_latent_evidence,
)
from vae_project.experiments.topology_selection import TOPOLOGY_CANDIDATES
from vae_project.models.vae import VariationalAutoencoder
from vae_project.reporting.artifact_manifest import ArtifactManifest
from vae_project.reporting.comparison_report import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    DimensionRunRecord,
    build_comparison_table,
    write_comparison_summary,
    write_dimension_run_summary,
)
from vae_project.training.checkpointing import CheckpointMetadata, ModelCheckpointer
from vae_project.training.history import TrainingHistory
from vae_project.training.trainer import VAETrainer
from vae_project.utils.device import get_device
from vae_project.utils.logging import setup_logging
from vae_project.utils.paths import ensure_dir, ensure_run_dir_is_free, ensure_smoke_test_dir
from vae_project.utils.reproducibility import set_global_seed
from vae_project.utils.timing import Timer
from vae_project.visualization.plot_settings import apply_plot_settings

logger = logging.getLogger(__name__)

ANALYZED_SPLIT = "val"


class LatentDimensionComparisonError(Exception):
    """Raised when the comparison cannot start or no dimension succeeded."""


def resolve_topology(config: ExperimentConfig) -> TopologyConfig:
    """Return the fixed topology used by every model of the comparison.

    The topology comes from ``model.topology``. ``fixed_topology`` must name it,
    which guards against a config whose fixed topology and hidden dimensions
    disagree. No topology search is performed here.

    Args:
        config: Experiment configuration.

    Returns:
        TopologyConfig shared by all dimensions.

    Raises:
        LatentDimensionComparisonError: If the names do not match, or the
            standard name does not match the standard hidden dimensions.
    """
    fixed_name = config.latent_dimension_comparison.fixed_topology
    topology = config.model.topology
    if topology.name != fixed_name:
        raise LatentDimensionComparisonError(
            f"latent_dimension_comparison.fixed_topology='{fixed_name}' does not match "
            f"model.topology.name='{topology.name}'. The comparison uses one fixed topology."
        )
    standard = TOPOLOGY_CANDIDATES.get(fixed_name)
    if standard is not None and list(topology.hidden_dimensions) != standard:
        logger.warning(
            "Topology '%s' uses hidden_dimensions=%s, which differs from the project "
            "standard %s. Proceeding with the configured values.",
            fixed_name,
            topology.hidden_dimensions,
            standard,
        )
    return TopologyConfig(name=topology.name, hidden_dimensions=list(topology.hidden_dimensions))


def _relative(path: Path | None, base: Path) -> str | None:
    """Return ``path`` relative to ``base`` using forward slashes."""
    if path is None:
        return None
    return Path(path).relative_to(base).as_posix()


def _first_relative(paths: list[Path], suffix: str, base: Path) -> str | None:
    """Return the first path with ``suffix`` relative to ``base``, or None."""
    for path in paths:
        if path.suffix == suffix:
            return _relative(path, base)
    return None


def _write_json(path: Path, data: object) -> None:
    """Write ``data`` as UTF-8 JSON."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


class _DimensionRunner:
    """Trains and analyzes one latent dimension with shared data and device."""

    def __init__(
        self,
        config: ExperimentConfig,
        topology: TopologyConfig,
        output_dir: Path,
        device: torch.device,
        splits: DataSplits,
        loaders: DataLoaders,
        synthetic_data: bool = False,
    ) -> None:
        self.synthetic_data = synthetic_data
        self.config = config
        self.topology = topology
        self.output_dir = output_dir
        self.device = device
        self.splits = splits
        self.loaders = loaders

    def run(self, latent_dim: int) -> DimensionRunRecord:
        """Run the complete flow for ``latent_dim`` and return its record."""
        config = self.config
        dim_dir = ensure_dir(self.output_dir / f"latent_dim_{latent_dim}")
        log_path = dim_dir / "logs" / "experiment.log"
        record = DimensionRunRecord(
            dataset=config.dataset.name.value,
            experiment_type=config.experiment_type,
            topology_name=self.topology.name,
            hidden_dimensions=list(self.topology.hidden_dimensions),
            latent_dimension=latent_dim,
            seed=config.reproducibility.global_seed,
            split=ANALYZED_SPLIT,
            device=str(self.device),
            quick_mode=config.project.quick_mode,
            synthetic_data=self.synthetic_data,
        )
        record.paths["log"] = _relative(log_path, self.output_dir)
        log_handler = self._attach_run_log(log_path)
        try:
            self._train_and_analyze(latent_dim, dim_dir, record)
        except Exception as exc:
            logger.error(
                "Latent dimension %d failed (dataset=%s, operation=training/analysis): %s",
                latent_dim,
                record.dataset,
                exc,
                exc_info=True,
            )
            record.status = STATUS_FAILED
            record.failures.append(
                {
                    "operation": "training_or_checkpoint",
                    "error": f"{type(exc).__name__}: {exc}",
                    "probable_cause": "training, checkpoint loading or latent extraction "
                    "raised an exception (see the log).",
                }
            )
            record.warnings.append(
                f"Partial artifacts were preserved. Traceback: {traceback.format_exc(limit=3)!r}"
            )
        finally:
            self._finalize(record, dim_dir)
            if log_handler is not None:
                logging.getLogger("vae_project").removeHandler(log_handler)
                log_handler.close()
        return record

    @staticmethod
    def _attach_run_log(log_path: Path) -> logging.Handler | None:
        """Attach a per-dimension file handler to the project logger."""
        ensure_dir(log_path.parent)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        project_logger = logging.getLogger("vae_project")
        project_logger.setLevel(logging.DEBUG)
        project_logger.addHandler(handler)
        return handler

    def _train_and_analyze(
        self, latent_dim: int, dim_dir: Path, record: DimensionRunRecord
    ) -> None:
        """Train, reload the best checkpoint and generate all evidence."""
        config = self.config
        logger.info(
            "Training latent_dim=%d (topology=%s, seed=%d, dataset=%s)",
            latent_dim,
            self.topology.name,
            config.reproducibility.global_seed,
            record.dataset,
        )

        # Seed first: it governs weight initialization of the new model.
        set_global_seed(
            config.reproducibility.global_seed,
            deterministic=config.reproducibility.deterministic_mode,
        )

        # 1. Fresh model; the trainer recreates optimizer, scheduler and early stopping.
        model = VariationalAutoencoder(
            ModelConfig(
                latent_dimension=latent_dim,
                topology=self.topology,
                input_dimension=config.model.input_dimension,
                output_activation=config.model.output_activation,
            )
        ).to(self.device)
        record.parameter_count = model.count_parameters()["total"]

        checkpointer = ModelCheckpointer(
            checkpoint_dir=dim_dir / "checkpoints",
            run_id=f"latent_dim_{latent_dim}",
        )
        tables_dir = ensure_dir(dim_dir / "tables")
        history = TrainingHistory(output_dir=tables_dir)

        apply_plot_settings(config.plotting)
        trainer = VAETrainer(
            model=model,
            config=config,
            device=self.device,
            checkpointer=checkpointer,
            history=history,
        )
        with Timer() as train_timer:
            trainer.fit(
                train_loader=self.loaders.train,
                val_loader=self.loaders.val,
                dataset_name=config.dataset.name.value,
                is_quick_mode=config.project.quick_mode,
            )
        history.save()
        record.training_time_seconds = train_timer.elapsed_seconds
        record.epochs_trained = history.num_epochs_completed
        record.tables.append("tables/training_history.csv")

        if history.get_best_epoch() is None:
            raise LatentDimensionComparisonError("training produced no epoch records")

        # 2. Reload the best checkpoint into a fresh model of the same dimension
        #    so the analysis never uses the last-epoch weights.
        best_model = VariationalAutoencoder(
            ModelConfig(
                latent_dimension=latent_dim,
                topology=self.topology,
                input_dimension=config.model.input_dimension,
                output_activation=config.model.output_activation,
            )
        ).to(self.device)
        metadata = checkpointer.load_best(model=best_model, device=self.device)
        best_model.eval()
        self._fill_training_results(record, metadata, checkpointer)

        # 3. Evidence from the reloaded best model.
        evidence = generate_latent_evidence(
            model=best_model,
            data_loader=self.loaders.val,
            device=self.device,
            class_names=self.splits.class_names,
            config=config,
            dirs=EvidenceDirs(
                figures_latent=dim_dir / "figures" / "latent_space",
                latent_data=dim_dir / "latent_data",
                tables=tables_dir,
            ),
            split_name=ANALYZED_SPLIT,
            run_id=f"latent_dim_{latent_dim}",
            raise_on_failure=False,
        )
        self._fill_evidence_results(record, evidence, dim_dir)
        record.status = STATUS_COMPLETED if evidence.is_complete else STATUS_PARTIAL

    def _fill_training_results(
        self,
        record: DimensionRunRecord,
        metadata: CheckpointMetadata,
        checkpointer: ModelCheckpointer,
    ) -> None:
        """Copy best-checkpoint values (validation split) into the record."""
        record.best_epoch = metadata.epoch
        record.val_reconstruction_loss = metadata.val_reconstruction_loss
        record.val_kl_divergence = metadata.val_kl_divergence_loss
        record.val_classification_loss = metadata.val_classification_loss
        record.val_total_loss = metadata.val_total_loss
        record.paths["checkpoint"] = _relative(checkpointer.best_checkpoint_path, self.output_dir)
        record.tables.append("checkpoints/checkpoint_metadata.json")

    def _fill_evidence_results(
        self,
        record: DimensionRunRecord,
        evidence: EvidenceResult,
        dim_dir: Path,
    ) -> None:
        """Copy evidence outputs and failures into the record."""
        base = self.output_dir
        record.visualization_mode = evidence.visualization_mode
        record.n_samples_total = evidence.n_samples_total
        record.n_projection_samples = evidence.n_projection_samples
        record.used_sampling = evidence.used_sampling
        record.pca_explained_variance = evidence.pca_explained_variance
        record.projection_metrics = list(evidence.metrics)
        record.failures.extend(failure.as_dict() for failure in evidence.failures)

        record.paths["latent_representations"] = _first_relative(
            evidence.latent_paths, ".csv", base
        )
        record.paths["projection_metrics"] = _relative(evidence.projection_metrics_path, base)
        for kind, key in (
            ("direct", "direct_plot"),
            ("pca", "pca_plot"),
            ("umap", "umap_plot"),
            ("tsne", "tsne_plot"),
            ("pca_explained_variance", "pca_explained_variance_plot"),
        ):
            record.paths[key] = _first_relative(evidence.figure_paths(kind), ".png", base)

        for path in evidence.all_files():
            relative_to_dim = path.relative_to(dim_dir).as_posix()
            if path.suffix in (".png", ".svg"):
                record.figures.append(relative_to_dim)
            else:
                record.tables.append(relative_to_dim)

    def _finalize(self, record: DimensionRunRecord, dim_dir: Path) -> None:
        """Write config, metrics, summary and manifest, whatever the status."""
        config = self.config
        base = self.output_dir
        try:
            if config.artifacts.save_resolved_config:
                dim_config = _config_for_dimension(config, self.topology, record.latent_dimension)
                save_resolved_config(dim_config, dim_dir / "config_resolved.yaml")

            _write_json(
                dim_dir / "metrics.json",
                {
                    "dataset": record.dataset,
                    "data_origin": record.data_origin,
                    "experiment_type": record.experiment_type,
                    "latent_dimension": record.latent_dimension,
                    "seed": record.seed,
                    "split": record.split,
                    "execution_status": record.status,
                    "best_epoch": record.best_epoch,
                    "validation_reconstruction_loss": record.val_reconstruction_loss,
                    "validation_kl_divergence": record.val_kl_divergence,
                    "validation_classification_loss": record.val_classification_loss,
                    "validation_total_loss": record.val_total_loss,
                    "parameter_count": record.parameter_count,
                    "training_time_seconds": record.training_time_seconds,
                    "pca_explained_variance": (
                        None
                        if record.pca_explained_variance is None
                        else {
                            "component_1": record.pca_explained_variance[0],
                            "component_2": record.pca_explained_variance[1],
                            "cumulative_2_components": record.pca_explained_variance[2],
                        }
                    ),
                    "n_samples_total": record.n_samples_total,
                    "n_projection_samples": record.n_projection_samples,
                    "projection_metrics": record.projection_metrics,
                    "errors": record.failures,
                },
            )

            summary_path = write_dimension_run_summary(record, dim_dir)
            record.paths["run_summary"] = _relative(summary_path, base)
            manifest_path = dim_dir / "artifact_manifest.json"
            record.paths["artifact_manifest"] = _relative(manifest_path, base)

            manifest = ArtifactManifest(
                dim_dir,
                context={
                    "dataset": record.dataset,
                    "data_origin": record.data_origin,
                    "experiment_type": record.experiment_type,
                    "latent_dimension": record.latent_dimension,
                    "seed": record.seed,
                },
            )
            manifest.set_execution_status(record.status, record.failures)
            manifest.register_run_dir_scan(compute_hash=True)
            manifest.save()
        except Exception as exc:  # reporting must never hide the training outcome
            logger.error(
                "Could not write reports for latent_dim=%d (%s): %s",
                record.latent_dimension,
                dim_dir,
                exc,
                exc_info=True,
            )
            record.failures.append(
                {
                    "operation": "reporting",
                    "error": f"{type(exc).__name__}: {exc}",
                    "probable_cause": "the output directory is not writable.",
                }
            )
            if record.status == STATUS_COMPLETED:
                record.status = STATUS_PARTIAL


def _config_for_dimension(
    config: ExperimentConfig,
    topology: TopologyConfig,
    latent_dim: int,
) -> ExperimentConfig:
    """Return a copy of ``config`` whose single model uses ``latent_dim``."""
    import copy

    dim_config = copy.deepcopy(config)
    dim_config.model = ModelConfig(
        latent_dimension=latent_dim,
        topology=topology,
        input_dimension=config.model.input_dimension,
        output_activation=config.model.output_activation,
    )
    return dim_config


def run_latent_dimension_comparison(
    config: ExperimentConfig,
    output_dir: Path,
    dimensions: list[int] | None = None,
    splits: DataSplits | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Train each latent dimension independently and generate its evidence.

    Holds constant: dataset, topology, seed, learning rate, batch size,
    max epochs, early stopping, loss definition, data splits, projection
    parameters and sampling.

    A failure in one dimension is recorded and does not stop the others; failed
    dimensions never take part in the best-dimension statement.

    ``output_dir`` is the directory of *this run*. It must be new or empty: an
    existing directory that already holds artifacts is rejected unless
    ``overwrite`` is True, so a finished run is never replaced silently.

    Args:
        config: Base experiment configuration.
        output_dir: Run directory for comparison artifacts.
        dimensions: Latent dimensions to compare. Defaults to
            ``config.latent_dimension_comparison.dimensions``.
        splits: Optional pre-built data splits. They are accepted **only for
            synthetic data** (tests and smoke tests, no downloads): the run is
            then labelled synthetic in every report and ``output_dir`` must
            contain ``smoke_test`` or ``synthetic`` in its path. When None, the
            configured real dataset is loaded.
        overwrite: Allow writing into a non-empty ``output_dir``. Files with the
            same name are replaced; nothing else is removed.

    Returns:
        Consolidated comparison table sorted by ``latent_dimension``.

    Raises:
        LatentDimensionComparisonError: If the comparison is disabled, the
            dimensions are invalid, the topology is inconsistent, or no
            dimension finished training.
        ArtifactError: If ``output_dir`` already holds artifacts and
            ``overwrite`` is False, or a synthetic run targets a directory
            without a ``smoke_test``/``synthetic`` marker.
    """
    if not config.latent_dimension_comparison.enabled:
        raise LatentDimensionComparisonError(
            "latent_dimension_comparison.enabled is false; nothing to run."
        )

    compare_dims = list(
        dimensions if dimensions is not None else config.latent_dimension_comparison.dimensions
    )
    dimension_errors = validate_comparison_dimensions(compare_dims)
    if dimension_errors:
        raise LatentDimensionComparisonError("; ".join(dimension_errors))

    topology = resolve_topology(config)
    synthetic_data = splits is not None
    if synthetic_data:
        ensure_smoke_test_dir(Path(output_dir))
    ensure_run_dir_is_free(Path(output_dir), overwrite=overwrite)
    output_dir = ensure_dir(Path(output_dir))
    setup_logging(config.logging, log_file=output_dir / "latent_dimension_comparison.log")

    device = get_device(config.runtime.device, config.runtime.cuda_fallback_to_cpu)

    if splits is None:
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
    # One split and one set of loaders for all dimensions.
    loaders = create_data_loaders(
        splits=splits,
        training_config=config.training,
        dataset_config=config.dataset,
    )

    runner = _DimensionRunner(
        config, topology, output_dir, device, splits, loaders, synthetic_data=synthetic_data
    )
    records: list[DimensionRunRecord] = []
    for latent_dim in compare_dims:
        record = runner.run(latent_dim)
        records.append(record)
        logger.info("latent_dim=%d finished with status=%s", latent_dim, record.status)

    table = build_comparison_table(records)
    output_csv = output_dir / "latent_dimension_comparison.csv"
    write_comparison_summary(records, output_dir)
    table.to_csv(output_csv, index=False)
    logger.info("Latent dimension comparison saved: %s", output_csv)

    if all(record.status == STATUS_FAILED for record in records):
        raise LatentDimensionComparisonError(
            f"All latent dimensions failed for dataset={config.dataset.name.value}. "
            f"See {output_dir / 'latent_dimension_comparison_summary.md'} and the "
            "per-dimension logs."
        )
    return table
