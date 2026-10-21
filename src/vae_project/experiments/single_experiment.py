"""Common service that executes one complete, single-configuration experiment.

Every pipeline of the project (individual run, topology selection, latent
dimension comparison, factorial grid, supervised extension, smoke tests) calls
:func:`execute_single_experiment`. The flow is always the same:

    validate -> seed -> device -> data -> model -> optimizer/scheduler/early
    stopping -> train -> validate -> best checkpoint -> reload best checkpoint
    -> training curves -> reconstructions -> prior samples -> latent extraction
    -> direct 2D plot or PCA/UMAP/t-SNE -> subsidiary metrics -> tables
    -> resolved config -> environment -> run_summary.md -> artifact manifest
    -> typed :class:`SingleExperimentResult`

Each call builds a new model, optimizer, scheduler and early stopping and
applies the seed again, so executions never share state. A failure is
recorded in the result and in the written reports; artifacts produced before
the failure are preserved.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
import torch

from vae_project.config.loader import save_resolved_config
from vae_project.config.models import ExperimentConfig, ModelConfig, TopologyConfig
from vae_project.config.validation import validate_config
from vae_project.data.datasets import get_raw_datasets
from vae_project.data.loaders import DataLoaders, create_data_loaders
from vae_project.data.splits import DataSplits, create_data_splits, save_class_distribution
from vae_project.experiments.experiment_result import (
    FIGURE_CATEGORY_TRAINING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    EvidenceProfile,
    SingleExperimentResult,
    relative_posix,
)
from vae_project.experiments.model_evidence import (
    DEFAULT_GENERATED_SAMPLES,
    EvidenceBundle,
    EvidenceDirectories,
    EvidenceRequest,
    collect_model_evidence,
    record_step,
)
from vae_project.models.supervised_head import SupervisedClassificationHead
from vae_project.models.vae import VariationalAutoencoder
from vae_project.reporting.artifact_manifest import ArtifactManifest
from vae_project.reporting.experiment_summary import write_experiment_run_summary
from vae_project.training.checkpointing import CheckpointMetadata, ModelCheckpointer
from vae_project.training.history import TrainingHistory
from vae_project.training.trainer import VAETrainer
from vae_project.utils.device import get_device
from vae_project.utils.environment import collect_environment_info
from vae_project.utils.json_io import write_json
from vae_project.utils.paths import ensure_dir, ensure_run_subdirectories
from vae_project.utils.reproducibility import set_global_seed
from vae_project.utils.timing import Timer
from vae_project.visualization.plot_settings import apply_plot_settings
from vae_project.visualization.training_plots import (
    plot_training_loss_components,
    plot_training_total_loss,
)

logger = logging.getLogger(__name__)

VALIDATION_SPLIT = "val"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
ENVIRONMENT_FILENAME = "environment.json"


class SingleExperimentError(Exception):
    """Raised when a single experiment cannot be configured or executed."""


@dataclass
class ExperimentData:
    """Data splits, loaders and device shared by executions of one dataset.

    Sharing them across a grid guarantees identical splits and batches for
    every combination.

    Attributes:
        splits: Train/validation/test splits.
        loaders: Loaders created from ``splits``.
        device: Compute device selected by the runtime policy.
        synthetic: True when the splits were injected (tests, smoke tests).
    """

    splits: DataSplits
    loaders: DataLoaders
    device: torch.device
    synthetic: bool = False


@dataclass
class SingleExperimentOptions:
    """Optional behaviour of one execution.

    Attributes:
        run_id: Execution identifier (default: output directory name).
        reconstruction_indices: Shared positions of the reconstructed images.
        generation_seed: Seed of the prior samples (default: global seed).
        num_generated_samples: Number of prior samples.
        include_latent_traversal: Also draw the 2D decoder manifold.
        write_environment: Write ``environment.json`` into the run directory.
        environment_reference: Relative path of a shared environment file,
            recorded in the summary instead of writing a new one.
        raise_on_failure: Re-raise the original error after the reports of the
            failed execution were written. Used by the individual pipeline.
        extra_context: Constant provenance values for latent data and metrics.
    """

    run_id: str | None = None
    reconstruction_indices: list[int] | None = None
    generation_seed: int | None = None
    num_generated_samples: int = DEFAULT_GENERATED_SAMPLES
    include_latent_traversal: bool = False
    write_environment: bool = False
    environment_reference: str | None = None
    raise_on_failure: bool = False
    extra_context: dict[str, object] | None = None


def prepare_experiment_data(
    config: ExperimentConfig,
    splits: DataSplits | None = None,
) -> ExperimentData:
    """Select the device and build the splits and loaders of a dataset.

    Args:
        config: Experiment configuration.
        splits: Pre-built splits. Accepted only for synthetic data (tests and
            smoke tests); when None the configured real dataset is loaded.

    Returns:
        ExperimentData reused by every execution of the dataset.
    """
    device = get_device(config.runtime.device, config.runtime.cuda_fallback_to_cpu)
    synthetic = splits is not None
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
    loaders = create_data_loaders(
        splits=splits,
        training_config=config.training,
        dataset_config=config.dataset,
    )
    return ExperimentData(splits=splits, loaders=loaders, device=device, synthetic=synthetic)


def derive_run_config(
    config: ExperimentConfig,
    topology: TopologyConfig,
    latent_dimension: int,
    supervised: bool,
) -> ExperimentConfig:
    """Return a copy of ``config`` for exactly one model.

    Only the topology, the latent dimension and the supervised switch are
    changed (and the protocol grid is switched off, because the copy describes
    a single model); every other value is inherited unchanged.

    Args:
        config: Base configuration (never mutated).
        topology: Topology of the model.
        latent_dimension: Latent dimension of the model.
        supervised: Whether the supervised component is enabled.

    Returns:
        Independent ExperimentConfig describing a single model.
    """
    run_config = copy.deepcopy(config)
    run_config.model = ModelConfig(
        latent_dimension=latent_dimension,
        topology=TopologyConfig(
            name=topology.name, hidden_dimensions=list(topology.hidden_dimensions)
        ),
        input_dimension=config.model.input_dimension,
        output_activation=config.model.output_activation,
    )
    run_config.supervised_component.enabled = supervised
    # A derived config describes one model, not a protocol grid.
    run_config.factorial_comparison.enabled = False
    return run_config


def attach_run_log(log_path: Path) -> logging.Handler:
    """Attach a file handler that records this execution in its own log."""
    ensure_dir(log_path.parent)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    project_logger = logging.getLogger("vae_project")
    project_logger.setLevel(logging.DEBUG)
    project_logger.addHandler(handler)
    return handler


def detach_run_log(handler: logging.Handler) -> None:
    """Remove and close a per-execution file handler."""
    logging.getLogger("vae_project").removeHandler(handler)
    handler.close()


def _new_result(
    run_config: ExperimentConfig,
    data: ExperimentData,
    run_dir: Path,
    options: SingleExperimentOptions,
) -> SingleExperimentResult:
    """Create the result object with the identity of the execution."""
    return SingleExperimentResult(
        dataset=run_config.dataset.name.value,
        experiment_type=run_config.experiment_type,
        topology_name=run_config.model.topology.name,
        hidden_dimensions=list(run_config.model.topology.hidden_dimensions),
        latent_dimension=run_config.model.latent_dimension,
        seed=run_config.reproducibility.global_seed,
        split=VALIDATION_SPLIT,
        device=str(data.device),
        quick_mode=run_config.project.quick_mode,
        synthetic_data=data.synthetic,
        execution_id=options.run_id or run_dir.name,
        run_directory=run_dir,
        log_path=run_dir / "logs" / "experiment.log",
        environment_reference=options.environment_reference,
    )


def _build_vae(run_config: ExperimentConfig, device: torch.device) -> VariationalAutoencoder:
    """Build a new, randomly initialized VAE for ``run_config.model``."""
    return VariationalAutoencoder(run_config.model).to(device)


def _build_head(
    run_config: ExperimentConfig,
    num_classes: int,
    device: torch.device,
) -> SupervisedClassificationHead:
    """Build a new classification head for the supervised component."""
    return SupervisedClassificationHead(
        config=run_config.supervised_component,
        latent_dimension=run_config.model.latent_dimension,
        num_classes=num_classes,
    ).to(device)


def _record_training_outcome(
    result: SingleExperimentResult,
    metadata: CheckpointMetadata,
    checkpointer: ModelCheckpointer,
) -> None:
    """Copy the best-checkpoint values (validation split) into the result."""
    result.best_epoch = metadata.epoch
    result.validation_reconstruction_loss = metadata.val_reconstruction_loss
    result.validation_kl_divergence = metadata.val_kl_divergence_loss
    result.validation_total_loss = metadata.val_total_loss
    if metadata.experiment_type == "supervised":
        result.validation_classification_loss = metadata.val_classification_loss
    result.best_checkpoint_path = checkpointer.best_checkpoint_path
    result.table_paths.append(checkpointer.checkpoint_dir / checkpointer.METADATA_FILENAME)


def _plot_training_curves(
    result: SingleExperimentResult,
    history: TrainingHistory,
    run_config: ExperimentConfig,
    figures_dir: Path,
) -> None:
    """Draw the training curves from the real history of this execution."""
    history_frame = pd.DataFrame([vars(record) for record in history.get_records()])
    plot_functions: dict[str, Callable[..., list[Path]]] = {
        "training_total_loss_plot": plot_training_total_loss,
        "training_loss_components_plot": plot_training_loss_components,
    }
    for operation, plot_function in plot_functions.items():
        paths = record_step(
            result.errors,
            operation,
            "matplotlib failed to render or write the figure (disk space or permissions).",
            partial(
                plot_function,
                history_df=history_frame,
                output_dir=figures_dir,
                config=run_config.plotting,
                dataset_name=run_config.dataset.name.value,
                is_quick_mode=run_config.project.quick_mode,
            ),
        )
        if paths:
            result.figure_paths.setdefault(FIGURE_CATEGORY_TRAINING, []).extend(paths)


def _train(
    run_config: ExperimentConfig,
    data: ExperimentData,
    subdirs: dict[str, Path],
    result: SingleExperimentResult,
) -> tuple[ModelCheckpointer, TrainingHistory]:
    """Seed, build every training component, train and save the history."""
    set_global_seed(
        run_config.reproducibility.global_seed,
        deterministic=run_config.reproducibility.deterministic_mode,
    )
    model = _build_vae(run_config, data.device)
    result.parameter_count = model.count_parameters()["total"]
    head = (
        _build_head(run_config, data.splits.num_classes, data.device)
        if run_config.supervised_component.enabled
        else None
    )
    checkpointer = ModelCheckpointer(
        checkpoint_dir=subdirs["checkpoints"], run_id=result.execution_id
    )
    history = TrainingHistory(output_dir=subdirs["tables"])
    apply_plot_settings(run_config.plotting)

    logger.info(
        "Training: dataset=%s, topology=%s, latent_dim=%d, supervised=%s, seed=%d",
        result.dataset,
        result.topology_name,
        result.latent_dimension,
        run_config.supervised_component.enabled,
        result.seed,
    )
    trainer = VAETrainer(
        model=model,
        config=run_config,
        device=data.device,
        checkpointer=checkpointer,
        history=history,
        supervised_head=head,
    )
    with Timer() as train_timer:
        trainer.fit(
            train_loader=data.loaders.train,
            val_loader=data.loaders.val,
            dataset_name=result.dataset,
            is_quick_mode=run_config.project.quick_mode,
        )
    result.training_duration_seconds = train_timer.elapsed_seconds
    csv_path, json_path = history.save()
    result.epochs_trained = history.num_epochs_completed
    result.table_paths.extend([csv_path, json_path])
    if history.get_best_epoch() is None:
        raise SingleExperimentError("training produced no epoch records")
    return checkpointer, history


def _reload_best_model(
    run_config: ExperimentConfig,
    data: ExperimentData,
    checkpointer: ModelCheckpointer,
    result: SingleExperimentResult,
) -> VariationalAutoencoder:
    """Load the best checkpoint into a fresh model (never the last-epoch weights)."""
    best_model = _build_vae(run_config, data.device)
    metadata = checkpointer.load_best(model=best_model, device=data.device)
    best_model.eval()
    if run_config.supervised_component.enabled:
        best_head = _build_head(run_config, data.splits.num_classes, data.device)
        checkpointer.load_supervised_head(best_head, data.device)
        best_head.eval()
    _record_training_outcome(result, metadata, checkpointer)
    logger.info(
        "Best checkpoint reloaded: epoch=%d, val_total_loss=%.6f",
        metadata.epoch,
        metadata.val_total_loss,
    )
    return best_model


def apply_evidence_bundle(result: SingleExperimentResult, bundle: EvidenceBundle) -> None:
    """Copy produced evidence files, metrics and failures into the result."""
    for category, paths in bundle.figure_paths.items():
        result.figure_paths.setdefault(category, []).extend(paths)
    result.errors.extend(bundle.failures)
    result.table_paths.extend(bundle.data_files())
    latent = bundle.latent
    if latent is None:
        return
    result.visualization_mode = latent.visualization_mode
    result.n_samples_total = latent.n_samples_total
    result.n_projection_samples = latent.n_projection_samples
    result.used_sampling = latent.used_sampling
    result.pca_explained_variance = latent.pca_explained_variance
    result.projection_metrics = list(latent.metrics)
    result.latent_figure_paths = {kind: list(paths) for kind, paths in latent.figures.items()}


def _collect_evidence(
    run_config: ExperimentConfig,
    data: ExperimentData,
    subdirs: dict[str, Path],
    best_model: VariationalAutoencoder,
    options: SingleExperimentOptions,
    result: SingleExperimentResult,
) -> None:
    """Generate reconstructions, prior samples and latent-space evidence."""
    request = EvidenceRequest(
        model=best_model,
        data_loader=data.loaders.val,
        device=data.device,
        class_names=data.splits.class_names,
        config=run_config,
        split_name=VALIDATION_SPLIT,
        run_id=result.execution_id,
        directories=EvidenceDirectories(
            reconstructions=subdirs["figures_reconstructions"],
            generated_samples=subdirs["figures_generated"],
            latent_space=subdirs["figures_latent"],
            latent_data=subdirs["latent_data"],
            tables=subdirs["tables"],
        ),
        reconstruction_indices=options.reconstruction_indices,
        generation_seed=options.generation_seed,
        num_generated_samples=options.num_generated_samples,
        include_traversal=options.include_latent_traversal,
        extra_context=dict(options.extra_context or {}),
        raise_on_failure=options.raise_on_failure,
    )
    apply_evidence_bundle(result, collect_model_evidence(request))


def _record_failure(result: SingleExperimentResult, error: Exception) -> None:
    """Record a failed training/checkpoint step without hiding it."""
    logger.error(
        "Execution failed (dataset=%s, topology=%s, latent_dimension=%d): %s",
        result.dataset,
        result.topology_name,
        result.latent_dimension,
        error,
        exc_info=True,
    )
    result.status = STATUS_FAILED
    result.errors.append(
        {
            "operation": "training_or_checkpoint",
            "error": f"{type(error).__name__}: {error}",
            "probable_cause": "training, checkpoint loading or evidence generation raised "
            "an exception (see the log).",
        }
    )
    result.warnings.append("Partial artifacts were preserved; see the log for the traceback.")


def _metrics_payload(result: SingleExperimentResult, run_dir: Path) -> dict[str, object]:
    """Build the ``metrics.json`` content from the typed result."""
    variance = result.pca_explained_variance
    return {
        "dataset": result.dataset,
        "data_origin": result.data_origin,
        "experiment_type": result.experiment_type,
        "topology_name": result.topology_name,
        "hidden_dimensions": result.hidden_dimensions,
        "latent_dimension": result.latent_dimension,
        "seed": result.seed,
        "split": result.split,
        "execution_id": result.execution_id,
        "execution_status": result.status,
        "best_epoch": result.best_epoch,
        "epochs_trained": result.epochs_trained,
        "validation_reconstruction_loss": result.validation_reconstruction_loss,
        "validation_kl_divergence": result.validation_kl_divergence,
        "validation_classification_loss": result.validation_classification_loss,
        "validation_total_loss": result.validation_total_loss,
        "parameter_count": result.parameter_count,
        "training_time_seconds": result.training_duration_seconds,
        "total_time_seconds": result.total_duration_seconds,
        "best_checkpoint": relative_posix(result.best_checkpoint_path, run_dir),
        "pca_explained_variance": (
            None
            if variance is None
            else {
                "component_1": variance[0],
                "component_2": variance[1],
                "cumulative_2_components": variance[2],
            }
        ),
        "n_samples_total": result.n_samples_total,
        "n_projection_samples": result.n_projection_samples,
        "projection_metrics": result.projection_metrics,
        "warnings": result.warnings,
        "errors": result.errors,
    }


def _write_run_reports(
    run_config: ExperimentConfig,
    data: ExperimentData,
    result: SingleExperimentResult,
    run_dir: Path,
    options: SingleExperimentOptions,
) -> None:
    """Write config, environment, metrics, summary and manifest."""
    if run_config.artifacts.save_resolved_config:
        result.config_path = run_dir / "config_resolved.yaml"
        save_resolved_config(run_config, result.config_path)
    if options.write_environment:
        write_json(run_dir / ENVIRONMENT_FILENAME, collect_environment_info())
        result.environment_reference = ENVIRONMENT_FILENAME
    if data.splits.class_distribution:
        save_class_distribution(
            data.splits.class_distribution, run_dir / "tables" / "class_distribution.csv"
        )

    result.metrics_path = write_json(run_dir / "metrics.json", _metrics_payload(result, run_dir))
    result.run_summary_path = write_experiment_run_summary(result, run_dir)
    result.manifest_path = run_dir / "artifact_manifest.json"

    manifest = ArtifactManifest(
        run_dir,
        context={
            "dataset": result.dataset,
            "data_origin": result.data_origin,
            "experiment_type": result.experiment_type,
            "topology_name": result.topology_name,
            "latent_dimension": result.latent_dimension,
            "seed": result.seed,
        },
    )
    manifest.set_execution_status(result.status, result.errors)
    manifest.register_run_dir_scan(compute_hash=True)
    manifest.save()


def _run_stages(
    run_config: ExperimentConfig,
    data: ExperimentData,
    subdirs: dict[str, Path],
    evidence_profile: EvidenceProfile,
    options: SingleExperimentOptions,
    result: SingleExperimentResult,
) -> None:
    """Train, reload the best checkpoint and generate the requested evidence."""
    checkpointer, history = _train(run_config, data, subdirs, result)
    best_model = _reload_best_model(run_config, data, checkpointer, result)
    _plot_training_curves(result, history, run_config, subdirs["figures_training"])
    if evidence_profile is EvidenceProfile.FULL:
        _collect_evidence(run_config, data, subdirs, best_model, options, result)
    result.status = STATUS_PARTIAL if result.errors else STATUS_COMPLETED


def execute_single_experiment(
    config: ExperimentConfig,
    topology: TopologyConfig,
    latent_dimension: int,
    supervised: bool,
    evidence_profile: EvidenceProfile,
    output_directory: Path,
    data: ExperimentData,
    options: SingleExperimentOptions | None = None,
) -> SingleExperimentResult:
    """Train one model independently and generate its complete evidence.

    Args:
        config: Base configuration; it is copied, never mutated.
        topology: Topology of the model (name and hidden dimensions).
        latent_dimension: Latent dimension of the model.
        supervised: Enable the supervised classification component.
        evidence_profile: ``FULL`` generates every evidence category,
            ``TRAINING_ONLY`` stops after the training curves.
        output_directory: Directory of this execution (created if missing).
        data: Shared splits, loaders and device.
        options: Optional behaviour (shared indices, environment, raising).

    Returns:
        Typed result with metrics, paths, warnings and errors. A failed
        execution has ``status == "failed"`` unless ``options.raise_on_failure``
        asks for the original error to be re-raised after the reports exist.

    Raises:
        SingleExperimentError: If the derived configuration is invalid.
    """
    options = options or SingleExperimentOptions()
    run_config = derive_run_config(config, topology, latent_dimension, supervised)
    config_errors = validate_config(run_config)
    if config_errors:
        raise SingleExperimentError("; ".join(config_errors))

    run_dir = ensure_dir(Path(output_directory))
    subdirs = ensure_run_subdirectories(run_dir)
    result = _new_result(run_config, data, run_dir, options)
    log_handler = attach_run_log(subdirs["logs"] / "experiment.log")
    failure: Exception | None = None
    try:
        with Timer() as total_timer:
            try:
                _run_stages(run_config, data, subdirs, evidence_profile, options, result)
            except Exception as error:
                failure = error
                _record_failure(result, error)
        result.total_duration_seconds = total_timer.elapsed_seconds
        _write_run_reports(run_config, data, result, run_dir, options)
    finally:
        detach_run_log(log_handler)
    logger.info("Execution finished: %s (status=%s)", run_dir, result.status)
    if failure is not None and options.raise_on_failure:
        raise failure
    return result
