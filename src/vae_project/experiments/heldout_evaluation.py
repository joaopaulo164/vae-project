"""Single evaluation of a trained configuration on the held-out test split.

Used twice by the factorial protocol: for the winning unsupervised
configuration and for the supervised extension. The function loads the best
checkpoint (no retraining), validates topology and dimension against the
checkpoint, measures the losses once on the test split and produces the same
evidence categories as a grid combination, computed on the test split.

The test split never changes the topology, the dimension, the hyperparameters,
the best epoch, the checkpoint or the selection decision: this module only
reads them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from vae_project.config.loader import save_resolved_config
from vae_project.config.models import ExperimentConfig, TopologyConfig
from vae_project.experiments.experiment_result import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    SingleExperimentResult,
)
from vae_project.experiments.model_evidence import (
    EvidenceDirectories,
    EvidenceRequest,
    collect_model_evidence,
)
from vae_project.experiments.single_experiment import (
    ExperimentData,
    SingleExperimentOptions,
    apply_evidence_bundle,
    derive_run_config,
)
from vae_project.models.supervised_head import SupervisedClassificationHead
from vae_project.models.vae import VariationalAutoencoder
from vae_project.reporting.artifact_manifest import ArtifactManifest
from vae_project.reporting.experiment_summary import write_experiment_run_summary
from vae_project.reporting.formatting import format_value
from vae_project.training.checkpointing import CheckpointMetadata, ModelCheckpointer
from vae_project.training.losses import compute_vae_loss
from vae_project.utils.json_io import write_json
from vae_project.utils.paths import ensure_run_subdirectories
from vae_project.utils.reproducibility import set_global_seed
from vae_project.utils.timing import Timer

logger = logging.getLogger(__name__)

HELD_OUT_SPLIT = "test"
PHASE_FINAL_EVALUATION = "final evaluation"
FINAL_TEST_METRICS_FILENAME = "final_test_metrics.csv"
CONFUSION_MATRIX_FILENAME = "test_confusion_matrix.csv"
RESULTS_FILENAME = "final_evaluation_results.json"


@dataclass
class HeldOutMetrics:
    """Losses and classification metrics measured on the test split.

    Classification values stay ``None`` for unsupervised models: they are not
    applicable and are never reported as zero.
    """

    n_samples: int
    reconstruction_loss: float
    kl_divergence: float
    total_loss: float
    classification_loss: float | None = None
    accuracy: float | None = None
    precision_macro: float | None = None
    recall_macro: float | None = None
    f1_macro: float | None = None
    confusion_matrix: np.ndarray | None = field(default=None, repr=False)

    def as_row(self) -> dict[str, object]:
        """Return the scalar values as a CSV/JSON row."""
        return {
            "n_samples": self.n_samples,
            "test_reconstruction_loss": self.reconstruction_loss,
            "test_kl_divergence": self.kl_divergence,
            "test_classification_loss": self.classification_loss,
            "test_total_loss": self.total_loss,
            "test_accuracy": self.accuracy,
            "test_precision_macro": self.precision_macro,
            "test_recall_macro": self.recall_macro,
            "test_f1_macro": self.f1_macro,
        }


@dataclass
class HeldOutEvaluation:
    """Result of one test evaluation.

    Attributes:
        result: Typed result (``split == "test"``) with paths and evidence.
        metrics: Test losses (None if the evaluation failed before measuring).
        output_directory: Directory holding every produced file.
    """

    result: SingleExperimentResult
    metrics: HeldOutMetrics | None
    output_directory: Path


@torch.no_grad()
def measure_test_losses(
    model: VariationalAutoencoder,
    head: SupervisedClassificationHead | None,
    run_config: ExperimentConfig,
    data: ExperimentData,
) -> HeldOutMetrics:
    """Measure losses (and classification quality) on the test split.

    Losses are averaged over batches exactly like the validation losses of the
    trainer, so the numbers are directly comparable with them.

    Args:
        model: VAE holding the best-checkpoint weights.
        head: Classification head (supervised models only).
        run_config: Configuration of this model.
        data: Shared data; only ``loaders.test`` is read.

    Returns:
        Test losses and, for supervised models, classification metrics.

    Raises:
        ValueError: If the test loader is empty.
    """
    model.eval()
    if head is not None:
        head.eval()
    weight = run_config.supervised_component.classification_loss_weight if head is not None else 0.0
    sums = {"reconstruction": 0.0, "kl": 0.0, "classification": 0.0, "total": 0.0}
    predictions: list[int] = []
    labels: list[int] = []
    batches = 0
    for images, batch_labels in data.loaders.test:
        images = images.to(data.device)
        batch_labels = batch_labels.to(data.device)
        output = model(images)
        logits = None
        if head is not None:
            latent = (
                output.latent_mean
                if run_config.supervised_component.use_latent_mean
                else output.latent_sample
            )
            logits = head(latent)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
        loss = compute_vae_loss(
            reconstruction=output.reconstruction,
            original_input=images,
            latent_mean=output.latent_mean,
            latent_log_variance=output.latent_log_variance,
            loss_config=run_config.loss,
            class_logits=logits,
            class_labels=batch_labels if head is not None else None,
            classification_loss_weight=weight,
        )
        sums["reconstruction"] += loss.reconstruction_loss.item()
        sums["kl"] += loss.kl_divergence_loss.item()
        sums["classification"] += loss.supervised_classification_loss.item()
        sums["total"] += loss.total_loss.item()
        batches += 1
    if batches == 0:
        raise ValueError("the test loader is empty")
    metrics = HeldOutMetrics(
        n_samples=len(data.loaders.test.dataset),  # type: ignore[arg-type]
        reconstruction_loss=sums["reconstruction"] / batches,
        kl_divergence=sums["kl"] / batches,
        total_loss=sums["total"] / batches,
    )
    if head is not None:
        _add_classification_metrics(metrics, sums["classification"] / batches, predictions, labels)
        metrics.confusion_matrix = confusion_matrix(
            labels, predictions, labels=list(range(data.splits.num_classes))
        )
    return metrics


def _add_classification_metrics(
    metrics: HeldOutMetrics,
    classification_loss: float,
    predictions: list[int],
    labels: list[int],
) -> None:
    """Fill the classification fields of ``metrics`` (macro averages)."""
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    metrics.classification_loss = classification_loss
    metrics.accuracy = float(np.mean(np.array(labels) == np.array(predictions)))
    metrics.precision_macro = float(precision)
    metrics.recall_macro = float(recall)
    metrics.f1_macro = float(f1_score)


def _load_trained_model(
    run_config: ExperimentConfig,
    data: ExperimentData,
    checkpoint_dir: Path,
) -> tuple[VariationalAutoencoder, SupervisedClassificationHead | None, CheckpointMetadata]:
    """Load the best checkpoint into a new model, validating its identity."""
    model = VariationalAutoencoder(run_config.model).to(data.device)
    checkpointer = ModelCheckpointer(checkpoint_dir=checkpoint_dir)
    metadata = checkpointer.load_best(model=model, device=data.device)
    if (
        metadata.topology_name != run_config.model.topology.name
        or metadata.latent_dimension != run_config.model.latent_dimension
    ):
        raise ValueError(
            f"checkpoint {checkpointer.best_checkpoint_path} was saved for topology "
            f"'{metadata.topology_name}' with latent dimension {metadata.latent_dimension}, "
            f"not for '{run_config.model.topology.name}' / "
            f"{run_config.model.latent_dimension}."
        )
    model.eval()
    head = None
    if run_config.supervised_component.enabled:
        head = SupervisedClassificationHead(
            config=run_config.supervised_component,
            latent_dimension=run_config.model.latent_dimension,
            num_classes=data.splits.num_classes,
        ).to(data.device)
        checkpointer.load_supervised_head(head, data.device)
        head.eval()
    return model, head, metadata


def _write_test_tables(
    metrics: HeldOutMetrics,
    class_names: list[str],
    tables_dir: Path,
) -> list[Path]:
    """Write the test metrics CSV (and the confusion matrix when supervised)."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths = [tables_dir / FINAL_TEST_METRICS_FILENAME]
    pd.DataFrame([metrics.as_row()]).to_csv(paths[0], index=False)
    if metrics.confusion_matrix is not None:
        matrix_path = tables_dir / CONFUSION_MATRIX_FILENAME
        frame = pd.DataFrame(metrics.confusion_matrix, index=class_names, columns=class_names)
        frame.index.name = "true_class"
        frame.to_csv(matrix_path)
        paths.append(matrix_path)
    return paths


def _phase_lines(
    phase_label: str,
    metrics: HeldOutMetrics | None,
    selected_configuration: bool,
) -> list[str]:
    """Return the summary section that identifies the phase and test numbers."""
    lines = [
        "## Phase",
        "",
        "- **Split**: test",
        f"- **Phase**: {phase_label}",
        f"- **Selected configuration**: {str(selected_configuration).lower()}",
        "- The test split did not change the topology, dimension, hyperparameters, best "
        "epoch, checkpoint or the selection.",
    ]
    if metrics is None:
        return [*lines, "- Test losses were not measured (see errors)."]
    lines += [
        "",
        "### Test losses",
        "",
        f"- **Samples**: {metrics.n_samples}",
        f"- **Test reconstruction loss**: {format_value(metrics.reconstruction_loss)}",
        f"- **Test KL divergence**: {format_value(metrics.kl_divergence)}",
        f"- **Test classification loss**: {format_value(metrics.classification_loss)}",
        f"- **Test total loss**: {format_value(metrics.total_loss)}",
        f"- **Test accuracy**: {format_value(metrics.accuracy)}",
        f"- **Test precision (macro)**: {format_value(metrics.precision_macro)}",
        f"- **Test recall (macro)**: {format_value(metrics.recall_macro)}",
        f"- **Test F1 (macro)**: {format_value(metrics.f1_macro)}",
    ]
    return lines


def _results_payload(
    result: SingleExperimentResult,
    metrics: HeldOutMetrics | None,
    phase_label: str,
    selected_configuration: bool,
) -> dict[str, object]:
    """Build ``final_evaluation_results.json``."""
    checkpoint = result.best_checkpoint_path
    return {
        "split": HELD_OUT_SPLIT,
        "phase": phase_label,
        "selected_configuration": selected_configuration,
        "dataset": result.dataset,
        "data_origin": result.data_origin,
        "experiment_type": result.experiment_type,
        "topology_name": result.topology_name,
        "hidden_dimensions": result.hidden_dimensions,
        "latent_dimension": result.latent_dimension,
        "seed": result.seed,
        "execution_status": result.status,
        "best_epoch": result.best_epoch,
        "checkpoint": None if checkpoint is None else checkpoint.as_posix(),
        "validation_reconstruction_loss": result.validation_reconstruction_loss,
        "validation_kl_divergence": result.validation_kl_divergence,
        "validation_classification_loss": result.validation_classification_loss,
        "validation_total_loss": result.validation_total_loss,
        "test_metrics": None if metrics is None else metrics.as_row(),
        "pca_explained_variance": result.pca_explained_variance,
        "projection_metrics": result.projection_metrics,
        "warnings": result.warnings,
        "errors": result.errors,
    }


def _record_metadata(result: SingleExperimentResult, metadata: CheckpointMetadata) -> None:
    """Copy the checkpoint's validation values (read-only) into the result."""
    result.best_epoch = metadata.epoch
    result.validation_reconstruction_loss = metadata.val_reconstruction_loss
    result.validation_kl_divergence = metadata.val_kl_divergence_loss
    result.validation_total_loss = metadata.val_total_loss
    if metadata.experiment_type == "supervised":
        result.validation_classification_loss = metadata.val_classification_loss


def _write_reports(
    run_config: ExperimentConfig,
    result: SingleExperimentResult,
    metrics: HeldOutMetrics | None,
    output_dir: Path,
    phase_label: str,
    selected_configuration: bool,
) -> None:
    """Write config, results JSON, summary and manifest of the evaluation."""
    result.config_path = output_dir / "config_resolved.yaml"
    save_resolved_config(run_config, result.config_path)
    result.metrics_path = write_json(
        output_dir / RESULTS_FILENAME,
        _results_payload(result, metrics, phase_label, selected_configuration),
    )
    result.run_summary_path = write_experiment_run_summary(
        result, output_dir, _phase_lines(phase_label, metrics, selected_configuration)
    )
    result.manifest_path = output_dir / "artifact_manifest.json"
    manifest = ArtifactManifest(
        output_dir,
        context={
            "dataset": result.dataset,
            "data_origin": result.data_origin,
            "experiment_type": result.experiment_type,
            "topology_name": result.topology_name,
            "latent_dimension": result.latent_dimension,
            "seed": result.seed,
            "split": HELD_OUT_SPLIT,
            "phase": phase_label,
            "selected_configuration": selected_configuration,
        },
    )
    manifest.set_execution_status(result.status, result.errors)
    manifest.register_run_dir_scan(compute_hash=True)
    manifest.save()


def _collect_test_evidence(
    run_config: ExperimentConfig,
    data: ExperimentData,
    subdirs: dict[str, Path],
    model: VariationalAutoencoder,
    options: SingleExperimentOptions,
    result: SingleExperimentResult,
    phase_label: str,
    selected_configuration: bool,
) -> None:
    """Generate reconstructions, prior samples and latent evidence on the test split."""
    extra_context: dict[str, object] = {
        "phase": phase_label,
        "selected_configuration": selected_configuration,
        **(options.extra_context or {}),
    }
    request = EvidenceRequest(
        model=model,
        data_loader=data.loaders.test,
        device=data.device,
        class_names=data.splits.class_names,
        config=run_config,
        split_name=HELD_OUT_SPLIT,
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
        extra_context=extra_context,
    )
    apply_evidence_bundle(result, collect_model_evidence(request))


def _remove_empty_directories(root: Path) -> None:
    """Remove the empty directories left below ``root`` (deepest first)."""
    for directory in sorted(root.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def evaluate_on_test(
    config: ExperimentConfig,
    topology: TopologyConfig,
    latent_dimension: int,
    supervised: bool,
    checkpoint_dir: Path,
    output_dir: Path,
    data: ExperimentData,
    options: SingleExperimentOptions | None = None,
    selected_configuration: bool = True,
    phase_label: str = PHASE_FINAL_EVALUATION,
) -> HeldOutEvaluation:
    """Evaluate a trained configuration once on the test split.

    Args:
        config: Protocol configuration (copied, never mutated).
        topology: Topology of the trained model.
        latent_dimension: Latent dimension of the trained model.
        supervised: Whether the checkpoint holds a classification head.
        checkpoint_dir: Directory with ``best_model.pt``.
        output_dir: Directory receiving the evaluation artifacts.
        data: Shared data; ``loaders.test`` is the evaluated split.
        options: Shared indices/seed and the execution identifier.
        selected_configuration: Marks the evaluated configuration as selected.
        phase_label: Phase recorded in tables and manifests.

    Returns:
        Evaluation with the typed result (status ``failed`` if the checkpoint
        could not be validated or loaded) and the measured test metrics.
    """
    options = options or SingleExperimentOptions()
    run_config = derive_run_config(config, topology, latent_dimension, supervised)
    subdirs = ensure_run_subdirectories(output_dir)
    result = SingleExperimentResult(
        dataset=run_config.dataset.name.value,
        experiment_type=run_config.experiment_type,
        topology_name=topology.name,
        hidden_dimensions=list(topology.hidden_dimensions),
        latent_dimension=latent_dimension,
        seed=run_config.reproducibility.global_seed,
        split=HELD_OUT_SPLIT,
        device=str(data.device),
        quick_mode=run_config.project.quick_mode,
        synthetic_data=data.synthetic,
        execution_id=options.run_id or output_dir.name,
        run_directory=output_dir,
        best_checkpoint_path=Path(checkpoint_dir) / ModelCheckpointer.BEST_CHECKPOINT_FILENAME,
    )
    metrics: HeldOutMetrics | None = None
    with Timer() as timer:
        try:
            set_global_seed(
                run_config.reproducibility.global_seed,
                deterministic=run_config.reproducibility.deterministic_mode,
            )
            model, head, metadata = _load_trained_model(run_config, data, Path(checkpoint_dir))
            _record_metadata(result, metadata)
            metrics = measure_test_losses(model, head, run_config, data)
            result.table_paths.extend(
                _write_test_tables(metrics, data.splits.class_names, subdirs["tables"])
            )
            _collect_test_evidence(
                run_config,
                data,
                subdirs,
                model,
                options,
                result,
                phase_label,
                selected_configuration,
            )
            result.status = STATUS_PARTIAL if result.errors else STATUS_COMPLETED
        except Exception as error:
            logger.error("Test evaluation failed: %s", error, exc_info=True)
            result.status = STATUS_FAILED
            result.errors.append(
                {
                    "operation": "test_evaluation",
                    "error": f"{type(error).__name__}: {error}",
                    "probable_cause": "the checkpoint is missing, incompatible with the "
                    "selected configuration, or evidence generation raised an exception.",
                }
            )
    result.total_duration_seconds = timer.elapsed_seconds
    _write_reports(run_config, result, metrics, output_dir, phase_label, selected_configuration)
    _remove_empty_directories(output_dir)
    return HeldOutEvaluation(result=result, metrics=metrics, output_directory=output_dir)
