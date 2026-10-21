"""Checkpoint saving and loading for VAE models.

Saves the best model checkpoint during training together with the full
training state (optimizer, scheduler, optional gradient scaler, early
stopping, resolved configuration) and a JSON metadata file for provenance.

Two loading scenarios are supported:

* **Inference / evidence generation** (:meth:`ModelCheckpointer.load_for_inference`):
  only the model weights and the checkpoint metadata are required.
* **Training resume** (:meth:`ModelCheckpointer.load_for_resume`): model
  weights, optimizer, scheduler, scaler and early stopping state are required.

Legacy checkpoints (a raw ``state_dict`` in ``best_model.pt`` plus
``checkpoint_metadata.json``) keep working for inference. A resume attempt
from such a checkpoint fails with an explicit :class:`CheckpointError`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from vae_project.models.vae import VariationalAutoencoder
from vae_project.training.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 2
MODEL_STATE_KEY = "model_state_dict"
OPTIMIZER_STATE_KEY = "optimizer_state_dict"
SCHEDULER_STATE_KEY = "scheduler_state_dict"
SCALER_STATE_KEY = "scaler_state_dict"
SUPERVISED_HEAD_STATE_KEY = "supervised_head_state_dict"
EARLY_STOPPING_STATE_KEY = "early_stopping_state"


class CheckpointError(Exception):
    """Raised when checkpoint operations fail."""


class _StatefulComponent(Protocol):
    """Anything exposing ``state_dict`` / ``load_state_dict`` (scheduler, scaler, head)."""

    def state_dict(self) -> dict[str, Any]:
        """Return the serializable state."""

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        """Restore the state."""


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside a model checkpoint.

    Attributes:
        epoch: Epoch at which this checkpoint was saved.
        val_total_loss: Validation total loss at this epoch.
        val_reconstruction_loss: Validation reconstruction loss.
        val_kl_divergence_loss: Validation KL divergence loss.
        topology_name: Name of the model topology.
        hidden_dimensions: Hidden layer dimensions.
        latent_dimension: Latent space dimensionality.
        dataset_name: Dataset used for training.
        is_quick_mode: Whether this was a quick smoke test run.
        run_id: Unique run identifier.
        val_classification_loss: Validation classification loss (0.0 when the
            supervised component is disabled).
        seed: Global seed of the run (None for checkpoints from older versions).
        experiment_type: 'unsupervised' or 'supervised'.
        learning_rate: Learning rate at the checkpointed epoch.
        epochs_without_improvement: Early stopping counter at the checkpointed epoch.
    """

    epoch: int
    val_total_loss: float
    val_reconstruction_loss: float
    val_kl_divergence_loss: float
    topology_name: str
    hidden_dimensions: list[int]
    latent_dimension: int
    dataset_name: str
    is_quick_mode: bool = False
    run_id: str = ""
    val_classification_loss: float = 0.0
    seed: int | None = None
    experiment_type: str = "unsupervised"
    learning_rate: float | None = None
    epochs_without_improvement: int | None = None


@dataclass
class ResumeState:
    """Result of restoring a checkpoint for training resume.

    Attributes:
        next_epoch: 0-based epoch index at which training must continue.
        metadata: Metadata of the restored checkpoint.
        best_val_total_loss: Best validation total loss stored in the checkpoint.
    """

    next_epoch: int
    metadata: CheckpointMetadata
    best_val_total_loss: float


class ModelCheckpointer:
    """Saves and loads the best VAE checkpoint during training.

    Args:
        checkpoint_dir: Directory where checkpoints will be stored.
        run_id: Unique identifier for this run.
    """

    BEST_CHECKPOINT_FILENAME = "best_model.pt"
    METADATA_FILENAME = "checkpoint_metadata.json"

    def __init__(self, checkpoint_dir: Path, run_id: str = "") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._best_val_loss: float = float("inf")

    @property
    def best_checkpoint_path(self) -> Path:
        """Path to the best model checkpoint file."""
        return self.checkpoint_dir / self.BEST_CHECKPOINT_FILENAME

    def save_if_best(
        self,
        model: VariationalAutoencoder,
        val_loss: float,
        metadata: CheckpointMetadata,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: _StatefulComponent | None = None,
        scaler: _StatefulComponent | None = None,
        early_stopping: EarlyStopping | None = None,
        supervised_head: _StatefulComponent | None = None,
        resolved_config: dict[str, Any] | None = None,
    ) -> bool:
        """Save the model and training state if the validation loss is the best seen.

        Optional components are stored only when provided: a checkpoint saved
        without ``optimizer`` can be used for inference but not to resume
        training.

        Args:
            model: The VAE model to checkpoint.
            val_loss: Current validation total loss.
            metadata: Checkpoint metadata for provenance tracking.
            optimizer: Optimizer whose state is stored (required for resume).
            scheduler: LR scheduler, if one is used.
            scaler: Gradient scaler, if mixed precision is used.
            early_stopping: Early stopping monitor, if one is used.
            supervised_head: Optional supervised classification head.
            resolved_config: Fully resolved configuration as plain Python types.

        Returns:
            True if checkpoint was saved (new best), False otherwise.
        """
        if val_loss >= self._best_val_loss:
            return False

        self._best_val_loss = val_loss
        checkpoint_path = self.best_checkpoint_path

        payload: dict[str, Any] = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            MODEL_STATE_KEY: model.state_dict(),
            "epoch": metadata.epoch,
            "best_epoch": metadata.epoch,
            "best_val_total_loss": metadata.val_total_loss,
            "val_reconstruction_loss": metadata.val_reconstruction_loss,
            "val_kl_divergence_loss": metadata.val_kl_divergence_loss,
            "val_classification_loss": metadata.val_classification_loss,
            "latent_dimension": metadata.latent_dimension,
            "topology_name": metadata.topology_name,
            "hidden_dimensions": list(metadata.hidden_dimensions),
            "global_seed": metadata.seed,
            "dataset_name": metadata.dataset_name,
            "experiment_type": metadata.experiment_type,
            "metadata": asdict(metadata),
        }
        if optimizer is not None:
            payload[OPTIMIZER_STATE_KEY] = optimizer.state_dict()
        if scheduler is not None:
            payload[SCHEDULER_STATE_KEY] = scheduler.state_dict()
        if scaler is not None:
            payload[SCALER_STATE_KEY] = scaler.state_dict()
        if early_stopping is not None:
            payload[EARLY_STOPPING_STATE_KEY] = early_stopping.state_dict()
        if supervised_head is not None:
            payload[SUPERVISED_HEAD_STATE_KEY] = supervised_head.state_dict()
        if resolved_config is not None:
            payload["resolved_config"] = resolved_config

        torch.save(payload, checkpoint_path)

        metadata_path = self.checkpoint_dir / self.METADATA_FILENAME
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(asdict(metadata), metadata_file, indent=2)

        logger.info(
            "New best checkpoint saved: epoch=%d, val_total_loss=%.6f -> %s",
            metadata.epoch,
            val_loss,
            checkpoint_path,
        )
        return True

    def load_for_inference(
        self,
        model: VariationalAutoencoder,
        device: torch.device,
    ) -> CheckpointMetadata:
        """Load the best checkpoint weights for inference / evidence generation.

        Only the model weights and the metadata are required. Both the current
        checkpoint format and legacy raw-``state_dict`` files are accepted.

        Args:
            model: The VAE model to load weights into.
            device: Device to map checkpoint tensors to.

        Returns:
            CheckpointMetadata from the saved checkpoint.

        Raises:
            CheckpointError: If no checkpoint exists, the metadata is missing or
                the checkpoint was saved for a different latent dimension or
                topology than ``model``.
        """
        payload = self._read_payload(device)
        metadata = self._metadata_from_payload(payload)
        self._validate_model_compatibility(model, payload, metadata)

        model.load_state_dict(self._model_state(payload))

        logger.info(
            "Best checkpoint loaded for inference: epoch=%d, val_total_loss=%.6f",
            metadata.epoch,
            metadata.val_total_loss,
        )
        return metadata

    def load_best(
        self,
        model: VariationalAutoencoder,
        device: torch.device,
    ) -> CheckpointMetadata:
        """Load the best checkpoint into a model for inference.

        Alias of :meth:`load_for_inference`, kept for existing callers.

        Args:
            model: The VAE model to load weights into.
            device: Device to map checkpoint tensors to.

        Returns:
            CheckpointMetadata from the saved checkpoint.
        """
        return self.load_for_inference(model, device)

    def load_supervised_head(
        self,
        supervised_head: _StatefulComponent,
        device: torch.device,
    ) -> None:
        """Load the classification head weights stored in the best checkpoint.

        Args:
            supervised_head: Head receiving the weights.
            device: Device to map checkpoint tensors to.

        Raises:
            CheckpointError: If the checkpoint holds no classification head
                (for example, it was saved by an unsupervised run).
        """
        payload = self._read_payload(device)
        if SUPERVISED_HEAD_STATE_KEY not in payload:
            raise CheckpointError(
                f"Checkpoint {self.best_checkpoint_path} holds no supervised head state "
                f"('{SUPERVISED_HEAD_STATE_KEY}'). It was saved by an unsupervised run."
            )
        supervised_head.load_state_dict(payload[SUPERVISED_HEAD_STATE_KEY])

    def load_for_resume(
        self,
        model: VariationalAutoencoder,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: _StatefulComponent | None = None,
        scaler: _StatefulComponent | None = None,
        early_stopping: EarlyStopping | None = None,
        supervised_head: _StatefulComponent | None = None,
    ) -> ResumeState:
        """Restore model and training state to resume training.

        Every component passed in must have its state stored in the
        checkpoint; nothing is silently skipped.

        Args:
            model: The VAE model to load weights into.
            optimizer: Optimizer to restore.
            device: Device to map checkpoint tensors to.
            scheduler: LR scheduler to restore, if one is used.
            scaler: Gradient scaler to restore, if mixed precision is used.
            early_stopping: Early stopping monitor to restore.
            supervised_head: Supervised head to restore, if one is used.

        Returns:
            ResumeState with the epoch at which training must continue.

        Raises:
            CheckpointError: If the checkpoint lacks any state needed to resume
                (for example legacy weights-only checkpoints) or is incompatible
                with ``model``.
        """
        payload = self._read_payload(device)
        metadata = self._metadata_from_payload(payload)
        self._validate_model_compatibility(model, payload, metadata)

        required: list[tuple[str, bool]] = [
            (OPTIMIZER_STATE_KEY, True),
            (SCHEDULER_STATE_KEY, scheduler is not None),
            (SCALER_STATE_KEY, scaler is not None),
            (EARLY_STOPPING_STATE_KEY, early_stopping is not None),
            (SUPERVISED_HEAD_STATE_KEY, supervised_head is not None),
        ]
        missing = [key for key, needed in required if needed and key not in payload]
        if missing:
            message = (
                f"Checkpoint {self.best_checkpoint_path} cannot be used to resume "
                f"training: it does not contain the required state ({', '.join(missing)}). "
                "It is a legacy or weights-only checkpoint and can only be used "
                "for inference. Re-train to obtain a resumable checkpoint."
            )
            logger.error(message)
            raise CheckpointError(message)

        model.load_state_dict(self._model_state(payload))
        optimizer.load_state_dict(payload[OPTIMIZER_STATE_KEY])
        if scheduler is not None:
            scheduler.load_state_dict(payload[SCHEDULER_STATE_KEY])
        if scaler is not None:
            scaler.load_state_dict(payload[SCALER_STATE_KEY])
        if early_stopping is not None:
            early_stopping.load_state_dict(payload[EARLY_STOPPING_STATE_KEY])
        if supervised_head is not None:
            supervised_head.load_state_dict(payload[SUPERVISED_HEAD_STATE_KEY])

        best_val_total_loss = float(payload["best_val_total_loss"])
        self._best_val_loss = best_val_total_loss

        logger.info(
            "Training state restored from %s: resuming at epoch %d (best_val_total_loss=%.6f)",
            self.best_checkpoint_path,
            metadata.epoch + 1,
            best_val_total_loss,
        )
        return ResumeState(
            next_epoch=metadata.epoch + 1,
            metadata=metadata,
            best_val_total_loss=best_val_total_loss,
        )

    def _read_payload(self, device: torch.device) -> dict[str, Any]:
        """Read ``best_model.pt`` and normalise it to the current layout.

        Args:
            device: Device to map checkpoint tensors to.

        Returns:
            The checkpoint payload. Legacy raw state dicts are wrapped as
            ``{"legacy": True, "model_state_dict": ...}``.

        Raises:
            CheckpointError: If the checkpoint file does not exist.
        """
        if not self.best_checkpoint_path.exists():
            raise CheckpointError(
                f"No checkpoint found at {self.best_checkpoint_path}. "
                "Ensure training completed successfully and saved a checkpoint."
            )
        loaded = torch.load(
            self.best_checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        if isinstance(loaded, dict) and MODEL_STATE_KEY in loaded:
            return loaded
        return {"legacy": True, MODEL_STATE_KEY: loaded}

    def _metadata_from_payload(self, payload: dict[str, Any]) -> CheckpointMetadata:
        """Return the checkpoint metadata for a payload.

        Args:
            payload: Normalised checkpoint payload.

        Returns:
            CheckpointMetadata stored in the checkpoint, or read from the JSON
            sidecar for legacy checkpoints.

        Raises:
            CheckpointError: If the metadata is missing or malformed.
        """
        if "metadata" in payload:
            try:
                return CheckpointMetadata(**payload["metadata"])
            except TypeError as error:
                raise CheckpointError(
                    f"Malformed metadata in checkpoint {self.best_checkpoint_path}: {error}"
                ) from error
        return self._load_metadata()

    def _model_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return the model weights stored in a payload.

        Args:
            payload: Normalised checkpoint payload.

        Returns:
            The model ``state_dict``.
        """
        state: dict[str, Any] = payload[MODEL_STATE_KEY]
        return state

    def _validate_model_compatibility(
        self,
        model: VariationalAutoencoder,
        payload: dict[str, Any],
        metadata: CheckpointMetadata,
    ) -> None:
        """Ensure the checkpoint belongs to the model's latent dimension and topology.

        Args:
            model: Model that will receive the weights.
            payload: Normalised checkpoint payload.
            metadata: Checkpoint metadata.

        Raises:
            CheckpointError: If latent dimension or hidden dimensions differ, or
                the payload disagrees with its own metadata.
        """
        stored_latent = payload.get("latent_dimension", metadata.latent_dimension)
        stored_hidden = list(payload.get("hidden_dimensions", metadata.hidden_dimensions))
        if stored_latent != metadata.latent_dimension or stored_hidden != list(
            metadata.hidden_dimensions
        ):
            raise CheckpointError(
                f"Checkpoint {self.best_checkpoint_path} is inconsistent: payload has "
                f"latent_dimension={stored_latent}, hidden_dimensions={stored_hidden} but "
                f"metadata has latent_dimension={metadata.latent_dimension}, "
                f"hidden_dimensions={list(metadata.hidden_dimensions)}."
            )
        if stored_latent != model.latent_dimension:
            raise CheckpointError(
                f"Checkpoint {self.best_checkpoint_path} was saved for "
                f"latent_dimension={stored_latent}, but the model has "
                f"latent_dimension={model.latent_dimension}."
            )
        if stored_hidden != list(model.hidden_dimensions):
            raise CheckpointError(
                f"Checkpoint {self.best_checkpoint_path} was saved for "
                f"hidden_dimensions={stored_hidden}, but the model has "
                f"hidden_dimensions={list(model.hidden_dimensions)} (topology mismatch)."
            )

    def _load_metadata(self) -> CheckpointMetadata:
        """Load checkpoint metadata from JSON.

        Returns:
            CheckpointMetadata instance.

        Raises:
            CheckpointError: If metadata file is missing or malformed.
        """
        metadata_path = self.checkpoint_dir / self.METADATA_FILENAME
        if not metadata_path.exists():
            raise CheckpointError(f"Checkpoint metadata not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            data = json.load(metadata_file)
        return CheckpointMetadata(**data)

    def get_metadata(self) -> CheckpointMetadata | None:
        """Return checkpoint metadata if available, or None.

        Returns:
            CheckpointMetadata or None if no checkpoint exists.
        """
        try:
            return self._load_metadata()
        except CheckpointError:
            return None
