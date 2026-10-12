"""Checkpoint saving and loading for VAE models.

Saves the best model checkpoint during training and metadata about
the checkpoint for reproducibility and provenance tracking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from vae_project.models.vae import VariationalAutoencoder

logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Raised when checkpoint operations fail."""


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
    ) -> bool:
        """Save the model if the current validation loss is the best seen.

        Args:
            model: The VAE model to checkpoint.
            val_loss: Current validation total loss.
            metadata: Checkpoint metadata for provenance tracking.

        Returns:
            True if checkpoint was saved (new best), False otherwise.
        """
        if val_loss >= self._best_val_loss:
            return False

        self._best_val_loss = val_loss
        checkpoint_path = self.best_checkpoint_path

        torch.save(model.state_dict(), checkpoint_path)

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

    def load_best(
        self,
        model: VariationalAutoencoder,
        device: torch.device,
    ) -> CheckpointMetadata:
        """Load the best checkpoint into a model.

        Args:
            model: The VAE model to load weights into.
            device: Device to map checkpoint tensors to.

        Returns:
            CheckpointMetadata from the saved checkpoint.

        Raises:
            CheckpointError: If no checkpoint exists.
        """
        if not self.best_checkpoint_path.exists():
            raise CheckpointError(
                f"No checkpoint found at {self.best_checkpoint_path}. "
                "Ensure training completed successfully and saved a checkpoint."
            )

        state_dict = torch.load(
            self.best_checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)

        metadata = self._load_metadata()
        logger.info(
            "Best checkpoint loaded: epoch=%d, val_total_loss=%.6f",
            metadata.epoch,
            metadata.val_total_loss,
        )
        return metadata

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
