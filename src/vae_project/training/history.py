"""Training history recording and persistence.

Records per-epoch training and validation metrics to CSV and JSON.
Provides a clean interface for trainers to append epoch results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EpochRecord:
    """Per-epoch training and validation metrics record.

    Attributes:
        epoch: Epoch index (0-based).
        train_reconstruction_loss: Training reconstruction loss.
        train_kl_divergence_loss: Training KL divergence loss.
        train_total_loss: Training total loss.
        val_reconstruction_loss: Validation reconstruction loss.
        val_kl_divergence_loss: Validation KL divergence loss.
        val_total_loss: Validation total loss.
        train_classification_loss: Training supervised classification loss (0 if disabled).
        val_classification_loss: Validation supervised classification loss (0 if disabled).
        learning_rate: Current learning rate.
        epoch_duration_seconds: Wall-clock time for this epoch.
        is_best_epoch: Whether this epoch produced the best validation loss.
        early_stopping_counter: Patience counter at this epoch.
    """

    epoch: int
    train_reconstruction_loss: float
    train_kl_divergence_loss: float
    train_total_loss: float
    val_reconstruction_loss: float
    val_kl_divergence_loss: float
    val_total_loss: float
    train_classification_loss: float = 0.0
    val_classification_loss: float = 0.0
    learning_rate: float = 0.0
    epoch_duration_seconds: float = 0.0
    is_best_epoch: bool = False
    early_stopping_counter: int = 0


class TrainingHistory:
    """Accumulates per-epoch records and saves them to CSV and JSON.

    Args:
        output_dir: Directory where history files will be saved.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[EpochRecord] = []

    def append(self, record: EpochRecord) -> None:
        """Add an epoch record to the history.

        Args:
            record: EpochRecord for the completed epoch.
        """
        self._records.append(record)

    def get_records(self) -> list[EpochRecord]:
        """Return all accumulated epoch records."""
        return list(self._records)

    def get_best_epoch(self) -> EpochRecord | None:
        """Return the epoch record with the best (lowest) val_total_loss.

        Returns:
            EpochRecord for the best epoch, or None if no records.
        """
        if not self._records:
            return None
        return min(self._records, key=lambda r: r.val_total_loss)

    def save(self) -> tuple[Path, Path]:
        """Save history to CSV and JSON files.

        Returns:
            Tuple of (csv_path, json_path).
        """
        csv_path = self.output_dir / "training_history.csv"
        json_path = self.output_dir / "training_history.json"

        if not self._records:
            logger.warning("TrainingHistory.save() called with no records.")
            return csv_path, json_path

        records_as_dicts = [asdict(record) for record in self._records]

        # Save CSV
        df = pd.DataFrame(records_as_dicts)
        df.to_csv(csv_path, index=False)

        # Save JSON
        with json_path.open("w", encoding="utf-8") as json_file:
            json.dump(records_as_dicts, json_file, indent=2)

        logger.info("Training history saved: %s, %s", csv_path, json_path)
        return csv_path, json_path

    @property
    def num_epochs_completed(self) -> int:
        """Number of epochs recorded."""
        return len(self._records)

    def latest_val_total_loss(self) -> float | None:
        """Return the most recent validation total loss, or None."""
        if not self._records:
            return None
        return self._records[-1].val_total_loss
