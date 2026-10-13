"""Early stopping implementation with best-weight restoration.

Monitors a validation metric and stops training when it stops improving.
Saves the best checkpoint seen during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from vae_project.config.models import EarlyStoppingConfig

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingState:
    """Snapshot of current early stopping state for logging.

    Attributes:
        best_value: Best metric value seen so far.
        best_epoch: Epoch index (0-based) of the best metric.
        patience_counter: Epochs without improvement since the best epoch.
        should_stop: Whether training should stop.
        triggered: Whether early stopping was actually triggered.
    """

    best_value: float
    best_epoch: int
    patience_counter: int
    should_stop: bool
    triggered: bool = False


class EarlyStopping:
    """Monitor validation metric and signal when training should stop.

    Supports:
    - Minimization (lower is better), used for loss values.
    - Best-epoch tracking for checkpoint selection.

    Args:
        config: EarlyStoppingConfig with patience, min_delta, etc.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.enabled = config.enabled
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.monitor = config.monitor

        self._best_value: float = float("inf")  # Lower is better (loss)
        self._best_epoch: int = 0
        self._patience_counter: int = 0
        self._triggered: bool = False

        if config.enabled:
            logger.info(
                "EarlyStopping configured: patience=%d, min_delta=%.6f, monitor=%s",
                self.patience,
                self.min_delta,
                self.monitor,
            )

    def step(self, current_value: float, current_epoch: int = 0) -> bool:
        """Update early stopping state with the latest metric value.

        Args:
            current_value: Current value of the monitored metric.
            current_epoch: Current epoch index (0-based).

        Returns:
            True if training should stop, False otherwise.
        """
        if not self.enabled:
            return False

        improved = current_value < (self._best_value - self.min_delta)

        if improved:
            self._best_value = current_value
            self._best_epoch = current_epoch
            self._patience_counter = 0
            logger.debug(
                "EarlyStopping: new best %.6f at epoch %d",
                self._best_value,
                self._best_epoch,
            )
        else:
            self._patience_counter += 1
            logger.debug(
                "EarlyStopping: no improvement for %d/%d epochs (best=%.6f at epoch %d)",
                self._patience_counter,
                self.patience,
                self._best_value,
                self._best_epoch,
            )

        if self._patience_counter >= self.patience:
            self._triggered = True
            logger.info(
                "Early stopping triggered at epoch %d (best epoch=%d, best_value=%.6f)",
                current_epoch,
                self._best_epoch,
                self._best_value,
            )
            return True

        return False

    @property
    def best_epoch(self) -> int:
        """Epoch index of the best validation metric."""
        return self._best_epoch

    @property
    def best_value(self) -> float:
        """Best validation metric value seen so far."""
        return self._best_value

    @property
    def triggered(self) -> bool:
        """Whether early stopping has been triggered."""
        return self._triggered

    @property
    def patience_counter(self) -> int:
        """Number of epochs without improvement."""
        return self._patience_counter

    def is_best_epoch(self, current_epoch: int) -> bool:
        """Check whether the current epoch is the best epoch.

        Args:
            current_epoch: Current epoch index.

        Returns:
            True if this epoch produced the best validation metric.
        """
        return current_epoch == self._best_epoch

    def state_dict(self) -> dict[str, Any]:
        """Return the state needed to resume monitoring, as plain Python types.

        Returns:
            Dictionary with best value/epoch, patience counter and trigger flag.
        """
        return {
            "best_value": self._best_value,
            "best_epoch": self._best_epoch,
            "patience_counter": self._patience_counter,
            "triggered": self._triggered,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the state produced by :meth:`state_dict`.

        Args:
            state: Dictionary previously returned by :meth:`state_dict`.
        """
        self._best_value = float(state["best_value"])
        self._best_epoch = int(state["best_epoch"])
        self._patience_counter = int(state["patience_counter"])
        self._triggered = bool(state["triggered"])

    def get_state(self) -> EarlyStoppingState:
        """Return a snapshot of the current state.

        Returns:
            EarlyStoppingState dataclass.
        """
        return EarlyStoppingState(
            best_value=self._best_value,
            best_epoch=self._best_epoch,
            patience_counter=self._patience_counter,
            should_stop=self._triggered,
            triggered=self._triggered,
        )
