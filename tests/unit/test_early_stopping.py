"""Unit tests for early stopping."""

from __future__ import annotations

import pytest

from vae_project.config.models import EarlyStoppingConfig
from vae_project.training.early_stopping import EarlyStopping

pytestmark = pytest.mark.unit


class TestEarlyStopping:
    """Tests for EarlyStopping behavior."""

    def _make_early_stopping(self, patience: int = 3, min_delta: float = 1e-4) -> EarlyStopping:
        config = EarlyStoppingConfig(
            enabled=True,
            patience=patience,
            min_delta=min_delta,
        )
        return EarlyStopping(config)

    def test_does_not_stop_on_improvement(self) -> None:
        """No early stopping when loss keeps decreasing."""
        es = self._make_early_stopping(patience=3)
        assert not es.step(1.0, current_epoch=0)
        assert not es.step(0.9, current_epoch=1)
        assert not es.step(0.8, current_epoch=2)

    def test_stops_after_patience_epochs(self) -> None:
        """Stops after patience epochs without improvement."""
        es = self._make_early_stopping(patience=3)
        es.step(1.0, current_epoch=0)  # Best
        assert not es.step(1.1, current_epoch=1)
        assert not es.step(1.1, current_epoch=2)
        should_stop = es.step(1.1, current_epoch=3)
        assert should_stop

    def test_best_epoch_is_tracked(self) -> None:
        """Best epoch is correctly identified."""
        es = self._make_early_stopping(patience=5)
        es.step(1.0, current_epoch=0)
        es.step(0.8, current_epoch=1)  # Best
        es.step(0.85, current_epoch=2)
        assert es.best_epoch == 1
        assert abs(es.best_value - 0.8) < 1e-6

    def test_is_best_epoch(self) -> None:
        """is_best_epoch correctly identifies the best epoch."""
        es = self._make_early_stopping(patience=5)
        es.step(1.0, current_epoch=0)
        es.step(0.8, current_epoch=1)
        assert es.is_best_epoch(1)
        assert not es.is_best_epoch(0)

    def test_disabled_never_stops(self) -> None:
        """When disabled, early stopping never triggers."""
        config = EarlyStoppingConfig(enabled=False, patience=1)
        es = EarlyStopping(config)
        for epoch in range(20):
            assert not es.step(1.0, current_epoch=epoch)  # Same loss, no improvement

    def test_patience_counter_resets_on_improvement(self) -> None:
        """Patience counter resets when improvement occurs."""
        es = self._make_early_stopping(patience=5)
        es.step(1.0, current_epoch=0)
        es.step(1.1, current_epoch=1)  # No improvement
        es.step(1.1, current_epoch=2)  # No improvement
        assert es.patience_counter == 2
        es.step(0.9, current_epoch=3)  # Improvement — counter resets
        assert es.patience_counter == 0

    def test_get_state_returns_correct_info(self) -> None:
        """get_state returns accurate state snapshot."""
        es = self._make_early_stopping(patience=3)
        es.step(1.0, current_epoch=0)
        state = es.get_state()
        assert state.best_value == 1.0
        assert state.best_epoch == 0
        assert state.patience_counter == 0
        assert not state.triggered
