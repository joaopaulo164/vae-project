"""Timing utilities for measuring experiment execution durations."""

from __future__ import annotations

import time


class Timer:
    """Context manager and manual timer for measuring elapsed time.

    Usage as context manager::

        with Timer() as timer:
            do_work()
        print(f"Elapsed: {timer.elapsed_seconds:.2f}s")

    Usage manually::

        timer = Timer()
        timer.start()
        do_work()
        timer.stop()
        print(timer.elapsed_seconds)
    """

    def __init__(self) -> None:
        self._start_time: float | None = None
        self._end_time: float | None = None

    def start(self) -> Timer:
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._end_time = None
        return self

    def stop(self) -> Timer:
        """Stop the timer."""
        if self._start_time is None:
            raise RuntimeError("Timer was not started. Call start() first.")
        self._end_time = time.perf_counter()
        return self

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds.

        Returns:
            Elapsed time since start (if running) or between start and stop.

        Raises:
            RuntimeError: If timer was never started.
        """
        if self._start_time is None:
            raise RuntimeError("Timer was not started.")
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return end - self._start_time

    @property
    def elapsed_minutes(self) -> float:
        """Elapsed time in minutes."""
        return self.elapsed_seconds / 60.0

    def __enter__(self) -> Timer:
        """Start timer on context entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop timer on context exit."""
        self.stop()

    def __str__(self) -> str:
        """Human-readable elapsed time."""
        seconds = self.elapsed_seconds
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes = seconds / 60
        return f"{minutes:.1f}min ({seconds:.0f}s)"
