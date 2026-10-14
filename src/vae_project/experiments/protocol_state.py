"""Persistent state of a factorial protocol run (``protocol_state.json``).

The state is rewritten after every phase and every combination, so an
interrupted run can be resumed. A resume is accepted only for the same
configuration (hash) and the same dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from vae_project.config.loader import config_to_dict
from vae_project.config.models import ExperimentConfig
from vae_project.utils.json_io import read_json, write_json

logger = logging.getLogger(__name__)

STATE_FILENAME = "protocol_state.json"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
STATUS_AWAITING_MANUAL = "awaiting_manual_supervised_execution"
STATUS_SKIPPED = "skipped"
PROTOCOL_STATUSES = (
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_AWAITING_MANUAL,
)

PHASE_INITIALIZING = "initializing"
PHASE_GRID = "grid"
PHASE_SELECTION = "selection"
PHASE_FINAL_EVALUATION = "final_evaluation"
PHASE_SUPERVISED_TRAINING = "supervised_training"
PHASE_SUPERVISED_EVALUATION = "supervised_evaluation"
PHASE_REPORTING = "reporting"
PHASE_DONE = "done"

# Configuration keys that never change the scientific content of a run.
HASH_EXCLUDED_KEYS = ("run_name",)


class ProtocolStateError(Exception):
    """Raised when a stored state cannot be used to resume a run."""


def now_timestamp() -> str:
    """Return the current local time as an ISO string (seconds)."""
    return datetime.now().isoformat(timespec="seconds")


def compute_config_hash(config: ExperimentConfig) -> str:
    """Hash the resolved configuration, ignoring cosmetic keys.

    Args:
        config: Protocol configuration.

    Returns:
        SHA-256 hex digest of the canonical JSON form.
    """
    payload = config_to_dict(config)
    for key in HASH_EXCLUDED_KEYS:
        payload.pop(key, None)
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ProtocolState:
    """Serializable progress of one protocol run.

    Attributes:
        run_id: Identifier (name of the run directory).
        config_hash: Hash of the configuration that started the run.
        dataset: Dataset name.
        protocol_name: ``protocol.name`` of the configuration.
        data_origin: ``real`` or ``synthetic``.
        status: One of ``PROTOCOL_STATUSES``.
        phase: Current phase.
        completed_combinations: Keys such as ``small/latent_dim_2`` that are done.
        failed_combinations: Keys whose training failed.
        pending_combinations: Keys not yet executed.
        current_combination: Key being executed, if any.
        selected_configuration: Winner summary, once selected.
        final_evaluation_status: Status of the test evaluation of the winner.
        supervised_training_status: Status of the supervised training.
        supervised_evaluation_status: Status of the supervised test evaluation.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        resumed_at: Timestamps of every resume.
        warnings: Warning messages.
        errors: Error messages.
    """

    run_id: str
    config_hash: str
    dataset: str
    protocol_name: str = ""
    data_origin: str = "real"
    status: str = STATUS_PENDING
    phase: str = PHASE_INITIALIZING
    completed_combinations: list[str] = field(default_factory=list)
    failed_combinations: list[str] = field(default_factory=list)
    pending_combinations: list[str] = field(default_factory=list)
    current_combination: str | None = None
    selected_configuration: dict[str, Any] | None = None
    final_evaluation_status: str = STATUS_PENDING
    supervised_training_status: str = STATUS_PENDING
    supervised_evaluation_status: str = STATUS_PENDING
    created_at: str = field(default_factory=now_timestamp)
    updated_at: str = field(default_factory=now_timestamp)
    resumed_at: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def mark_combination(self, key: str, succeeded: bool) -> None:
        """Move a combination from pending/current to completed or failed."""
        if key in self.pending_combinations:
            self.pending_combinations.remove(key)
        for bucket in (self.completed_combinations, self.failed_combinations):
            if key in bucket:
                bucket.remove(key)
        (self.completed_combinations if succeeded else self.failed_combinations).append(key)
        if self.current_combination == key:
            self.current_combination = None

    def add_warning(self, message: str) -> None:
        """Record a warning once."""
        if message not in self.warnings:
            self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Record an error once."""
        if message not in self.errors:
            self.errors.append(message)


def save_state(state: ProtocolState, protocol_dir: Path) -> Path:
    """Write ``protocol_state.json`` (refreshing ``updated_at``)."""
    state.updated_at = now_timestamp()
    return write_json(Path(protocol_dir) / STATE_FILENAME, asdict(state))


def load_state(protocol_dir: Path) -> ProtocolState:
    """Read ``protocol_state.json`` of an existing run.

    Args:
        protocol_dir: Protocol run directory.

    Returns:
        The stored state.

    Raises:
        ProtocolStateError: If the file is missing or malformed.
    """
    path = Path(protocol_dir) / STATE_FILENAME
    if not path.is_file():
        raise ProtocolStateError(
            f"No {STATE_FILENAME} in {protocol_dir}. Only a directory created by "
            "run-factorial-protocol can be resumed."
        )
    try:
        payload = read_json(path)
    except (json.JSONDecodeError, OSError) as error:
        raise ProtocolStateError(f"Cannot read {path}: {error}") from error
    known = {item.name for item in fields(ProtocolState)}
    if not isinstance(payload, dict) or not {"run_id", "config_hash", "dataset"} <= set(payload):
        raise ProtocolStateError(f"{path} does not look like a protocol state file.")
    return ProtocolState(**{key: value for key, value in payload.items() if key in known})


def validate_resume(state: ProtocolState, config: ExperimentConfig) -> None:
    """Refuse to resume a run with an incompatible configuration or dataset.

    Args:
        state: State stored in the run directory.
        config: Configuration passed to ``--resume``.

    Raises:
        ProtocolStateError: If the dataset or the configuration hash differ,
            or the run already completed without any warning. A run that
            finished with warnings or failed can be resumed: its failed
            combinations and phases are retried.
    """
    if state.dataset != config.dataset.name.value:
        raise ProtocolStateError(
            f"Cannot resume: the run was created for dataset '{state.dataset}' but the "
            f"configuration uses '{config.dataset.name.value}'."
        )
    current_hash = compute_config_hash(config)
    if state.config_hash != current_hash:
        raise ProtocolStateError(
            "Cannot resume: the configuration differs from the one that created this run "
            f"(stored hash {state.config_hash[:12]}, current hash {current_hash[:12]}). "
            "Results of different configurations must not be mixed; start a new run instead."
        )
    if state.status == STATUS_COMPLETED:
        raise ProtocolStateError("Cannot resume: the protocol run already finished.")
