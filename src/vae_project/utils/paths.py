"""Path utilities for reproducible, cross-platform artifact management.

Uses pathlib.Path exclusively. No absolute paths. No Bash-specific separators.
All paths are relative to the project root.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


class ArtifactError(Exception):
    """Raised when artifact directory management fails."""


def sanitize_path_component(name: str) -> str:
    """Remove characters unsafe for directory names on Windows and Linux.

    Replaces any character that is not alphanumeric, dash, or underscore
    with an underscore, and collapses consecutive underscores.

    Args:
        name: The raw string to sanitize.

    Returns:
        A safe directory name component string.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9\-_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def build_run_timestamp() -> str:
    """Build a filesystem-safe timestamp string without colons.

    Windows does not allow colons in file names, so we use underscores
    instead of the standard ISO 8601 time separator.

    Returns:
        Timestamp string in format: YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_dir(
    artifacts_dir: Path,
    dataset_name: str,
    experiment_type: str,
    latent_dimension: int,
    run_name: str | None = None,
) -> Path:
    """Build the directory path for a new experiment run.

    Each run gets a unique directory based on:
    - Dataset name (mnist / fashion_mnist)
    - Experiment type (unsupervised / supervised)
    - Latent dimension
    - Timestamp
    - Optional human-readable run name

    Args:
        artifacts_dir: Base artifacts directory path.
        dataset_name: Name of the dataset (e.g. 'mnist').
        experiment_type: Type of experiment (e.g. 'unsupervised').
        latent_dimension: Latent space dimensionality.
        run_name: Optional human-readable suffix for the run directory.

    Returns:
        Path object for the run directory (not yet created).
    """
    timestamp = build_run_timestamp()
    safe_dataset = sanitize_path_component(dataset_name)
    safe_exp_type = sanitize_path_component(experiment_type)

    folder_name = f"run_{timestamp}"
    if run_name:
        safe_run_name = sanitize_path_component(run_name)
        folder_name = f"{folder_name}_{safe_run_name}"

    run_dir = (
        artifacts_dir / safe_dataset / safe_exp_type / f"latent_{latent_dimension}" / folder_name
    )

    return run_dir


def ensure_dir(path: Path) -> Path:
    """Create a directory and all parents if they do not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path, after creation.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_run_structure(run_dir: Path) -> dict[str, Path]:
    """Create the standard subdirectory structure for a run.

    Args:
        run_dir: Root directory for this experiment run.

    Returns:
        Dictionary mapping subdirectory names to their Path objects.

    Raises:
        ArtifactError: If the run directory already exists (prevents overwrite).
    """
    if run_dir.exists():
        raise ArtifactError(
            f"Run directory already exists: {run_dir}\n"
            "Each run generates a unique timestamped directory. "
            "If you see this error, the same run was started within the same second. "
            "Wait a moment and retry."
        )

    subdirs = {
        "checkpoints": run_dir / "checkpoints",
        "figures_training": run_dir / "figures" / "training",
        "figures_reconstructions": run_dir / "figures" / "reconstructions",
        "figures_generated": run_dir / "figures" / "generated_samples",
        "figures_latent": run_dir / "figures" / "latent_space",
        "logs": run_dir / "logs",
        "tables": run_dir / "tables",
        "latent_data": run_dir / "latent_data",
    }

    run_dir.mkdir(parents=True, exist_ok=False)
    for subdir_path in subdirs.values():
        subdir_path.mkdir(parents=True, exist_ok=True)

    return subdirs
