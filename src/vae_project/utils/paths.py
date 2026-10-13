"""Path utilities for reproducible, cross-platform artifact management.

Uses pathlib.Path exclusively. No absolute paths. No Bash-specific separators.
All paths are relative to the project root.
"""

from __future__ import annotations

import logging
import re
import secrets
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SMOKE_TEST_MARKERS = ("smoke_test", "synthetic")
_RUN_DIR_NAME_PATTERN = re.compile(r"^run_\d{8}_\d{6}_")


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


def build_latent_dimension_comparison_dir(
    artifacts_dir: Path,
    dataset_name: str,
    experiment_type: str,
) -> Path:
    """Build the directory for a latent dimension comparison.

    Follows the repository convention ``<artifacts>/<dataset>/<experiment_type>/...``
    so datasets and supervised/unsupervised results are never mixed.

    Args:
        artifacts_dir: Base artifacts directory path.
        dataset_name: Dataset name (e.g. 'mnist').
        experiment_type: 'unsupervised' or 'supervised'.

    Returns:
        Path of the comparison directory (not yet created).
    """
    return (
        artifacts_dir
        / sanitize_path_component(dataset_name)
        / sanitize_path_component(experiment_type)
        / "latent_dimension_comparison"
    )


def build_comparison_run_dir(base_dir: Path, run_name: str | None = None) -> Path:
    """Build a new, exclusive directory for one latent dimension comparison run.

    The name is ``run_YYYYMMDD_HHMMSS_<identifier>``. The identifier is the
    sanitized ``run_name`` or, by default, a short random hex string, so two
    runs started within the same second never share a directory. The name is
    Windows-safe (no colons or other reserved characters).

    Args:
        base_dir: Directory that groups comparison runs.
        run_name: Optional human-readable identifier.

    Returns:
        Path of the run directory (not yet created).

    Raises:
        ArtifactError: If an explicit ``run_name`` produces a directory that
            already exists.
    """
    timestamp = build_run_timestamp()
    safe_name = sanitize_path_component(run_name) if run_name else ""
    if safe_name:
        candidate = Path(base_dir) / f"run_{timestamp}_{safe_name}"
        if candidate.exists():
            raise ArtifactError(
                f"Run directory already exists: {candidate}. "
                "Choose another --run-name or wait one second."
            )
        return candidate
    while True:
        candidate = Path(base_dir) / f"run_{timestamp}_{secrets.token_hex(3)}"
        if not candidate.exists():
            return candidate


def is_comparison_run_dir(path: Path) -> bool:
    """Tell whether an existing directory is (or holds the output of) a comparison run.

    A directory counts as a run directory when it is named
    ``run_YYYYMMDD_HHMMSS_<identifier>`` or already contains comparison outputs.

    Args:
        path: Directory to inspect.

    Returns:
        True when ``path`` exists and looks like a comparison run directory.
    """
    path = Path(path)
    if not path.is_dir():
        return False
    if _RUN_DIR_NAME_PATTERN.match(path.name):
        return True
    return (path / "latent_dimension_comparison.csv").exists() or any(path.glob("latent_dim_*"))


def ensure_run_dir_is_free(run_dir: Path, overwrite: bool = False) -> None:
    """Refuse to write into a non-empty run directory unless overwrite is explicit.

    Args:
        run_dir: Directory about to receive run artifacts.
        overwrite: Whether the caller explicitly accepted overwriting.

    Raises:
        ArtifactError: If ``run_dir`` is a file, or holds files and ``overwrite``
            is False.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise ArtifactError(f"Run path exists and is not a directory: {run_dir}")
    if not any(run_dir.iterdir()):
        return
    if not overwrite:
        raise ArtifactError(
            f"Conflict: run directory already contains artifacts: {run_dir}. "
            "Nothing was written. Use a new run directory (omit --output-dir or pass its "
            "parent as --output-dir), or pass --overwrite to replace files with the same name."
        )
    logger.warning("Overwrite enabled: files with the same name in %s will be replaced.", run_dir)


def resolve_comparison_run_dir(
    output_dir: Path | None,
    default_base_dir: Path,
    overwrite: bool = False,
    run_name: str | None = None,
) -> Path:
    """Decide the run directory of a comparison from the CLI arguments.

    ``output_dir`` is treated as a *base* directory: a new exclusive
    ``run_<timestamp>_<identifier>`` directory is created inside it. When it
    already is a run directory, it is rejected unless ``overwrite`` is set, in
    which case it is reused as the run directory itself.

    Args:
        output_dir: User-supplied ``--output-dir`` or None.
        default_base_dir: Base used when ``output_dir`` is None.
        overwrite: Explicit permission to reuse an existing run directory.
        run_name: Optional identifier for the new run directory.

    Returns:
        The run directory to pass to the comparison.

    Raises:
        ArtifactError: If ``output_dir`` is an existing run directory and
            ``overwrite`` is False, or the new run directory name is taken.
    """
    if output_dir is not None and is_comparison_run_dir(output_dir):
        ensure_run_dir_is_free(output_dir, overwrite=overwrite)
        return Path(output_dir)
    base_dir = Path(output_dir) if output_dir is not None else Path(default_base_dir)
    return build_comparison_run_dir(base_dir, run_name)


def ensure_smoke_test_dir(path: Path) -> None:
    """Require 'smoke_test' or 'synthetic' in the path of synthetic-data runs.

    Args:
        path: Output directory of a run that uses synthetic data.

    Raises:
        ArtifactError: If no path component contains a smoke-test marker.
    """
    parts = [part.lower() for part in Path(path).parts]
    if not any(marker in part for part in parts for marker in SMOKE_TEST_MARKERS):
        raise ArtifactError(
            f"Synthetic-data runs must write into a directory whose path contains "
            f"'smoke_test' or 'synthetic', got: {path}"
        )


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
