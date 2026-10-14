"""Artifact manifest: tracks all files generated in a run.

Ensures transparency and reproducibility by recording every output
file with its purpose, creation date, run context and (optionally) hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ArtifactManifest:
    """Records and persists all artifacts generated in an experiment run.

    Args:
        run_dir: Root directory of the experiment run.
        context: Optional constant values (e.g. latent_dimension, dataset,
            seed, experiment_type) copied into every entry and into the
            manifest header.
    """

    def __init__(self, run_dir: Path, context: dict[str, object] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.context: dict[str, object] = dict(context or {})
        self._entries: list[dict[str, object]] = []
        self._execution_status: str | None = None
        self._errors: list[dict[str, str]] = []

    def set_execution_status(
        self,
        status: str,
        errors: list[dict[str, str]] | None = None,
    ) -> None:
        """Record the overall execution status and any errors in the header.

        Args:
            status: 'completed', 'partial' or 'failed'.
            errors: Failure records (operation, error, probable_cause).
        """
        self._execution_status = status
        self._errors = list(errors or [])

    def _build_entry(
        self,
        file_path: Path,
        artifact_type: str,
        description: str,
        compute_hash: bool,
    ) -> dict[str, object]:
        """Build one manifest entry for ``file_path``."""
        exists = file_path.exists()
        entry: dict[str, object] = {
            "path": file_path.relative_to(self.run_dir).as_posix(),
            "artifact_type": artifact_type,
            "description": description,
            "exists": exists,
            "status": "present" if exists else "missing",
            "size_bytes": file_path.stat().st_size if exists else 0,
            "created_at": (
                datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(timespec="seconds")
                if exists
                else None
            ),
        }
        entry.update(self.context)
        if compute_hash and exists:
            entry["sha256"] = _compute_file_hash(file_path)
        return entry

    def register(
        self,
        file_path: Path,
        artifact_type: str,
        description: str,
        compute_hash: bool = False,
    ) -> None:
        """Register a generated file in the manifest.

        Args:
            file_path: Path to the generated file.
            artifact_type: Category (e.g. 'figure', 'table', 'checkpoint').
            description: Human-readable description of the artifact.
            compute_hash: Whether to compute the SHA-256 hash.
        """
        self._entries.append(
            self._build_entry(Path(file_path), artifact_type, description, compute_hash)
        )

    def register_run_dir_scan(self, compute_hash: bool = False) -> None:
        """Scan the run directory and register all found files.

        Args:
            compute_hash: Whether to compute the SHA-256 hash of each file.
        """
        for file_path in sorted(self.run_dir.rglob("*")):
            if file_path.is_file() and "artifact_manifest" not in file_path.name:
                artifact_type = _infer_artifact_type(file_path.suffix.lower(), file_path.name)
                self._entries.append(
                    self._build_entry(
                        file_path,
                        artifact_type,
                        f"Auto-registered: {file_path.name}",
                        compute_hash=compute_hash,
                    )
                )

    def save(self) -> Path:
        """Write the manifest to JSON.

        Returns:
            Path to the saved manifest file.
        """
        manifest_path = self.run_dir / "artifact_manifest.json"
        manifest_data: dict[str, object] = {
            "run_dir": self.run_dir.name,
            **self.context,
            "execution_status": self._execution_status,
            "errors": self._errors,
            "total_artifacts": len(self._entries),
            "artifacts": self._entries,
        }
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest_data, manifest_file, indent=2, default=str)
        logger.info(
            "Artifact manifest saved: %s (%d entries)",
            manifest_path,
            len(self._entries),
        )
        return manifest_path

    def get_entries(self) -> list[dict[str, object]]:
        """Return a copy of all registered entries."""
        return list(self._entries)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        file_path: Path to file.

    Returns:
        Hex digest string.
    """
    hasher = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_compute_file_hash = compute_file_hash


def _infer_artifact_type(suffix: str, filename: str) -> str:
    """Infer artifact type from file extension.

    Args:
        suffix: File extension including dot.
        filename: File name for additional heuristics.

    Returns:
        Artifact type string.
    """
    if suffix in {".png", ".svg", ".pdf"}:
        return "figure"
    if suffix == ".csv":
        return "table"
    if suffix in {".json", ".yaml", ".yml"}:
        return "config_or_metadata"
    if suffix in {".pt", ".pth"}:
        return "checkpoint"
    if suffix in {".npz", ".npy"}:
        return "numeric_data"
    if suffix == ".md":
        return "report"
    if suffix == ".log":
        return "log"
    return "other"
