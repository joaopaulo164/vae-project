"""Artifact manifest: tracks all files generated in a run.

Ensures transparency and reproducibility by recording every output
file with its purpose and (optionally) hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ArtifactManifest:
    """Records and persists all artifacts generated in an experiment run.

    Args:
        run_dir: Root directory of the experiment run.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self._entries: list[dict[str, object]] = []

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
        file_path = Path(file_path)
        relative_path = str(file_path.relative_to(self.run_dir))
        entry: dict[str, object] = {
            "path": relative_path,
            "artifact_type": artifact_type,
            "description": description,
            "exists": file_path.exists(),
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        }

        if compute_hash and file_path.exists():
            entry["sha256"] = _compute_file_hash(file_path)

        self._entries.append(entry)

    def register_run_dir_scan(self) -> None:
        """Scan the run directory and register all found files."""
        for file_path in sorted(self.run_dir.rglob("*")):
            if file_path.is_file() and "artifact_manifest" not in file_path.name:
                relative = str(file_path.relative_to(self.run_dir))
                suffix = file_path.suffix.lower()
                artifact_type = _infer_artifact_type(suffix, file_path.name)
                self._entries.append(
                    {
                        "path": relative,
                        "artifact_type": artifact_type,
                        "description": f"Auto-registered: {file_path.name}",
                        "exists": True,
                        "size_bytes": file_path.stat().st_size,
                    }
                )

    def save(self) -> Path:
        """Write the manifest to JSON.

        Returns:
            Path to the saved manifest file.
        """
        manifest_path = self.run_dir / "artifact_manifest.json"
        manifest_data = {
            "run_dir": str(self.run_dir),
            "total_artifacts": len(self._entries),
            "artifacts": self._entries,
        }
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest_data, manifest_file, indent=2)
        logger.info(
            "Artifact manifest saved: %s (%d entries)",
            manifest_path,
            len(self._entries),
        )
        return manifest_path

    def get_entries(self) -> list[dict[str, object]]:
        """Return a copy of all registered entries."""
        return list(self._entries)


def _compute_file_hash(file_path: Path) -> str:
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
