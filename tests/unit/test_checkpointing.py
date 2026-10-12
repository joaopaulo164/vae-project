"""Unit tests for path utilities and artifact management."""

from __future__ import annotations

from pathlib import Path

import pytest

from vae_project.utils.paths import (
    ArtifactError,
    build_run_dir,
    build_run_timestamp,
    create_run_structure,
    sanitize_path_component,
)

pytestmark = pytest.mark.unit


class TestPathUtils:
    """Tests for path sanitization and run directory creation."""

    def test_sanitize_removes_colons(self) -> None:
        """Colons (unsafe on Windows) are replaced with underscores."""
        result = sanitize_path_component("run:2024:01:01")
        assert ":" not in result

    def test_sanitize_removes_spaces(self) -> None:
        """Spaces are replaced with underscores."""
        result = sanitize_path_component("my run name")
        assert " " not in result

    def test_sanitize_preserves_alphanumeric(self) -> None:
        """Alphanumeric characters and dashes are preserved."""
        result = sanitize_path_component("valid-name_123")
        assert result == "valid-name_123"

    def test_timestamp_has_no_colons(self) -> None:
        """Timestamp string contains no colons (Windows-safe)."""
        ts = build_run_timestamp()
        assert ":" not in ts

    def test_build_run_dir_structure(self, tmp_path: Path) -> None:
        """build_run_dir creates expected directory path components."""
        run_dir = build_run_dir(
            artifacts_dir=tmp_path,
            dataset_name="mnist",
            experiment_type="unsupervised",
            latent_dimension=16,
        )
        assert "mnist" in str(run_dir)
        assert "unsupervised" in str(run_dir)
        assert "latent_16" in str(run_dir)

    def test_create_run_structure_creates_all_subdirs(self, tmp_path: Path) -> None:
        """create_run_structure creates all required subdirectories."""
        run_dir = tmp_path / "test_run"
        subdirs = create_run_structure(run_dir)
        for subdir_path in subdirs.values():
            assert subdir_path.exists()

    def test_create_run_structure_raises_if_exists(self, tmp_path: Path) -> None:
        """ArtifactError raised if run directory already exists."""
        run_dir = tmp_path / "existing_run"
        run_dir.mkdir()
        with pytest.raises(ArtifactError, match="already exists"):
            create_run_structure(run_dir)
