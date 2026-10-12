"""Integration test: smoke CLI test."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestSmokeCLI:
    """Tests that the CLI smoke-test command runs successfully."""

    def test_smoke_test_command_exits_zero(self) -> None:
        """CLI smoke-test runs without error."""
        import argparse

        from vae_project.cli import _cmd_smoke_test

        args = argparse.Namespace()
        exit_code = _cmd_smoke_test(args)
        assert exit_code == 0

    def test_validate_config_exits_zero_for_valid_config(self, tmp_path) -> None:
        """validate-config command returns 0 for a valid config."""
        import argparse

        import yaml

        from vae_project.cli import _cmd_validate_config

        config_data = {
            "dataset": {"name": "mnist"},
            "model": {
                "latent_dimension": 8,
                "topology": {"name": "small", "hidden_dimensions": [64]},
            },
        }
        config_file = tmp_path / "test.yaml"
        with config_file.open("w") as f:
            yaml.safe_dump(config_data, f)

        args = argparse.Namespace(config=config_file)
        exit_code = _cmd_validate_config(args)
        assert exit_code == 0

    def test_validate_config_exits_nonzero_for_invalid(self, tmp_path) -> None:
        """validate-config command returns non-zero for invalid config."""
        import argparse

        import yaml

        from vae_project.cli import _cmd_validate_config

        config_data = {
            "model": {
                "latent_dimension": 0,  # Invalid
                "topology": {"hidden_dimensions": [64]},
            },
        }
        config_file = tmp_path / "bad_config.yaml"
        with config_file.open("w") as f:
            yaml.safe_dump(config_data, f)

        args = argparse.Namespace(config=config_file)
        exit_code = _cmd_validate_config(args)
        assert exit_code != 0
