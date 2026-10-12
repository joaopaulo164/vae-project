"""YAML configuration loader with validation and error reporting.

Loads YAML configuration files, merges with defaults, validates structure,
and produces fully-resolved ExperimentConfig objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from vae_project.config.models import ExperimentConfig
from vae_project.config.validation import validate_config

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when a configuration is invalid or missing required fields."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict.

    Args:
        base: The base dictionary (defaults).
        override: The override dictionary (user config).

    Returns:
        Merged dictionary where override values take precedence.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _convert_paths(data: dict[str, Any]) -> dict[str, Any]:
    """Convert path strings in 'paths' section to Path objects.

    Args:
        data: Raw configuration dictionary.

    Returns:
        Dictionary with path strings preserved (Path conversion happens in models).
    """
    return data


def load_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Build an ExperimentConfig from a raw dictionary.

    Args:
        data: Raw configuration dictionary (e.g., from YAML parsing).

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        ConfigurationError: If the configuration is invalid.
    """
    try:
        config = ExperimentConfig(**data)
    except TypeError as exc:
        raise ConfigurationError(
            f"Invalid configuration structure: {exc}\n"
            "Check that all configuration keys are spelled correctly and "
            "nested sections use the correct format."
        ) from exc

    errors = validate_config(config)
    if errors:
        error_list = "\n  - ".join(errors)
        raise ConfigurationError(
            f"Configuration validation failed with {len(errors)} error(s):\n"
            f"  - {error_list}\n"
            "Please review your YAML configuration and correct the issues above."
        )

    return config


def load_config(config_path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    The YAML file is merged with default values from ExperimentConfig,
    so only values that differ from defaults need to be specified.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        ConfigurationError: If the file does not exist, cannot be parsed,
            or fails validation.
        FileNotFoundError: If the configuration file is not found.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Expected absolute or relative path to a YAML file.\n"
            f"Available configs are in the 'configs/' directory."
        )

    logger.debug("Loading configuration from: %s", config_path)

    try:
        with config_path.open("r", encoding="utf-8") as file_handle:
            raw_data = yaml.safe_load(file_handle)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse YAML file: {config_path}\n"
            f"YAML error: {exc}\n"
            "Check the file for syntax errors (indentation, colons, quotes)."
        ) from exc

    if raw_data is None:
        raw_data = {}

    if not isinstance(raw_data, dict):
        raise ConfigurationError(
            f"Configuration file must contain a YAML mapping (dict), "
            f"got: {type(raw_data).__name__}\n"
            f"File: {config_path}"
        )

    logger.debug("Configuration loaded successfully, building config object.")
    config = load_config_from_dict(raw_data)

    if config.project.quick_mode:
        logger.warning(
            "QUICK MODE ENABLED: Results are for smoke testing only and "
            "must NOT be reported as scientific findings."
        )

    return config


def save_resolved_config(config: ExperimentConfig, output_path: Path) -> None:
    """Serialize the resolved configuration to a YAML file.

    This saves the fully-resolved configuration (with all defaults filled in)
    to allow exact reproduction of an experiment.

    Args:
        config: The resolved ExperimentConfig instance.
        output_path: Path where the YAML file will be written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_serializable(obj: Any) -> Any:
        """Recursively convert dataclass/enum/Path to serializable types."""
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _to_serializable(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "value"):  # Enum
            return obj.value
        if isinstance(obj, list):
            return [_to_serializable(item) for item in obj]
        if isinstance(obj, tuple):
            return [_to_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        return obj

    serializable = _to_serializable(config)

    with output_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(serializable, file_handle, default_flow_style=False, sort_keys=True)

    logger.debug("Resolved configuration saved to: %s", output_path)
