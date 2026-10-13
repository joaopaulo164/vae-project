"""Configuration validation logic.

Validates ExperimentConfig instances and returns descriptive error messages
that guide the user toward fixing configuration issues.
"""

from __future__ import annotations

from vae_project.config.models import ExperimentConfig


def validate_config(config: ExperimentConfig) -> list[str]:
    """Validate an ExperimentConfig instance.

    Args:
        config: The configuration object to validate.

    Returns:
        List of validation error messages. Empty list means config is valid.
    """
    errors: list[str] = []

    errors.extend(_validate_model(config))
    errors.extend(_validate_splits(config))
    errors.extend(_validate_training(config))
    errors.extend(_validate_loss(config))
    errors.extend(_validate_early_stopping(config))
    errors.extend(_validate_projections(config))
    errors.extend(_validate_supervised_component(config))
    errors.extend(_validate_plotting(config))
    errors.extend(_validate_latent_dimension_comparison(config))

    return errors


def _is_plain_int(value: object) -> bool:
    """Return True for real integers (``bool`` is rejected)."""
    return isinstance(value, int) and not isinstance(value, bool)


def validate_comparison_dimensions(dimensions: object) -> list[str]:
    """Validate a list of latent dimensions used by the comparison experiment.

    Rejects empty lists, non-integers, values below 2 and duplicates. Duplicates
    are rejected (not silently normalized) because every entry is an independent
    training run and would otherwise overwrite the same output directory.

    Args:
        dimensions: Candidate value for ``latent_dimension_comparison.dimensions``.

    Returns:
        List of error messages; empty when the value is valid.
    """
    errors: list[str] = []
    if not isinstance(dimensions, list) or not dimensions:
        return [
            "latent_dimension_comparison.dimensions must be a non-empty list of integers, "
            "e.g. [2, 8, 16, 32]."
        ]

    valid_ints: list[int] = []
    for dim in dimensions:
        if not _is_plain_int(dim):
            errors.append(
                f"latent_dimension_comparison.dimensions must contain only integers, found {dim!r}"
            )
        elif dim < 2:
            errors.append(f"latent_dimension_comparison.dimensions must be >= 2, found {dim}")
        else:
            valid_ints.append(dim)

    duplicated = sorted({dim for dim in valid_ints if valid_ints.count(dim) > 1})
    if duplicated:
        errors.append(
            f"latent_dimension_comparison.dimensions contains duplicates: {duplicated}. "
            "Each dimension is an independent training run and must appear once."
        )
    return errors


def _validate_latent_dimension_comparison(config: ExperimentConfig) -> list[str]:
    """Validate the latent dimension comparison block."""
    comparison = config.latent_dimension_comparison
    errors = validate_comparison_dimensions(comparison.dimensions)

    if not isinstance(comparison.fixed_topology, str) or not comparison.fixed_topology:
        errors.append("latent_dimension_comparison.fixed_topology must be a non-empty string.")

    return errors


def _validate_model(config: ExperimentConfig) -> list[str]:
    """Validate model configuration."""
    errors: list[str] = []

    latent_dimension = config.model.latent_dimension
    if not _is_plain_int(latent_dimension):
        errors.append(
            "model.latent_dimension must be a single integer, got "
            f"{latent_dimension!r}. A list of dimensions belongs to "
            "latent_dimension_comparison.dimensions."
        )
    elif latent_dimension < 1:
        errors.append(f"model.latent_dimension must be >= 1, got {latent_dimension}")

    if config.model.input_dimension <= 0:
        errors.append(f"model.input_dimension must be > 0, got {config.model.input_dimension}")

    if not config.model.topology.hidden_dimensions:
        errors.append(
            "model.topology.hidden_dimensions must not be empty. "
            "Provide at least one hidden layer dimension, e.g. [256]."
        )

    for dim in config.model.topology.hidden_dimensions:
        if dim <= 0:
            errors.append(f"All model.topology.hidden_dimensions must be > 0, found {dim}")

    valid_topologies = {"small", "medium", "deep", "custom"}
    if (
        config.model.topology.name not in valid_topologies
        and not config.model.topology.hidden_dimensions
    ):
        errors.append(
            f"Unknown topology name '{config.model.topology.name}'. "
            f"Use one of {sorted(valid_topologies)} or provide hidden_dimensions."
        )

    return errors


def _validate_splits(config: ExperimentConfig) -> list[str]:
    """Validate data split configuration."""
    errors: list[str] = []

    val_frac = config.splits.validation_fraction
    if not 0.0 < val_frac < 1.0:
        errors.append(f"splits.validation_fraction must be in (0, 1), got {val_frac}")

    if config.splits.quick_mode_train_size < 10:
        errors.append("splits.quick_mode_train_size must be >= 10 for meaningful smoke tests.")

    return errors


def _validate_training(config: ExperimentConfig) -> list[str]:
    """Validate training configuration."""
    errors: list[str] = []

    if config.training.max_epochs < 1:
        errors.append(f"training.max_epochs must be >= 1, got {config.training.max_epochs}")

    if config.training.batch_size < 1:
        errors.append(f"training.batch_size must be >= 1, got {config.training.batch_size}")

    if config.optimizer.learning_rate <= 0:
        errors.append(f"optimizer.learning_rate must be > 0, got {config.optimizer.learning_rate}")

    return errors


def _validate_loss(config: ExperimentConfig) -> list[str]:
    """Validate loss function configuration."""
    errors: list[str] = []

    if config.loss.beta <= 0:
        errors.append(
            f"loss.beta must be > 0, got {config.loss.beta}. " "Standard VAE uses beta=1.0."
        )

    valid_reductions = {"sum", "mean"}
    if config.loss.reduction not in valid_reductions:
        errors.append(
            f"loss.reduction must be one of {valid_reductions}, " f"got '{config.loss.reduction}'."
        )

    return errors


def _validate_early_stopping(config: ExperimentConfig) -> list[str]:
    """Validate early stopping configuration."""
    errors: list[str] = []

    if config.early_stopping.enabled:
        if config.early_stopping.patience < 1:
            errors.append(
                f"early_stopping.patience must be >= 1, " f"got {config.early_stopping.patience}"
            )
        if config.early_stopping.min_delta < 0:
            errors.append(
                f"early_stopping.min_delta must be >= 0, " f"got {config.early_stopping.min_delta}"
            )

    return errors


def _validate_projections(config: ExperimentConfig) -> list[str]:
    """Validate projection configuration."""
    errors: list[str] = []

    valid_methods = {"pca", "umap", "tsne"}
    for method in config.projections.methods:
        if method not in valid_methods:
            errors.append(
                f"Unknown projection method '{method}'. " f"Valid methods: {sorted(valid_methods)}."
            )

    if config.projections.umap.n_neighbors < 2:
        errors.append(
            f"projections.umap.n_neighbors must be >= 2, "
            f"got {config.projections.umap.n_neighbors}"
        )

    if config.projections.tsne.perplexity <= 0:
        errors.append(
            f"projections.tsne.perplexity must be > 0, " f"got {config.projections.tsne.perplexity}"
        )

    if config.projections.sampling.enabled:
        if config.projections.sampling.max_samples < 10:
            errors.append(
                f"projections.sampling.max_samples must be >= 10, "
                f"got {config.projections.sampling.max_samples}"
            )

    return errors


def _validate_supervised_component(config: ExperimentConfig) -> list[str]:
    """Validate supervised component configuration."""
    errors: list[str] = []

    if config.supervised_component.enabled:
        if config.supervised_component.classification_loss_weight <= 0:
            errors.append(
                "supervised_component.classification_loss_weight must be > 0 "
                "when supervised component is enabled."
            )
        if not config.supervised_component.hidden_dimensions:
            errors.append(
                "supervised_component.hidden_dimensions must not be empty "
                "when supervised component is enabled."
            )

    return errors


def _validate_plotting(config: ExperimentConfig) -> list[str]:
    """Validate plotting configuration."""
    errors: list[str] = []

    if config.plotting.dpi < 72:
        errors.append(
            f"plotting.dpi should be >= 72, got {config.plotting.dpi}. "
            "Low DPI figures may be unsuitable for publication."
        )

    return errors
