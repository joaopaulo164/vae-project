"""Configuration validation logic.

Validates ExperimentConfig instances and returns descriptive error messages
that guide the user toward fixing configuration issues.
"""

from __future__ import annotations

from vae_project.config.models import (
    FINAL_EVALUATION_SPLIT,
    SELECTION_NUMERIC_DIRECTIONS,
    SELECTION_NUMERIC_METRICS,
    SELECTION_TEXT_DIRECTIONS,
    SELECTION_TEXT_METRICS,
    STANDARD_TOPOLOGIES,
    SUPERVISED_EXECUTION_MODES,
    ExperimentConfig,
    FactorialComparisonConfig,
    SelectionConfig,
)


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
    errors.extend(_validate_factorial_protocol(config))

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


def _validate_selection_metric(metric: object, direction: object, label: str) -> list[str]:
    """Validate one (metric, direction) pair used to rank combinations."""
    if metric in SELECTION_NUMERIC_METRICS:
        valid_directions: tuple[str, ...] = SELECTION_NUMERIC_DIRECTIONS
    elif metric in SELECTION_TEXT_METRICS:
        valid_directions = SELECTION_TEXT_DIRECTIONS
    else:
        known = [*SELECTION_NUMERIC_METRICS, *SELECTION_TEXT_METRICS]
        return [f"{label}: unknown metric {metric!r}. Valid metrics: {known}."]
    if direction not in valid_directions:
        return [
            f"{label}: invalid direction {direction!r} for metric {metric!r}. "
            f"Valid directions: {list(valid_directions)}."
        ]
    return []


def _validate_selection(selection: SelectionConfig) -> list[str]:
    """Validate the winner selection rule."""
    errors: list[str] = []
    label = "factorial_comparison.selection"
    if selection.primary_metric not in SELECTION_NUMERIC_METRICS:
        errors.append(
            f"{label}.primary_metric: unknown or non-numeric metric "
            f"{selection.primary_metric!r}. Valid metrics: {list(SELECTION_NUMERIC_METRICS)}."
        )
    elif selection.direction not in SELECTION_NUMERIC_DIRECTIONS:
        errors.append(
            f"{label}.direction must be one of {list(SELECTION_NUMERIC_DIRECTIONS)}, "
            f"got {selection.direction!r}."
        )
    if (
        isinstance(selection.tolerance, bool)
        or not isinstance(selection.tolerance, (int, float))
        or selection.tolerance < 0
    ):
        errors.append(f"{label}.tolerance must be a number >= 0, got {selection.tolerance!r}.")
    for position, tie_breaker in enumerate(selection.tie_breakers):
        errors.extend(
            _validate_selection_metric(
                tie_breaker.metric,
                tie_breaker.direction,
                f"{label}.tie_breakers[{position}]",
            )
        )
    return errors


def _validate_factorial_grid(factorial: FactorialComparisonConfig) -> list[str]:
    """Validate topologies, dimensions and the resulting combinations."""
    errors: list[str] = []
    topologies = factorial.topologies
    if not isinstance(topologies, list) or not topologies:
        errors.append(
            "factorial_comparison.topologies must be a non-empty list, "
            f"e.g. {list(STANDARD_TOPOLOGIES)}."
        )
        topologies = []
    for name in topologies:
        if name not in STANDARD_TOPOLOGIES:
            errors.append(
                f"factorial_comparison.topologies: unknown topology {name!r}. "
                f"Valid topologies: {list(STANDARD_TOPOLOGIES)}."
            )
    duplicated_topologies = sorted({n for n in topologies if topologies.count(n) > 1})
    if duplicated_topologies:
        errors.append(
            f"factorial_comparison.topologies contains duplicates: {duplicated_topologies}. "
            "Duplicates would create duplicated (topology, dimension) combinations."
        )

    dimension_errors = validate_comparison_dimensions(factorial.latent_dimensions)
    errors.extend(
        message.replace(
            "latent_dimension_comparison.dimensions", "factorial_comparison.latent_dimensions"
        )
        for message in dimension_errors
    )
    return errors


def _validate_factorial_protocol(config: ExperimentConfig) -> list[str]:
    """Validate the factorial protocol blocks.

    The grid and the selection rule are validated only when
    ``factorial_comparison.enabled`` is true, so that every other command keeps
    accepting its existing configurations.
    """
    errors: list[str] = []
    factorial = config.factorial_comparison
    if factorial.enabled:
        errors.extend(_validate_factorial_grid(factorial))
        errors.extend(_validate_selection(factorial.selection))
        if config.supervised_component.enabled:
            errors.append(
                "supervised_component.enabled must be false in the factorial protocol: "
                "the main grid is unsupervised. Use supervised_extension instead."
            )

    if config.final_evaluation.split != FINAL_EVALUATION_SPLIT:
        errors.append(
            f"final_evaluation.split must be {FINAL_EVALUATION_SPLIT!r}, "
            f"got {config.final_evaluation.split!r}."
        )
    if config.final_evaluation.retrain_selected_configuration:
        errors.append(
            "final_evaluation.retrain_selected_configuration=true is not supported: the "
            "final evaluation reuses the best checkpoint of the factorial phase."
        )

    extension = config.supervised_extension
    if extension.execution_mode not in SUPERVISED_EXECUTION_MODES:
        errors.append(
            f"supervised_extension.execution_mode must be one of "
            f"{list(SUPERVISED_EXECUTION_MODES)}, got {extension.execution_mode!r}."
        )
    if extension.enabled:
        if extension.classification_loss_weight <= 0:
            errors.append(
                "supervised_extension.classification_loss_weight must be > 0 when the "
                "supervised extension is enabled (no classification configured otherwise)."
            )
        if not config.supervised_component.hidden_dimensions:
            errors.append(
                "supervised_component.hidden_dimensions must not be empty when the "
                "supervised extension is enabled: the classification head needs a "
                "configured architecture."
            )
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
