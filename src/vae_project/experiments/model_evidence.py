"""Evidence generated from a trained model that already holds its best weights.

This module is the single place that turns a model and a data split into the
post-training evidence categories:

* reconstructions (``figures/reconstructions``),
* samples decoded from the prior (``figures/generated_samples``),
* latent-space evidence (direct 2D plot or PCA/UMAP/t-SNE, tables, metrics).

It is shared by the single-experiment service (validation split) and by the
final test evaluation (test split), so both produce identical artifact kinds.
Step failures are recorded, never hidden.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from vae_project.config.models import ExperimentConfig
from vae_project.experiments.experiment_result import (
    FIGURE_CATEGORY_GENERATED,
    FIGURE_CATEGORY_LATENT,
    FIGURE_CATEGORY_RECONSTRUCTIONS,
)
from vae_project.experiments.latent_evidence import (
    EvidenceDirs,
    EvidenceResult,
    generate_latent_evidence,
)
from vae_project.models.vae import VariationalAutoencoder
from vae_project.visualization.reconstruction_plots import (
    plot_generated_samples,
    plot_latent_traversal_2d,
    plot_reconstruction_comparison,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_GENERATED_SAMPLES = 64
FIGURE_SUFFIXES = (".png", ".svg")


@dataclass
class EvidenceDirectories:
    """Output directories of the post-training evidence.

    Attributes:
        reconstructions: Directory for the reconstruction comparison.
        generated_samples: Directory for samples decoded from the prior.
        latent_space: Directory for latent-space figures.
        latent_data: Directory for latent representations and sample indices.
        tables: Directory for CSV tables (PCA variance, projection metrics).
    """

    reconstructions: Path
    generated_samples: Path
    latent_space: Path
    latent_data: Path
    tables: Path


@dataclass
class EvidenceRequest:
    """Everything needed to generate the evidence of one model on one split.

    Attributes:
        model: Trained VAE holding the best checkpoint weights.
        data_loader: Non-shuffled loader of the analyzed split.
        device: Compute device.
        class_names: Class names indexed by label.
        config: Configuration of this model (projections, plotting, metrics).
        split_name: ``val`` or ``test``.
        run_id: Execution identifier stored in latent data and metrics.
        directories: Output directories.
        reconstruction_indices: Fixed positions in the split to reconstruct
            (shared across models); None uses the first images of the loader.
        generation_seed: Seed of the prior samples.
        num_generated_samples: Number of prior samples.
        include_traversal: Also draw the 2D decoder manifold (2D models only).
        extra_context: Constant provenance columns (phase, selected_configuration).
        raise_on_failure: Re-raise evidence errors instead of recording them.
    """

    model: VariationalAutoencoder
    data_loader: DataLoader  # type: ignore[type-arg]
    device: torch.device
    class_names: list[str]
    config: ExperimentConfig
    split_name: str
    run_id: str
    directories: EvidenceDirectories
    reconstruction_indices: list[int] | None = None
    generation_seed: int | None = None
    num_generated_samples: int = DEFAULT_GENERATED_SAMPLES
    include_traversal: bool = False
    extra_context: dict[str, object] = field(default_factory=dict)
    raise_on_failure: bool = False


@dataclass
class EvidenceBundle:
    """Files and failures produced by :func:`collect_model_evidence`.

    Attributes:
        figure_paths: Figure files per category (reconstructions,
            generated_samples, latent_space).
        latent: Latent evidence result, or None when extraction failed.
        failures: Failed steps as ``operation/error/probable_cause`` records.
    """

    figure_paths: dict[str, list[Path]] = field(default_factory=dict)
    latent: EvidenceResult | None = None
    failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True when no evidence step failed."""
        return not self.failures

    def data_files(self) -> list[Path]:
        """Return latent data and tables (every non-figure file)."""
        if self.latent is None:
            return []
        return [path for path in self.latent.all_files() if path.suffix not in FIGURE_SUFFIXES]


def record_step(
    failures: list[dict[str, str]],
    operation: str,
    probable_cause: str,
    action: Callable[[], T],
    raise_on_failure: bool = False,
) -> T | None:
    """Run one evidence step and record (not hide) any failure.

    Args:
        failures: List receiving the failure record.
        operation: Step name.
        probable_cause: Hint stored with the failure.
        action: Zero-argument callable performing the step.
        raise_on_failure: Re-raise instead of recording.

    Returns:
        The action result, or None when it failed.
    """
    try:
        return action()
    except Exception as error:
        if raise_on_failure:
            raise
        plt.close("all")
        logger.error(
            "Evidence step failed: operation=%s, error=%s", operation, error, exc_info=True
        )
        failures.append(
            {
                "operation": operation,
                "error": f"{type(error).__name__}: {error}",
                "probable_cause": probable_cause,
            }
        )
        return None


def _title_suffix(request: EvidenceRequest) -> str:
    """Return the topology/dimension text appended to figure titles."""
    model = request.model
    return f", topology={model.topology_name}, latent_dim={model.latent_dimension}"


def _reconstruction_step(request: EvidenceRequest, bundle: EvidenceBundle) -> None:
    """Generate the original-versus-reconstruction comparison."""
    config = request.config
    paths = record_step(
        bundle.failures,
        "reconstruction_plot",
        "matplotlib failed to render or write the figure, or the shared indices are invalid.",
        lambda: plot_reconstruction_comparison(
            model=request.model,
            data_loader=request.data_loader,
            device=request.device,
            output_dir=request.directories.reconstructions,
            config=config.plotting,
            dataset_name=config.dataset.name.value,
            split_name=request.split_name,
            class_names=request.class_names,
            is_quick_mode=config.project.quick_mode,
            seed=config.reproducibility.global_seed,
            sample_indices=request.reconstruction_indices,
            title_suffix=_title_suffix(request),
        ),
        request.raise_on_failure,
    )
    if paths:
        bundle.figure_paths[FIGURE_CATEGORY_RECONSTRUCTIONS] = paths


def _generated_samples_step(request: EvidenceRequest, bundle: EvidenceBundle) -> None:
    """Generate samples decoded from the standard normal prior."""
    config = request.config
    seed = (
        request.generation_seed
        if request.generation_seed is not None
        else config.reproducibility.global_seed
    )
    paths = record_step(
        bundle.failures,
        "generated_samples_plot",
        "matplotlib failed to render or write the figure (disk space or permissions).",
        lambda: plot_generated_samples(
            model=request.model,
            device=request.device,
            output_dir=request.directories.generated_samples,
            config=config.plotting,
            dataset_name=config.dataset.name.value,
            num_samples=request.num_generated_samples,
            is_quick_mode=config.project.quick_mode,
            seed=seed,
            title_suffix=f", topology={request.model.topology_name}, n={request.num_generated_samples}",
        ),
        request.raise_on_failure,
    )
    if paths:
        bundle.figure_paths[FIGURE_CATEGORY_GENERATED] = paths


def _latent_step(request: EvidenceRequest, bundle: EvidenceBundle) -> None:
    """Extract latent means and generate the latent-space evidence."""
    config = request.config
    directories = request.directories
    latent = record_step(
        bundle.failures,
        "latent_extraction",
        "latent extraction or the latent evidence generation raised an exception (see the log).",
        lambda: generate_latent_evidence(
            model=request.model,
            data_loader=request.data_loader,
            device=request.device,
            class_names=request.class_names,
            config=config,
            dirs=EvidenceDirs(
                figures_latent=directories.latent_space,
                latent_data=directories.latent_data,
                tables=directories.tables,
            ),
            split_name=request.split_name,
            run_id=request.run_id,
            raise_on_failure=request.raise_on_failure,
            extra_context=request.extra_context,
        ),
        request.raise_on_failure,
    )
    if latent is None:
        return
    bundle.latent = latent
    bundle.failures.extend(failure.as_dict() for failure in latent.failures)
    latent_figures = [path for path in latent.all_files() if path.suffix in FIGURE_SUFFIXES]
    bundle.figure_paths.setdefault(FIGURE_CATEGORY_LATENT, []).extend(latent_figures)


def _traversal_step(request: EvidenceRequest, bundle: EvidenceBundle) -> None:
    """Draw the decoder manifold of a 2D latent space (2D models only)."""
    if not request.include_traversal or request.model.latent_dimension != 2:
        return
    config = request.config
    paths = record_step(
        bundle.failures,
        "latent_traversal_plot",
        "matplotlib failed to render or write the figure (disk space or permissions).",
        lambda: plot_latent_traversal_2d(
            model=request.model,
            device=request.device,
            output_dir=request.directories.latent_space,
            config=config.plotting,
            dataset_name=config.dataset.name.value,
            is_quick_mode=config.project.quick_mode,
        ),
        request.raise_on_failure,
    )
    if paths:
        bundle.figure_paths.setdefault(FIGURE_CATEGORY_LATENT, []).extend(paths)


def collect_model_evidence(request: EvidenceRequest) -> EvidenceBundle:
    """Generate reconstructions, prior samples and latent-space evidence.

    The model must already hold the best-checkpoint weights. A failing step is
    recorded in the bundle and does not stop the remaining steps.

    Args:
        request: Model, split, configuration and output directories.

    Returns:
        EvidenceBundle with produced files and failures.
    """
    request.model.eval()
    bundle = EvidenceBundle()
    _reconstruction_step(request, bundle)
    _generated_samples_step(request, bundle)
    _latent_step(request, bundle)
    _traversal_step(request, bundle)
    return bundle
