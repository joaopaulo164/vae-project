"""Files shared by every combination of a factorial protocol run.

The numeric latent representation changes between models, but the source
examples must not: all combinations of a dataset are projected and
reconstructed from the same examples and decode prior samples drawn with the
same seed. The files written here document that shared choice:

    shared/environment.json
    shared/class_distribution.csv
    shared/projection_sample_indices.csv
    shared/reconstruction_sample_indices.csv
    shared/generation_seed.json
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vae_project.config.models import ExperimentConfig
from vae_project.data.splits import save_class_distribution
from vae_project.experiments.latent_evidence import projection_sample_positions
from vae_project.experiments.model_evidence import DEFAULT_GENERATED_SAMPLES
from vae_project.experiments.single_experiment import ExperimentData
from vae_project.utils.environment import collect_environment_info
from vae_project.utils.json_io import write_json

logger = logging.getLogger(__name__)

ENVIRONMENT_FILENAME = "environment.json"
CLASS_DISTRIBUTION_FILENAME = "class_distribution.csv"
PROJECTION_INDICES_FILENAME = "projection_sample_indices.csv"
RECONSTRUCTION_INDICES_FILENAME = "reconstruction_sample_indices.csv"
GENERATION_SEED_FILENAME = "generation_seed.json"
VALIDATION_SPLIT = "val"


@dataclass
class SharedSamples:
    """Sample choices shared by all combinations of the run.

    Attributes:
        projection_positions: Positions in the validation split used by
            PCA/UMAP/t-SNE (informational; the evidence code derives the same
            positions from the labels and the sampling seed).
        reconstruction_indices: Positions reconstructed in every figure.
        generation_seed: Seed of the samples decoded from the prior.
        num_generated_samples: Number of prior samples.
        files: Files written into ``shared/``.
    """

    projection_positions: list[int]
    reconstruction_indices: list[int]
    generation_seed: int
    num_generated_samples: int
    files: list[Path]


def _validation_labels(data: ExperimentData) -> np.ndarray:
    """Collect the labels of the validation split in loader order."""
    labels: list[int] = []
    for _, batch_labels in data.loaders.val:
        labels.extend(batch_labels.tolist())
    return np.array(labels, dtype=np.int64)


def choose_reconstruction_indices(
    split_sizes: list[int],
    count: int,
    seed: int,
) -> list[int]:
    """Pick fixed positions that exist in every analyzed split.

    Args:
        split_sizes: Sizes of the splits that will be reconstructed.
        count: Number of images to show.
        seed: Seed of the choice.

    Returns:
        Sorted positions valid for the smallest split.
    """
    available = min(split_sizes)
    chosen = np.random.default_rng(seed).choice(
        available, size=min(count, available), replace=False
    )
    return sorted(int(position) for position in chosen)


def _write_projection_indices(
    shared_dir: Path,
    positions: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    total: int,
    config: ExperimentConfig,
) -> Path:
    """Write the shared projection sample indices."""
    frame = pd.DataFrame(
        {
            "sample_index": positions,
            "label": labels[positions],
            "class_name": [class_names[int(label)] for label in labels[positions]],
        }
    )
    frame["split"] = VALIDATION_SPLIT
    frame["dataset"] = config.dataset.name.value
    frame["n_projection_samples"] = len(positions)
    frame["n_samples_total"] = total
    frame["sampling_seed"] = config.projections.sampling.seed
    path = shared_dir / PROJECTION_INDICES_FILENAME
    frame.to_csv(path, index=False)
    return path


def _write_reconstruction_indices(
    shared_dir: Path,
    indices: list[int],
    labels: np.ndarray,
    class_names: list[str],
) -> Path:
    """Write the shared reconstruction sample indices."""
    in_validation = [index for index in indices if index < len(labels)]
    frame = pd.DataFrame(
        {
            "sample_index": indices,
            "label": [int(labels[index]) if index in in_validation else None for index in indices],
            "class_name": [
                class_names[int(labels[index])] if index in in_validation else None
                for index in indices
            ],
        }
    )
    path = shared_dir / RECONSTRUCTION_INDICES_FILENAME
    frame.to_csv(path, index=False)
    return path


def write_shared_files(
    shared_dir: Path,
    config: ExperimentConfig,
    data: ExperimentData,
) -> SharedSamples:
    """Create (or complete) the ``shared`` directory of a protocol run.

    Existing valid files are kept; the content is deterministic, so rewriting
    would produce the same values.

    Args:
        shared_dir: ``shared`` directory of the run.
        config: Protocol configuration.
        data: Shared splits and loaders.

    Returns:
        The shared sample choices and the written files.
    """
    shared_dir.mkdir(parents=True, exist_ok=True)
    labels = _validation_labels(data)
    class_names = data.splits.class_names
    seed = config.reproducibility.global_seed
    positions = projection_sample_positions(labels, config)
    reconstruction = choose_reconstruction_indices(
        [len(labels), len(data.loaders.test.dataset)],  # type: ignore[arg-type]
        config.plotting.reconstruction_grid_cols,
        seed,
    )

    files: list[Path] = []
    environment_path = shared_dir / ENVIRONMENT_FILENAME
    if not environment_path.is_file():
        write_json(environment_path, collect_environment_info())
    files.append(environment_path)
    if data.splits.class_distribution:
        distribution_path = shared_dir / CLASS_DISTRIBUTION_FILENAME
        save_class_distribution(data.splits.class_distribution, distribution_path)
        files.append(distribution_path)
    files.append(
        _write_projection_indices(shared_dir, positions, labels, class_names, len(labels), config)
    )
    files.append(_write_reconstruction_indices(shared_dir, reconstruction, labels, class_names))
    seed_path = write_json(
        shared_dir / GENERATION_SEED_FILENAME,
        {
            "generation_seed": seed,
            "num_generated_samples": DEFAULT_GENERATED_SAMPLES,
            "distribution": "standard normal prior N(0, I)",
        },
    )
    files.append(seed_path)
    logger.info("Shared files written to %s", shared_dir)
    return SharedSamples(
        projection_positions=[int(position) for position in positions],
        reconstruction_indices=reconstruction,
        generation_seed=seed,
        num_generated_samples=DEFAULT_GENERATED_SAMPLES,
        files=files,
    )
