"""Unit tests proving that a 4D latent space is handled like any dimension above 2."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vae_project.config.models import (
    STANDARD_TOPOLOGIES,
    ModelConfig,
    SupervisedComponentConfig,
    TopologyConfig,
)
from vae_project.models.supervised_head import SupervisedClassificationHead
from vae_project.models.vae import VariationalAutoencoder
from vae_project.training.checkpointing import (
    CheckpointError,
    CheckpointMetadata,
    ModelCheckpointer,
)

pytestmark = pytest.mark.unit

FOUR_DIMENSIONAL = 4
BATCH_SIZE = 3
NUM_CLASSES = 10


def _model(topology_name: str = "small", latent_dimension: int = FOUR_DIMENSIONAL):
    hidden = STANDARD_TOPOLOGIES[topology_name]
    return VariationalAutoencoder(
        ModelConfig(
            latent_dimension=latent_dimension,
            topology=TopologyConfig(name=topology_name, hidden_dimensions=list(hidden)),
        )
    )


def _metadata(topology_name: str = "small") -> CheckpointMetadata:
    return CheckpointMetadata(
        epoch=1,
        val_total_loss=0.5,
        val_reconstruction_loss=0.4,
        val_kl_divergence_loss=0.1,
        topology_name=topology_name,
        hidden_dimensions=list(STANDARD_TOPOLOGIES[topology_name]),
        latent_dimension=FOUR_DIMENSIONAL,
        dataset_name="mnist",
    )


@pytest.mark.parametrize("topology_name", sorted(STANDARD_TOPOLOGIES))
def test_vae_exposes_four_latent_components_for_every_topology(topology_name: str) -> None:
    model = _model(topology_name)
    images = torch.rand(BATCH_SIZE, model.input_dimension)

    output = model(images)

    assert output.latent_mean.shape == (BATCH_SIZE, FOUR_DIMENSIONAL)
    assert output.latent_log_variance.shape == (BATCH_SIZE, FOUR_DIMENSIONAL)
    assert output.latent_sample.shape == (BATCH_SIZE, FOUR_DIMENSIONAL)
    assert output.reconstruction.shape == (BATCH_SIZE, model.input_dimension)


def test_decoder_accepts_four_components_and_rejects_other_sizes() -> None:
    model = _model()

    decoded = model.decode(torch.randn(BATCH_SIZE, FOUR_DIMENSIONAL))

    assert decoded.shape == (BATCH_SIZE, model.input_dimension)
    with pytest.raises(RuntimeError):
        model.decode(torch.randn(BATCH_SIZE, FOUR_DIMENSIONAL + 1))


def test_generate_samples_from_the_prior_has_four_components() -> None:
    model = _model()
    model.eval()

    samples = model.generate_samples(num_samples=5, device=torch.device("cpu"))

    assert samples.shape[0] == 5
    assert samples.shape[-1] == model.input_dimension


def test_supervised_head_accepts_four_components() -> None:
    head = SupervisedClassificationHead(
        SupervisedComponentConfig(enabled=True, hidden_dimensions=[16]),
        latent_dimension=FOUR_DIMENSIONAL,
        num_classes=NUM_CLASSES,
    )

    logits = head(torch.randn(BATCH_SIZE, FOUR_DIMENSIONAL))

    assert head.latent_dimension == FOUR_DIMENSIONAL
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    with pytest.raises(RuntimeError):
        head(torch.randn(BATCH_SIZE, 8))


@pytest.mark.parametrize("topology_name", sorted(STANDARD_TOPOLOGIES))
def test_four_dimensional_checkpoint_round_trip(tmp_path: Path, topology_name: str) -> None:
    model = _model(topology_name)
    checkpointer = ModelCheckpointer(checkpoint_dir=tmp_path / "ckpt")
    checkpointer.save_if_best(model, 0.5, _metadata(topology_name))

    restored = _model(topology_name)
    for parameter in restored.parameters():
        parameter.data.zero_()
    checkpointer.load_best(restored, device=torch.device("cpu"))
    metadata = checkpointer.get_metadata()

    for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.allclose(original, loaded)
    assert metadata is not None
    assert metadata.latent_dimension == FOUR_DIMENSIONAL
    assert metadata.topology_name == topology_name


def test_four_dimensional_checkpoint_is_not_loaded_into_another_dimension(
    tmp_path: Path,
) -> None:
    checkpointer = ModelCheckpointer(checkpoint_dir=tmp_path / "ckpt")
    checkpointer.save_if_best(_model(), 0.5, _metadata())

    with pytest.raises(CheckpointError, match="latent"):
        checkpointer.load_best(_model(latent_dimension=8), device=torch.device("cpu"))
