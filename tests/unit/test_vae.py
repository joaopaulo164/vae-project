"""Unit tests for VAE model architecture."""

from __future__ import annotations

import pytest
import torch

from vae_project.config.models import ModelConfig, TopologyConfig
from vae_project.models.vae import VAEOutput, VariationalAutoencoder

pytestmark = pytest.mark.unit


class TestVAEArchitecture:
    """Tests for VAE encoder, decoder, and latent layer shapes."""

    def test_encoder_output_shape(
        self,
        vae_model: VariationalAutoencoder,
        synthetic_batch: tuple[torch.Tensor, torch.Tensor],
        latent_dim: int,
        batch_size: int,
    ) -> None:
        """Encoder produces mu and log_var of correct shape."""
        images, _ = synthetic_batch
        flat_input = images.view(batch_size, -1)
        mu, log_var = vae_model.encode(flat_input)
        assert mu.shape == (batch_size, latent_dim)
        assert log_var.shape == (batch_size, latent_dim)

    def test_latent_mean_shape(
        self,
        vae_model: VariationalAutoencoder,
        synthetic_batch: tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        latent_dim: int,
    ) -> None:
        """latent_mean_layer output has correct shape."""
        images, _ = synthetic_batch
        vae_out = vae_model(images)
        assert vae_out.latent_mean.shape == (batch_size, latent_dim)

    def test_latent_log_variance_shape(
        self,
        vae_model: VariationalAutoencoder,
        synthetic_batch: tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        latent_dim: int,
    ) -> None:
        """latent_log_variance_layer output has correct shape."""
        images, _ = synthetic_batch
        vae_out = vae_model(images)
        assert vae_out.latent_log_variance.shape == (batch_size, latent_dim)

    def test_decoder_output_shape(
        self,
        vae_model: VariationalAutoencoder,
        synthetic_batch: tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        input_dim: int,
    ) -> None:
        """Decoder output has correct shape: (batch, input_dim)."""
        images, _ = synthetic_batch
        vae_out = vae_model(images)
        assert vae_out.reconstruction.shape == (batch_size, input_dim)

    def test_reconstruction_range(
        self,
        vae_model: VariationalAutoencoder,
        synthetic_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Reconstruction values are in [0, 1] due to Sigmoid output."""
        images, _ = synthetic_batch
        vae_out = vae_model(images)
        assert vae_out.reconstruction.min().item() >= 0.0 - 1e-6
        assert vae_out.reconstruction.max().item() <= 1.0 + 1e-6

    def test_forward_returns_vae_output(
        self,
        vae_model: VariationalAutoencoder,
        synthetic_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Forward pass returns a VAEOutput dataclass (not a tuple)."""
        images, _ = synthetic_batch
        output = vae_model(images)
        assert isinstance(output, VAEOutput)
        assert hasattr(output, "reconstruction")
        assert hasattr(output, "latent_mean")
        assert hasattr(output, "latent_log_variance")
        assert hasattr(output, "latent_sample")

    def test_parameter_count_is_positive(
        self,
        vae_model: VariationalAutoencoder,
    ) -> None:
        """Model has a positive number of trainable parameters."""
        counts = vae_model.count_parameters()
        assert counts["total"] > 0
        assert counts["encoder"] > 0
        assert counts["decoder"] > 0

    def test_latent_sample_stochastic_in_training_mode(
        self,
        vae_model: VariationalAutoencoder,
        synthetic_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """In training mode, two forward passes produce different latent_samples."""
        images, _ = synthetic_batch
        vae_model.train()
        out1 = vae_model(images)
        out2 = vae_model(images)
        # latent_mean should be identical (deterministic)
        assert torch.allclose(out1.latent_mean, out2.latent_mean, atol=1e-6)
        # latent_sample should differ due to reparameterization randomness
        # (with high probability over 784-dim space)
        assert not torch.allclose(out1.latent_sample, out2.latent_sample, atol=1e-6)

    def test_latent_sample_deterministic_in_eval_mode(
        self,
        vae_model: VariationalAutoencoder,
        synthetic_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """In eval mode, forward pass is deterministic (returns latent_mean)."""
        images, _ = synthetic_batch
        vae_model.eval()
        with torch.no_grad():
            out1 = vae_model(images)
            out2 = vae_model(images)
        assert torch.allclose(out1.latent_sample, out2.latent_sample, atol=1e-6)

    def test_decoder_can_be_called_standalone(
        self,
        vae_model: VariationalAutoencoder,
        latent_dim: int,
        batch_size: int,
        input_dim: int,
    ) -> None:
        """Decoder accepts latent vectors and produces correct-shape outputs."""
        latent_vector = torch.randn(batch_size, latent_dim)
        reconstruction = vae_model.decode(latent_vector)
        assert reconstruction.shape == (batch_size, input_dim)

    def test_generate_samples_shape(
        self,
        vae_model: VariationalAutoencoder,
        input_dim: int,
    ) -> None:
        """generate_samples returns correct shape."""
        device = torch.device("cpu")
        generated = vae_model.generate_samples(num_samples=16, device=device)
        assert generated.shape == (16, input_dim)

    def test_different_topologies_different_params(self) -> None:
        """Small and deep topologies produce different parameter counts."""
        small_config = ModelConfig(
            latent_dimension=8,
            topology=TopologyConfig(name="small", hidden_dimensions=[128]),
        )
        deep_config = ModelConfig(
            latent_dimension=8,
            topology=TopologyConfig(name="deep", hidden_dimensions=[512, 256, 128]),
        )
        small_model = VariationalAutoencoder(small_config)
        deep_model = VariationalAutoencoder(deep_config)
        assert small_model.count_parameters()["total"] < deep_model.count_parameters()["total"]
