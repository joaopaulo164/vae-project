"""Unit tests for reparameterization trick."""

from __future__ import annotations

import pytest
import torch

from vae_project.config.models import ModelConfig, TopologyConfig
from vae_project.models.vae import VariationalAutoencoder

pytestmark = pytest.mark.unit


class TestReparameterization:
    """Tests for the reparameterization trick."""

    def test_reparameterize_shape(self) -> None:
        """Reparameterized sample has same shape as mu and log_var."""
        config = ModelConfig(
            latent_dimension=8,
            topology=TopologyConfig(hidden_dimensions=[32]),
        )
        model = VariationalAutoencoder(config)
        model.train()

        mu = torch.randn(16, 8)
        log_var = torch.zeros(16, 8)
        sample = model.reparameterize(mu, log_var)
        assert sample.shape == (16, 8)

    def test_reparameterize_training_stochastic(self) -> None:
        """Two samples from same mu/log_var differ in training mode."""
        config = ModelConfig(
            latent_dimension=16,
            topology=TopologyConfig(hidden_dimensions=[32]),
        )
        model = VariationalAutoencoder(config)
        model.train()

        mu = torch.zeros(32, 16)
        log_var = torch.zeros(32, 16)  # sigma = 1
        sample1 = model.reparameterize(mu, log_var)
        sample2 = model.reparameterize(mu, log_var)
        assert not torch.allclose(sample1, sample2, atol=1e-6)

    def test_reparameterize_eval_returns_mean(self) -> None:
        """In eval mode, reparameterize returns mu directly."""
        config = ModelConfig(
            latent_dimension=8,
            topology=TopologyConfig(hidden_dimensions=[32]),
        )
        model = VariationalAutoencoder(config)
        model.eval()

        mu = torch.randn(16, 8)
        log_var = torch.randn(16, 8)
        with torch.no_grad():
            sample = model.reparameterize(mu, log_var)
        assert torch.allclose(sample, mu, atol=1e-6)

    def test_reparameterize_gradient_flows(self) -> None:
        """Gradient flows through the reparameterized sample to mu."""
        config = ModelConfig(
            latent_dimension=4,
            topology=TopologyConfig(hidden_dimensions=[16]),
        )
        model = VariationalAutoencoder(config)
        model.train()

        mu = torch.randn(4, 4, requires_grad=True)
        log_var = torch.zeros(4, 4, requires_grad=True)
        sample = model.reparameterize(mu, log_var)
        loss = sample.sum()
        loss.backward()
        assert mu.grad is not None
        assert log_var.grad is not None
