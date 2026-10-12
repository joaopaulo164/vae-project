"""Unit tests for VAE loss functions."""

from __future__ import annotations

import pytest
import torch

from vae_project.config.models import LossConfig, ReconstructionLossType
from vae_project.training.losses import (
    VAELossOutput,
    compute_kl_divergence_loss,
    compute_reconstruction_loss,
    compute_vae_loss,
)

pytestmark = pytest.mark.unit


class TestReconstructionLoss:
    """Tests for reconstruction loss computation."""

    def test_bce_loss_positive(self) -> None:
        """BCE reconstruction loss is non-negative."""
        reconstruction = torch.sigmoid(torch.randn(16, 784))
        original = torch.rand(16, 784)
        loss = compute_reconstruction_loss(reconstruction, original, ReconstructionLossType.BCE)
        assert loss.item() >= 0.0

    def test_mse_loss_positive(self) -> None:
        """MSE reconstruction loss is non-negative."""
        reconstruction = torch.rand(16, 784)
        original = torch.rand(16, 784)
        loss = compute_reconstruction_loss(reconstruction, original, ReconstructionLossType.MSE)
        assert loss.item() >= 0.0

    def test_perfect_reconstruction_near_zero(self) -> None:
        """Loss is near zero when reconstruction matches input."""
        target = torch.full((16, 784), 0.5)
        loss = compute_reconstruction_loss(target.clone(), target, ReconstructionLossType.MSE)
        assert loss.item() < 1e-6

    def test_4d_input_is_flattened(self) -> None:
        """Loss function accepts 4D image tensors."""
        reconstruction = torch.rand(8, 784)
        original_4d = torch.rand(8, 1, 28, 28)
        loss = compute_reconstruction_loss(reconstruction, original_4d, ReconstructionLossType.MSE)
        assert loss.item() >= 0.0

    def test_scalar_output(self) -> None:
        """Loss is a scalar tensor."""
        reconstruction = torch.rand(16, 784)
        original = torch.rand(16, 784)
        loss = compute_reconstruction_loss(reconstruction, original, ReconstructionLossType.BCE)
        assert loss.shape == torch.Size([])

    def test_unknown_loss_type_raises(self) -> None:
        """ValueError raised for unknown loss type."""
        reconstruction = torch.rand(4, 784)
        original = torch.rand(4, 784)
        with pytest.raises(ValueError, match="Unknown reconstruction loss"):
            compute_reconstruction_loss(
                reconstruction,
                original,
                "unknown_type",  # type: ignore[arg-type]
            )


class TestKLDivergenceLoss:
    """Tests for KL divergence loss computation."""

    def test_standard_normal_kl_near_zero(self) -> None:
        """KL divergence is near zero when q matches the standard normal prior."""
        batch_size, latent_dim = 64, 16
        # mu=0, log_var=0 => sigma=1 => exactly N(0,1)
        mu = torch.zeros(batch_size, latent_dim)
        log_var = torch.zeros(batch_size, latent_dim)
        kl = compute_kl_divergence_loss(mu, log_var)
        assert abs(kl.item()) < 1e-5

    def test_kl_positive_for_non_standard(self) -> None:
        """KL divergence is positive when q != prior."""
        batch_size, latent_dim = 16, 8
        mu = torch.ones(batch_size, latent_dim) * 5.0  # Far from prior
        log_var = torch.zeros(batch_size, latent_dim)
        kl = compute_kl_divergence_loss(mu, log_var)
        assert kl.item() > 0.0

    def test_kl_analytical_formula(self) -> None:
        """Verify KL formula against analytical computation.

        For N(mu, sigma^2) vs N(0,1):
        KL = -0.5 * (1 + log_var - mu^2 - exp(log_var))
        """
        mu = torch.tensor([[1.0, 2.0]])
        log_var = torch.tensor([[0.5, -0.5]])

        computed_kl = compute_kl_divergence_loss(mu, log_var)

        # Analytical per-element:
        expected = -0.5 * ((1 + log_var - mu**2 - log_var.exp()).sum(dim=1).mean())
        assert abs(computed_kl.item() - expected.item()) < 1e-5

    def test_kl_scalar_output(self) -> None:
        """KL divergence is a scalar tensor."""
        mu = torch.randn(8, 4)
        log_var = torch.zeros(8, 4)
        kl = compute_kl_divergence_loss(mu, log_var)
        assert kl.shape == torch.Size([])


class TestTotalVAELoss:
    """Tests for the combined VAE loss."""

    def test_total_loss_components_present(self) -> None:
        """VAELossOutput contains all required fields."""
        reconstruction = torch.rand(16, 784)
        original = torch.rand(16, 1, 28, 28)
        mu = torch.randn(16, 8)
        log_var = torch.zeros(16, 8)
        config = LossConfig()

        output = compute_vae_loss(
            reconstruction=reconstruction,
            original_input=original,
            latent_mean=mu,
            latent_log_variance=log_var,
            loss_config=config,
        )

        assert isinstance(output, VAELossOutput)
        assert hasattr(output, "reconstruction_loss")
        assert hasattr(output, "kl_divergence_loss")
        assert hasattr(output, "supervised_classification_loss")
        assert hasattr(output, "total_loss")

    def test_total_loss_is_sum_of_components(self) -> None:
        """total_loss = reconstruction_loss + beta * kl_divergence_loss."""
        reconstruction = torch.sigmoid(torch.randn(16, 784))
        original = torch.rand(16, 784)
        mu = torch.randn(16, 8)
        log_var = torch.zeros(16, 8)
        config = LossConfig(beta=2.0)

        output = compute_vae_loss(
            reconstruction=reconstruction,
            original_input=original,
            latent_mean=mu,
            latent_log_variance=log_var,
            loss_config=config,
        )

        expected = output.reconstruction_loss + 2.0 * output.kl_divergence_loss
        assert abs(output.total_loss.item() - expected.item()) < 1e-5

    def test_supervised_classification_loss_zero_when_disabled(self) -> None:
        """Supervised loss is zero when class_logits are not provided."""
        reconstruction = torch.rand(8, 784)
        original = torch.rand(8, 784)
        mu = torch.randn(8, 4)
        log_var = torch.zeros(8, 4)
        config = LossConfig()

        output = compute_vae_loss(
            reconstruction=reconstruction,
            original_input=original,
            latent_mean=mu,
            latent_log_variance=log_var,
            loss_config=config,
        )
        assert output.supervised_classification_loss.item() == 0.0

    def test_supervised_loss_nonzero_when_enabled(self) -> None:
        """Supervised loss is nonzero when class logits and labels are provided."""
        batch = 16
        n_classes = 10
        reconstruction = torch.rand(batch, 784)
        original = torch.rand(batch, 784)
        mu = torch.randn(batch, 8)
        log_var = torch.zeros(batch, 8)
        class_logits = torch.randn(batch, n_classes)
        class_labels = torch.randint(0, n_classes, (batch,))
        config = LossConfig()

        output = compute_vae_loss(
            reconstruction=reconstruction,
            original_input=original,
            latent_mean=mu,
            latent_log_variance=log_var,
            loss_config=config,
            class_logits=class_logits,
            class_labels=class_labels,
            classification_loss_weight=1.0,
        )
        assert output.supervised_classification_loss.item() > 0.0

    def test_as_dict_returns_floats(self) -> None:
        """VAELossOutput.as_dict() returns float values."""
        reconstruction = torch.rand(4, 784)
        original = torch.rand(4, 784)
        mu = torch.randn(4, 4)
        log_var = torch.zeros(4, 4)
        config = LossConfig()

        output = compute_vae_loss(
            reconstruction=reconstruction,
            original_input=original,
            latent_mean=mu,
            latent_log_variance=log_var,
            loss_config=config,
        )
        loss_dict = output.as_dict()
        assert all(isinstance(v, float) for v in loss_dict.values())
