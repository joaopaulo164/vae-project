"""Variational Autoencoder (VAE) model implementation.

This module implements a fully-connected VAE suitable for 28x28 image
datasets (MNIST, Fashion-MNIST) flattened to 784-dimensional vectors.

Architecture:
    Input (784) -> Encoder MLP -> [latent_mean, latent_log_variance]
                -> Reparameterization -> latent_sample
                -> Decoder MLP -> Reconstruction (784, Sigmoid)

Key VAE components (all explicitly implemented and testable):
    - encoder: Maps input to a hidden representation.
    - latent_mean (mu): Mean of the approximate posterior q(z|x).
    - latent_log_variance (log_var): Log variance of q(z|x).
    - reparameterization trick: z = mu + epsilon * exp(0.5 * log_var)
    - latent_sample: The sampled latent vector z.
    - decoder: Maps latent vector to reconstruction.
    - reconstruction_loss: BCE or MSE between input and reconstruction.
    - kl_divergence_loss: KL(q(z|x) || p(z)) where p(z) = N(0, I).
    - total_vae_loss: reconstruction_loss + beta * kl_divergence_loss.

References:
    Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
    ICLR 2014. https://arxiv.org/abs/1312.6114
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from vae_project.config.models import ModelConfig
from vae_project.models.components import build_mlp_decoder, build_mlp_encoder

logger = logging.getLogger(__name__)


@dataclass
class VAEOutput:
    """Structured output from a VAE forward pass.

    Attributes:
        reconstruction: Decoded output tensor, shape (batch, input_dim).
        latent_mean: Mean of the approximate posterior, shape (batch, latent_dim).
        latent_log_variance: Log variance of the posterior, shape (batch, latent_dim).
        latent_sample: Sampled latent vector via reparameterization, shape (batch, latent_dim).
    """

    reconstruction: Tensor
    latent_mean: Tensor
    latent_log_variance: Tensor
    latent_sample: Tensor


class VariationalAutoencoder(nn.Module):
    """Fully-connected Variational Autoencoder for MNIST/Fashion-MNIST.

    Architecture is configured via ModelConfig, specifically:
    - model.topology.hidden_dimensions: encoder/decoder hidden layer sizes.
    - model.latent_dimension: dimensionality of the latent space.
    - model.input_dimension: input flattened size (default 784).

    The decoder is built as the symmetric reverse of the encoder for
    consistent architecture comparisons across experiments.

    Args:
        model_config: Configuration object specifying the architecture.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        self.input_dimension = model_config.input_dimension
        self.latent_dimension = model_config.latent_dimension
        self.hidden_dimensions = model_config.topology.hidden_dimensions
        self.topology_name = model_config.topology.name

        # Build encoder body: input -> hidden_dimensions[-1]
        self.encoder = build_mlp_encoder(
            input_dimension=self.input_dimension,
            hidden_dimensions=self.hidden_dimensions,
        )
        encoder_output_dim = self.hidden_dimensions[-1]

        # Latent space projection layers
        self.latent_mean_layer = nn.Linear(encoder_output_dim, self.latent_dimension)
        self.latent_log_variance_layer = nn.Linear(encoder_output_dim, self.latent_dimension)

        # Build decoder body: latent_dimension -> input_dimension (with Sigmoid)
        self.decoder = build_mlp_decoder(
            latent_dimension=self.latent_dimension,
            hidden_dimensions=self.hidden_dimensions,
            output_dimension=self.input_dimension,
        )

        self._log_architecture()

    def _log_architecture(self) -> None:
        """Log the model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "VAE architecture: topology=%s, hidden=%s, latent_dim=%d, " "params=%d (trainable=%d)",
            self.topology_name,
            self.hidden_dimensions,
            self.latent_dimension,
            total_params,
            trainable_params,
        )

    def encode(self, input_tensor: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            input_tensor: Flattened input tensor, shape (batch, input_dimension).

        Returns:
            Tuple of (latent_mean, latent_log_variance), each shape (batch, latent_dim).
        """
        hidden_representation = self.encoder(input_tensor)
        latent_mean = self.latent_mean_layer(hidden_representation)
        latent_log_variance = self.latent_log_variance_layer(hidden_representation)
        return latent_mean, latent_log_variance

    def reparameterize(self, latent_mean: Tensor, latent_log_variance: Tensor) -> Tensor:
        """Apply the reparameterization trick to sample from the posterior.

        During training, samples z = mu + epsilon * sigma where:
            sigma = exp(0.5 * log_var)
            epsilon ~ N(0, I)

        During evaluation (self.training == False), returns the mean
        directly (deterministic encoding) to avoid stochastic outputs.

        Args:
            latent_mean: Mean of q(z|x), shape (batch, latent_dim).
            latent_log_variance: Log variance of q(z|x), shape (batch, latent_dim).

        Returns:
            Sampled latent vector, shape (batch, latent_dim).
        """
        if self.training:
            standard_deviation = torch.exp(0.5 * latent_log_variance)
            epsilon = torch.randn_like(standard_deviation)
            latent_sample = latent_mean + epsilon * standard_deviation
        else:
            # At inference time, use the mean (deterministic)
            latent_sample = latent_mean
        return latent_sample

    def decode(self, latent_sample: Tensor) -> Tensor:
        """Decode a latent sample to the reconstructed input space.

        Args:
            latent_sample: Latent vector, shape (batch, latent_dim).

        Returns:
            Reconstructed tensor in [0, 1], shape (batch, input_dimension).
        """
        return cast(Tensor, self.decoder(latent_sample))

    def forward(self, input_tensor: Tensor) -> VAEOutput:
        """Full VAE forward pass.

        Args:
            input_tensor: Input image tensor, shape (batch, C, H, W) or
                (batch, input_dimension). Will be flattened automatically.

        Returns:
            VAEOutput with reconstruction, latent_mean, latent_log_variance,
            and latent_sample fields.
        """
        # Flatten: (batch, C, H, W) -> (batch, C*H*W)
        flat_input = input_tensor.view(input_tensor.size(0), -1)

        latent_mean, latent_log_variance = self.encode(flat_input)
        latent_sample = self.reparameterize(latent_mean, latent_log_variance)
        reconstruction = self.decode(latent_sample)

        return VAEOutput(
            reconstruction=reconstruction,
            latent_mean=latent_mean,
            latent_log_variance=latent_log_variance,
            latent_sample=latent_sample,
        )

    def generate_samples(
        self,
        num_samples: int,
        device: torch.device,
    ) -> Tensor:
        """Generate new samples by decoding vectors sampled from the prior.

        Samples z ~ N(0, I) and decodes them to the image space.

        Args:
            num_samples: Number of samples to generate.
            device: Device for the random tensor.

        Returns:
            Generated image tensor, shape (num_samples, input_dimension).
        """
        self.eval()
        with torch.no_grad():
            prior_samples = torch.randn(num_samples, self.latent_dimension, device=device)
            generated = self.decode(prior_samples)
        return generated

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters by component.

        Returns:
            Dictionary with parameter counts per component.
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        mean_params = sum(p.numel() for p in self.latent_mean_layer.parameters())
        log_var_params = sum(p.numel() for p in self.latent_log_variance_layer.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "encoder": encoder_params,
            "latent_mean_layer": mean_params,
            "latent_log_variance_layer": log_var_params,
            "decoder": decoder_params,
            "total": total_params,
        }
