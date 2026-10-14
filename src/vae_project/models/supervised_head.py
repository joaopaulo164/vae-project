"""Optional supervised classification head for the VAE latent space.

When the supervised_component is enabled in configuration, this head
is trained jointly with the VAE. Its classification loss is combined
with the VAE loss.

When disabled, this component is not instantiated and has zero effect
on training dynamics.

IMPORTANT: This component is optional and secondary. The primary
experiment is the unsupervised VAE. Supervised results must be
reported separately and never mixed with unsupervised results.
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torch import Tensor

from vae_project.config.models import SupervisedComponentConfig
from vae_project.models.components import build_classifier_mlp

logger = logging.getLogger(__name__)


class SupervisedClassificationHead(nn.Module):
    """Classification head that operates on VAE latent representations.

    Receives either latent_mean (preferred, deterministic) or latent_sample
    as input and produces class logits.

    Args:
        config: Supervised component configuration.
        latent_dimension: Dimensionality of the latent space input.
        num_classes: Number of target classes.
    """

    def __init__(
        self,
        config: SupervisedComponentConfig,
        latent_dimension: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.use_latent_mean = config.use_latent_mean
        self.num_classes = num_classes
        self.latent_dimension = latent_dimension

        self.classifier = build_classifier_mlp(
            input_dimension=latent_dimension,
            hidden_dimensions=config.hidden_dimensions,
            num_classes=num_classes,
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "SupervisedClassificationHead: latent_dim=%d, hidden=%s, " "num_classes=%d, params=%d",
            latent_dimension,
            config.hidden_dimensions,
            num_classes,
            total_params,
        )

    def forward(self, latent_representation: Tensor) -> Tensor:
        """Compute class logits from a latent representation.

        Args:
            latent_representation: Latent vector (mean or sample),
                shape (batch, latent_dimension).

        Returns:
            Class logits, shape (batch, num_classes).
            Note: raw logits, not probabilities. Use with CrossEntropyLoss.
        """
        logits: Tensor = self.classifier(latent_representation)
        return logits
