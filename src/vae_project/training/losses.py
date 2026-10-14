"""VAE loss functions: reconstruction loss, KL divergence, and total VAE loss.

The VAE objective is the Evidence Lower BOund (ELBO):

    ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

Maximizing the ELBO is equivalent to minimizing:

    total_vae_loss = reconstruction_loss + beta * kl_divergence_loss

Reconstruction loss choices:
    - BCE (Binary Cross-Entropy): Treats pixel values as Bernoulli probabilities.
      Methodologically preferred for MNIST when images are in [0, 1].
      Loss = -sum(x * log(x_hat) + (1-x) * log(1-x_hat))

    - MSE (Mean Squared Error): Treats pixel values as continuous Gaussians.
      Loss = sum((x - x_hat)^2)

KL Divergence (analytical, closed-form for Gaussian q and standard normal p):
    KL(q||p) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

Both losses are summed over dimensions then averaged over the batch,
following the canonical VAE formulation (Kingma & Welling, 2014).

References:
    Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as torch_functional
from torch import Tensor

from vae_project.config.models import LossConfig, ReconstructionLossType

logger = logging.getLogger(__name__)


@dataclass
class VAELossOutput:
    """Structured output of the VAE loss computation.

    Using a dataclass instead of a tuple prevents ambiguous positional
    access and makes each component explicitly named.

    Attributes:
        reconstruction_loss: Reconstruction term of the ELBO.
        kl_divergence_loss: KL divergence term of the ELBO.
        supervised_classification_loss: Cross-entropy loss from the optional
            supervised head. Zero when supervised component is disabled.
        total_loss: Weighted sum of all loss components.
        beta: KL weight used in this computation.
        classification_loss_weight: Supervised head loss weight.
    """

    reconstruction_loss: Tensor
    kl_divergence_loss: Tensor
    supervised_classification_loss: Tensor
    total_loss: Tensor
    beta: float
    classification_loss_weight: float

    def as_dict(self) -> dict[str, float]:
        """Return a dictionary of scalar loss values for logging.

        Returns:
            Dictionary with string keys and float scalar values.
        """
        return {
            "reconstruction_loss": self.reconstruction_loss.item(),
            "kl_divergence_loss": self.kl_divergence_loss.item(),
            "supervised_classification_loss": self.supervised_classification_loss.item(),
            "total_loss": self.total_loss.item(),
        }


def compute_reconstruction_loss(
    reconstruction: Tensor,
    original_input: Tensor,
    loss_type: ReconstructionLossType,
    reduction: str = "sum",
) -> Tensor:
    """Compute the reconstruction term of the VAE loss.

    Args:
        reconstruction: Decoder output, shape (batch, input_dim).
            Values must be in [0, 1] (Sigmoid output assumed).
        original_input: Flattened original input, shape (batch, input_dim).
            Values must be in [0, 1].
        loss_type: 'bce' for Binary Cross-Entropy, 'mse' for Mean Squared Error.
        reduction: 'sum' (sum over dims, mean over batch) or 'mean'.

    Returns:
        Scalar reconstruction loss tensor (averaged over batch).

    Note:
        'sum' reduction matches the canonical VAE ELBO formulation.
        The loss is summed over pixel dimensions and then divided by
        batch size to maintain consistent scale across batch sizes.
    """
    flat_input = original_input.view(original_input.size(0), -1)

    if loss_type == ReconstructionLossType.BCE:
        # reduction='sum' sums over all pixels; we then divide by batch size
        if reduction == "sum":
            loss = torch_functional.binary_cross_entropy(
                reconstruction, flat_input, reduction="sum"
            )
            return loss / flat_input.size(0)
        return torch_functional.binary_cross_entropy(reconstruction, flat_input, reduction="mean")

    if loss_type == ReconstructionLossType.MSE:
        if reduction == "sum":
            loss = torch_functional.mse_loss(reconstruction, flat_input, reduction="sum")
            return loss / flat_input.size(0)
        return torch_functional.mse_loss(reconstruction, flat_input, reduction="mean")

    raise ValueError(
        f"Unknown reconstruction loss type: {loss_type}. "
        f"Supported: {list(ReconstructionLossType)}"
    )


def compute_kl_divergence_loss(
    latent_mean: Tensor,
    latent_log_variance: Tensor,
) -> Tensor:
    """Compute the KL divergence between approximate posterior and standard normal prior.

    Closed-form KL divergence for:
        q(z|x) = N(mu, diag(sigma^2))
        p(z)   = N(0, I)

    Formula:
        KL = -0.5 * sum_j (1 + log_var_j - mu_j^2 - exp(log_var_j))

    The sum is over latent dimensions j. The result is averaged over the batch.

    Args:
        latent_mean: Mean of the approximate posterior, shape (batch, latent_dim).
        latent_log_variance: Log variance of the posterior, shape (batch, latent_dim).

    Returns:
        Scalar KL divergence loss (averaged over batch).

    Reference:
        Kingma & Welling (2014), Appendix B.
    """
    # Per-sample KL: sum over latent dimensions
    per_sample_kl = -0.5 * torch.sum(
        1 + latent_log_variance - latent_mean.pow(2) - latent_log_variance.exp(),
        dim=1,
    )
    # Average over the batch
    return per_sample_kl.mean()


def compute_vae_loss(
    reconstruction: Tensor,
    original_input: Tensor,
    latent_mean: Tensor,
    latent_log_variance: Tensor,
    loss_config: LossConfig,
    class_logits: Tensor | None = None,
    class_labels: Tensor | None = None,
    classification_loss_weight: float = 0.0,
) -> VAELossOutput:
    """Compute the complete VAE loss with optional supervised component.

    total_loss = reconstruction_loss + beta * kl_divergence_loss
                 [+ classification_loss_weight * supervised_classification_loss]

    Args:
        reconstruction: Decoder output, shape (batch, input_dim).
        original_input: Original input tensor (any shape, will be flattened).
        latent_mean: Posterior mean, shape (batch, latent_dim).
        latent_log_variance: Posterior log variance, shape (batch, latent_dim).
        loss_config: Loss function configuration.
        class_logits: Optional classification head output, shape (batch, num_classes).
            Required if supervised component is enabled.
        class_labels: Optional integer class labels, shape (batch,).
            Required if class_logits is provided.
        classification_loss_weight: Weight for the classification loss term.
            Should be 0.0 when supervised component is disabled.

    Returns:
        VAELossOutput with all loss components.
    """
    reconstruction_loss = compute_reconstruction_loss(
        reconstruction=reconstruction,
        original_input=original_input,
        loss_type=loss_config.reconstruction_loss_type,
        reduction=loss_config.reduction,
    )

    kl_divergence_loss = compute_kl_divergence_loss(
        latent_mean=latent_mean,
        latent_log_variance=latent_log_variance,
    )

    # Supervised component
    zero_loss = torch.zeros(1, device=reconstruction.device)
    supervised_classification_loss: Tensor = zero_loss

    if class_logits is not None and class_labels is not None:
        supervised_classification_loss = torch_functional.cross_entropy(class_logits, class_labels)

    # Total VAE loss
    total_loss = (
        reconstruction_loss
        + loss_config.beta * kl_divergence_loss
        + classification_loss_weight * supervised_classification_loss
    )

    return VAELossOutput(
        reconstruction_loss=reconstruction_loss,
        kl_divergence_loss=kl_divergence_loss,
        supervised_classification_loss=supervised_classification_loss,
        total_loss=total_loss,
        beta=loss_config.beta,
        classification_loss_weight=classification_loss_weight,
    )
