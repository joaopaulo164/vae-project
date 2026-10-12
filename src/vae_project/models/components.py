"""Reusable MLP building blocks for encoder and decoder networks.

Provides factory functions that build sequential MLP blocks from
lists of hidden layer dimensions. This avoids duplicating topology
construction logic across VAE variants.
"""

from __future__ import annotations

import torch.nn as nn


def build_mlp_encoder(
    input_dimension: int,
    hidden_dimensions: list[int],
) -> nn.Sequential:
    """Build the encoder MLP body (without the latent projection layers).

    Builds a sequence of Linear -> ReLU layers from input_dimension
    through the hidden_dimensions. The output feeds into separate
    mu and log_var projection layers.

    Args:
        input_dimension: Flattened input size (e.g. 784 for 28x28 images).
        hidden_dimensions: List of hidden layer sizes, e.g. [512, 256].

    Returns:
        nn.Sequential module: input_dimension -> hidden_dimensions[-1].

    Example:
        >>> encoder = build_mlp_encoder(784, [512, 256])
        >>> # encoder: 784 -> 512 -> 256
    """
    if not hidden_dimensions:
        raise ValueError("hidden_dimensions must not be empty.")

    layers: list[nn.Module] = []
    current_dim = input_dimension

    for hidden_dim in hidden_dimensions:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim

    return nn.Sequential(*layers)


def build_mlp_decoder(
    latent_dimension: int,
    hidden_dimensions: list[int],
    output_dimension: int,
) -> nn.Sequential:
    """Build the decoder MLP from latent space to reconstructed output.

    Builds a sequence of Linear -> ReLU layers in reverse order of
    hidden_dimensions, followed by a final Linear -> Sigmoid to produce
    values in [0, 1] matching the normalized input range.

    Args:
        latent_dimension: Dimensionality of the latent space.
        hidden_dimensions: Hidden layer dimensions in encoder order (reversed here).
            E.g. [512, 256] produces decoder: latent -> 256 -> 512 -> output.
        output_dimension: Reconstructed output size (e.g. 784).

    Returns:
        nn.Sequential module: latent_dimension -> ... -> output_dimension.

    Example:
        >>> decoder = build_mlp_decoder(16, [512, 256], 784)
        >>> # decoder: 16 -> 256 -> 512 -> 784 (sigmoid)
    """
    if not hidden_dimensions:
        raise ValueError("hidden_dimensions must not be empty.")

    layers: list[nn.Module] = []
    current_dim = latent_dimension

    # Decoder uses reversed hidden dimensions for symmetry
    for hidden_dim in reversed(hidden_dimensions):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim

    # Final projection to output with Sigmoid for [0, 1] range
    layers.append(nn.Linear(current_dim, output_dimension))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def build_classifier_mlp(
    input_dimension: int,
    hidden_dimensions: list[int],
    num_classes: int,
) -> nn.Sequential:
    """Build a classification MLP head.

    Args:
        input_dimension: Input feature dimension (e.g. latent dimension).
        hidden_dimensions: Hidden layer dimensions.
        num_classes: Number of output classes.

    Returns:
        nn.Sequential: input -> hidden -> num_classes (logits).
    """
    layers: list[nn.Module] = []
    current_dim = input_dimension

    for hidden_dim in hidden_dimensions:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, num_classes))
    # No softmax here — CrossEntropyLoss expects raw logits

    return nn.Sequential(*layers)
