"""Command-line interface for the VAE project.

All operations are accessible via:
    python -m vae_project.cli <command> [options]

Available commands:
    validate-config         Validate a YAML configuration file
    train                   Train a VAE model
    select-topology         Run topology selection experiment
    compare-latent-dims     Compare multiple latent dimensions
    run-pipeline            Run the full experiment pipeline
    smoke-test              Run a quick smoke test with synthetic data
    extract-latents         Extract latent representations from a checkpoint
    generate-projections    Generate dimensionality reduction projections
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the main CLI argument parser with all subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m vae_project.cli",
        description="VAE Project â€” Variational Autoencoder experiments for MNIST and Fashion-MNIST",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate-config
    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate a YAML configuration file without running experiments",
    )
    validate_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file",
    )

    # train
    train_parser = subparsers.add_parser(
        "train",
        help="Train a VAE with the specified configuration",
    )
    train_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file",
    )

    # select-topology
    select_topo_parser = subparsers.add_parser(
        "select-topology",
        help="Run topology selection experiment",
    )
    select_topo_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file",
    )
    select_topo_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: derived from config artifacts path)",
    )

    # compare-latent-dims
    compare_parser = subparsers.add_parser(
        "compare-latent-dims",
        help="Compare multiple latent dimensions with fixed topology",
    )
    compare_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file",
    )
    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )

    # run-pipeline
    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run the full experiment pipeline (train + evaluate + visualize)",
    )
    pipeline_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file",
    )

    # smoke-test
    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Run a quick smoke test (uses synthetic data, no downloads required)",
    )
    smoke_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smoke_test.yaml"),
        help="Path to smoke test configuration (default: configs/smoke_test.yaml)",
    )

    return parser


def _cmd_validate_config(args: argparse.Namespace) -> int:
    """Handle the validate-config command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    from vae_project.config.loader import ConfigurationError, load_config

    print(f"Validating configuration: {args.config}")

    try:
        config = load_config(args.config)
        print("âœ“ Configuration is valid.")
        print(f"  Dataset: {config.dataset.name.value}")
        print(
            f"  Topology: {config.model.topology.name} = {config.model.topology.hidden_dimensions}"
        )
        print(f"  Latent dimension: {config.model.latent_dimension}")
        print(f"  Quick mode: {config.project.quick_mode}")
        return 0
    except FileNotFoundError as exc:
        print(f"âœ— File not found: {exc}", file=sys.stderr)
        return 1
    except ConfigurationError as exc:
        print(f"âœ— Configuration error:\n{exc}", file=sys.stderr)
        return 1


def _cmd_train(args: argparse.Namespace) -> int:
    """Handle the train command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    from vae_project.config.loader import ConfigurationError, load_config
    from vae_project.experiments.runner import ExperimentRunner

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ConfigurationError) as exc:
        print(f"âœ— Configuration error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Starting training: dataset={config.dataset.name.value}, "
        f"latent_dim={config.model.latent_dimension}"
    )

    try:
        runner = ExperimentRunner(config)
        run_dir = runner.run()
        print(f"âœ“ Training complete. Artifacts at: {run_dir}")
        return 0
    except Exception as exc:
        print(f"âœ— Training failed: {exc}", file=sys.stderr)
        logger.exception("Training failed with exception")
        return 1


def _cmd_select_topology(args: argparse.Namespace) -> int:
    """Handle the select-topology command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    from vae_project.config.loader import ConfigurationError, load_config
    from vae_project.experiments.topology_selection import run_topology_selection

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ConfigurationError) as exc:
        print(f"âœ— Configuration error: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or (
        config.paths.artifacts_dir / config.dataset.name.value / "topology_selection"
    )

    print(f"Running topology selection for {config.dataset.name.value}...")

    try:
        results_df = run_topology_selection(config, output_dir)
        print("Topology selection results (sorted by val_total_loss):")
        print(results_df.to_string(index=False))
        if not results_df.empty:
            best = results_df.iloc[0]
            print(
                f"\nâœ“ Best topology: {best['topology_name']} "
                f"(val_loss={best['best_val_total_loss']:.6f})"
            )
        return 0
    except Exception as exc:
        print(f"âœ— Topology selection failed: {exc}", file=sys.stderr)
        logger.exception("Topology selection failed")
        return 1


def _cmd_compare_latent_dims(args: argparse.Namespace) -> int:
    """Handle the compare-latent-dims command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    from vae_project.config.loader import ConfigurationError, load_config
    from vae_project.experiments.latent_dimension_comparison import (
        run_latent_dimension_comparison,
    )

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ConfigurationError) as exc:
        print(f"âœ— Configuration error: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or (
        config.paths.artifacts_dir / config.dataset.name.value / "latent_dimension_comparison"
    )

    dims = config.latent_dimension_comparison.dimensions
    print(f"Comparing latent dimensions {dims} for {config.dataset.name.value}...")

    try:
        results_df = run_latent_dimension_comparison(config, output_dir)
        print("Latent dimension comparison results:")
        print(results_df.to_string(index=False))
        return 0
    except Exception as exc:
        print(f"âœ— Comparison failed: {exc}", file=sys.stderr)
        logger.exception("Dimension comparison failed")
        return 1


def _cmd_run_pipeline(args: argparse.Namespace) -> int:
    """Handle the run-pipeline command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    return _cmd_train(args)


def _cmd_smoke_test(args: argparse.Namespace) -> int:
    """Handle the smoke-test command using synthetic data.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    import torch
    from torch.utils.data import TensorDataset

    print("Running smoke test with synthetic data...")
    print("(This test does NOT require internet access or dataset downloads)")

    try:
        # Create synthetic MNIST-like data
        n_train = 100
        n_val = 20
        n_test = 20
        image_size = 784
        n_classes = 10

        torch.manual_seed(42)
        train_images = torch.rand(n_train, 1, 28, 28)
        train_labels = torch.randint(0, n_classes, (n_train,))
        val_images = torch.rand(n_val, 1, 28, 28)
        val_labels = torch.randint(0, n_classes, (n_val,))
        test_images = torch.rand(n_test, 1, 28, 28)
        test_labels = torch.randint(0, n_classes, (n_test,))

        train_dataset = TensorDataset(train_images, train_labels)
        val_dataset = TensorDataset(val_images, val_labels)
        test_dataset = TensorDataset(test_images, test_labels)

        from torch.utils.data import DataLoader

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        DataLoader(val_dataset, batch_size=32)
        DataLoader(test_dataset, batch_size=32)

        # Build minimal model
        from vae_project.config.models import (
            LossConfig,
            ModelConfig,
            TopologyConfig,
        )
        from vae_project.models.vae import VariationalAutoencoder
        from vae_project.training.losses import compute_vae_loss

        model_config = ModelConfig(
            latent_dimension=4,
            topology=TopologyConfig(name="small", hidden_dimensions=[64, 32]),
            input_dimension=image_size,
        )
        model = VariationalAutoencoder(model_config)
        loss_config = LossConfig()

        # Run one training step
        model.train()
        batch_images, _ = next(iter(train_loader))
        vae_out = model(batch_images)
        loss_out = compute_vae_loss(
            reconstruction=vae_out.reconstruction,
            original_input=batch_images,
            latent_mean=vae_out.latent_mean,
            latent_log_variance=vae_out.latent_log_variance,
            loss_config=loss_config,
        )

        if not (loss_out.total_loss.item() > 0):
            raise ValueError("Smoke test: Loss should be positive")
        if vae_out.reconstruction.shape != (32, image_size):
            raise ValueError(
                f"Smoke test: unexpected reconstruction shape {vae_out.reconstruction.shape}"
            )
        if vae_out.latent_mean.shape != (32, 4):
            raise ValueError(
                f"Smoke test: unexpected latent mean shape {vae_out.latent_mean.shape}"
            )
        if vae_out.latent_log_variance.shape != (32, 4):
            raise ValueError(
                f"Smoke test: unexpected log variance shape {vae_out.latent_log_variance.shape}"
            )

        print("âœ“ Smoke test passed:")
        print("  - VAE forward pass: OK")
        print(f"  - Reconstruction shape: {vae_out.reconstruction.shape}")
        print(f"  - Latent mean shape: {vae_out.latent_mean.shape}")
        print(f"  - Total loss: {loss_out.total_loss.item():.4f}")
        print(f"  - Reconstruction loss: {loss_out.reconstruction_loss.item():.4f}")
        print(f"  - KL divergence loss: {loss_out.kl_divergence_loss.item():.4f}")
        print("")
        print("NOTE: This was a smoke test with SYNTHETIC data.")
        print(
            "      For real experiments, run: python -m vae_project.cli run-pipeline --config configs/mnist_unsupervised.yaml"
        )
        return 0

    except Exception as exc:
        print(f"âœ— Smoke test FAILED: {exc}", file=sys.stderr)
        logger.exception("Smoke test failed")
        return 1


def main() -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # Configure basic logging for CLI
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    command_handlers = {
        "validate-config": _cmd_validate_config,
        "train": _cmd_train,
        "select-topology": _cmd_select_topology,
        "compare-latent-dims": _cmd_compare_latent_dims,
        "run-pipeline": _cmd_run_pipeline,
        "smoke-test": _cmd_smoke_test,
    }

    handler = command_handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)

    exit_code = handler(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
