"""Core experiment runner: orchestrates the complete VAE experiment pipeline.

Coordinates:
1. Configuration loading and validation
2. Reproducibility setup (seeding)
3. Data loading and splitting
4. Model construction
5. Training with early stopping and checkpointing
6. Evaluation (validation and optional final test)
7. Projection and visualization
8. Artifact generation and manifesting

Usage::

    runner = ExperimentRunner(config)
    runner.run()
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

from vae_project.config.loader import save_resolved_config
from vae_project.config.models import ExperimentConfig
from vae_project.data.datasets import get_raw_datasets
from vae_project.data.loaders import DataLoaders, create_data_loaders
from vae_project.data.splits import DataSplits, create_data_splits, save_class_distribution
from vae_project.evaluation.clustering_metrics import compute_clustering_metrics
from vae_project.evaluation.latent_extraction import extract_latent_representations
from vae_project.models.supervised_head import SupervisedClassificationHead
from vae_project.models.vae import VariationalAutoencoder
from vae_project.projections.pca import PCAProjection
from vae_project.projections.sampling import stratified_subsample
from vae_project.projections.tsne import TSNEProjection
from vae_project.projections.umap_projection import UMAPProjection
from vae_project.reporting.artifact_manifest import ArtifactManifest
from vae_project.reporting.run_summary import generate_run_summary
from vae_project.training.checkpointing import ModelCheckpointer
from vae_project.training.history import TrainingHistory
from vae_project.training.trainer import VAETrainer
from vae_project.utils.device import get_device
from vae_project.utils.environment import collect_environment_info
from vae_project.utils.logging import setup_logging
from vae_project.utils.paths import build_run_dir, create_run_structure
from vae_project.utils.reproducibility import set_global_seed
from vae_project.utils.timing import Timer
from vae_project.visualization.latent_plots import (
    plot_latent_space_direct_2d,
    plot_pca_explained_variance,
    plot_pca_projection,
    plot_tsne_projection,
    plot_umap_projection,
)
from vae_project.visualization.plot_settings import apply_plot_settings
from vae_project.visualization.reconstruction_plots import (
    plot_generated_samples,
    plot_latent_traversal_2d,
    plot_reconstruction_comparison,
)
from vae_project.visualization.training_plots import (
    plot_training_loss_components,
    plot_training_total_loss,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the complete VAE experiment pipeline.

    Args:
        config: Fully validated ExperimentConfig.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._run_dir: Path | None = None
        self._run_dirs: dict[str, Path] | None = None
        self._manifest: ArtifactManifest | None = None

    def run(self) -> Path:
        """Execute the full experiment pipeline.

        Returns:
            Path to the run directory containing all artifacts.
        """
        with Timer() as total_timer:
            self._setup_run_directory()
            self._setup_logging_for_run()
            logger.info("Experiment started: %s", self._run_dir)

            set_global_seed(
                self.config.reproducibility.global_seed,
                deterministic=self.config.reproducibility.deterministic_mode,
            )

            device = get_device(
                self.config.runtime.device,
                fallback_to_cpu=self.config.runtime.cuda_fallback_to_cpu,
            )

            # Save environment info
            env_info = collect_environment_info()
            self._save_environment_info(env_info)

            # Data
            train_full, test_dataset = get_raw_datasets(
                dataset_name=self.config.dataset.name,
                data_dir=self.config.paths.data_dir,
                download=self.config.dataset.download,
            )

            splits = create_data_splits(
                train_full_dataset=train_full,
                test_dataset=test_dataset,
                dataset_name=self.config.dataset.name,
                splits_config=self.config.splits,
                quick_mode=self.config.project.quick_mode,
            )

            # Save class distribution
            save_class_distribution(
                splits.class_distribution,
                self._run_dirs["tables"] / "class_distribution.csv",  # type: ignore[index]
            )

            loaders = create_data_loaders(
                splits=splits,
                training_config=self.config.training,
                dataset_config=self.config.dataset,
            )

            # Model
            model = VariationalAutoencoder(self.config.model).to(device)

            supervised_head: SupervisedClassificationHead | None = None
            if self.config.supervised_component.enabled:
                supervised_head = SupervisedClassificationHead(
                    config=self.config.supervised_component,
                    latent_dimension=self.config.model.latent_dimension,
                    num_classes=splits.num_classes,
                ).to(device)
                logger.info("Supervised component ENABLED " "(results will be stored separately)")

            # Training
            checkpointer = ModelCheckpointer(
                checkpoint_dir=self._run_dirs["checkpoints"],  # type: ignore[index]
                run_id=self._run_dir.name if self._run_dir else "",  # type: ignore[union-attr]
            )
            history = TrainingHistory(self._run_dirs["tables"])  # type: ignore[index]

            trainer = VAETrainer(
                model=model,
                config=self.config,
                device=device,
                checkpointer=checkpointer,
                history=history,
                supervised_head=supervised_head,
            )

            apply_plot_settings(self.config.plotting)

            with Timer() as train_timer:
                trainer.fit(
                    train_loader=loaders.train,
                    val_loader=loaders.val,
                    dataset_name=self.config.dataset.name.value,
                    is_quick_mode=self.config.project.quick_mode,
                )

            logger.info("Training completed in %s", train_timer)

            # Save training history
            history.save()

            # Generate training plots
            history_df = pd.DataFrame([record.__dict__ for record in history.get_records()])
            self._generate_training_plots(history_df)

            # Load best checkpoint
            best_metadata = checkpointer.load_best(model=model, device=device)
            logger.info(
                "Best checkpoint loaded: epoch=%d, val_loss=%.6f",
                best_metadata.epoch,
                best_metadata.val_total_loss,
            )

            # Reconstructions and generated samples
            self._generate_reconstruction_figures(model, loaders, device, splits)

            # Latent extraction and projections
            self._run_latent_analysis(model, loaders, device, splits)

            # Save resolved config
            if self.config.artifacts.save_resolved_config:
                config_path = self._run_dir / "config_resolved.yaml"  # type: ignore[operator]
                save_resolved_config(self.config, config_path)

            # Generate run summary
            summary_path = generate_run_summary(
                run_dir=self._run_dir,  # type: ignore[arg-type]
                config=self.config,
                history=history,
                best_epoch=best_metadata.epoch,
                best_val_loss=best_metadata.val_total_loss,
                training_duration=train_timer.elapsed_seconds,
                device_info={"device": str(device)},
            )

            logger.info("Run summary: %s", summary_path)

        logger.info("Total experiment time: %s", total_timer)
        logger.info("Artifacts at: %s", self._run_dir)

        return self._run_dir  # type: ignore[return-value]

    def _setup_run_directory(self) -> None:
        """Create the run directory structure."""
        experiment_type = (
            "supervised" if self.config.supervised_component.enabled else "unsupervised"
        )
        self._run_dir = build_run_dir(
            artifacts_dir=self.config.paths.artifacts_dir,
            dataset_name=self.config.dataset.name.value,
            experiment_type=experiment_type,
            latent_dimension=self.config.model.latent_dimension,
            run_name=self.config.run_name,
        )
        self._run_dirs = create_run_structure(self._run_dir)

    def _setup_logging_for_run(self) -> None:
        """Configure logging for this specific run."""
        if self._run_dirs is None:
            return
        log_file = self._run_dirs["logs"] / "experiment.log"
        setup_logging(
            log_config=self.config.logging,
            log_file=log_file,
        )

    def _save_environment_info(self, env_info: dict) -> None:
        """Save environment metadata to JSON."""
        import json

        if self._run_dir is None:
            return
        env_path = self._run_dir / "environment.json"
        with env_path.open("w", encoding="utf-8") as env_file:
            json.dump(env_info, env_file, indent=2, default=str)
        logger.debug("Environment info saved: %s", env_path)

    def _generate_training_plots(self, history_df: pd.DataFrame) -> None:
        """Generate training curve figures.

        Args:
            history_df: DataFrame with per-epoch training records.
        """
        if self._run_dirs is None:
            return

        figures_dir = self._run_dirs["figures_training"]
        dataset_name = self.config.dataset.name.value
        is_quick = self.config.project.quick_mode

        plot_training_total_loss(
            history_df=history_df,
            output_dir=figures_dir,
            config=self.config.plotting,
            dataset_name=dataset_name,
            is_quick_mode=is_quick,
        )

        plot_training_loss_components(
            history_df=history_df,
            output_dir=figures_dir,
            config=self.config.plotting,
            dataset_name=dataset_name,
            is_quick_mode=is_quick,
        )

    def _generate_reconstruction_figures(
        self,
        model: VariationalAutoencoder,
        loaders: DataLoaders,
        device: torch.device,
        splits: DataSplits,
    ) -> None:
        """Generate reconstruction comparison and generated sample figures."""
        if self._run_dirs is None:
            return

        plot_reconstruction_comparison(
            model=model,
            data_loader=loaders.val,
            device=device,
            output_dir=self._run_dirs["figures_reconstructions"],
            config=self.config.plotting,
            dataset_name=self.config.dataset.name.value,
            split_name="val",
            class_names=splits.class_names,
            is_quick_mode=self.config.project.quick_mode,
            seed=self.config.reproducibility.global_seed,
        )

        plot_generated_samples(
            model=model,
            device=device,
            output_dir=self._run_dirs["figures_generated"],
            config=self.config.plotting,
            dataset_name=self.config.dataset.name.value,
            is_quick_mode=self.config.project.quick_mode,
            seed=self.config.reproducibility.global_seed,
        )

        if self.config.model.latent_dimension == 2:
            plot_latent_traversal_2d(
                model=model,
                device=device,
                output_dir=self._run_dirs["figures_latent"],
                config=self.config.plotting,
                dataset_name=self.config.dataset.name.value,
                is_quick_mode=self.config.project.quick_mode,
            )

    def _run_latent_analysis(
        self,
        model: VariationalAutoencoder,
        loaders: DataLoaders,
        device: torch.device,
        splits: DataSplits,
    ) -> None:
        """Extract latent representations and generate projections."""
        if self._run_dirs is None:
            return

        val_reps = extract_latent_representations(
            model=model,
            data_loader=loaders.val,
            device=device,
            split_name="val",
            class_names=splits.class_names,
            dataset_name=self.config.dataset.name.value,
            is_quick_mode=self.config.project.quick_mode,
        )
        val_reps.save(
            output_dir=self._run_dirs["latent_data"],
            save_npz=self.config.artifacts.save_latent_npz,
        )

        latent_dim = self.config.model.latent_dimension
        dataset_name = self.config.dataset.name.value
        figures_latent = self._run_dirs["figures_latent"]
        is_quick = self.config.project.quick_mode
        proj_config = self.config.projections
        plot_config = self.config.plotting

        if latent_dim == 2:
            plot_latent_space_direct_2d(
                latent_means=val_reps.latent_means,
                labels=val_reps.labels,
                class_names=splits.class_names,
                output_dir=figures_latent,
                config=plot_config,
                dataset_name=dataset_name,
                split_name="val",
                is_quick_mode=is_quick,
            )
        else:
            # Subsample for projection
            sample_cfg = proj_config.sampling
            sample_vecs, sample_labels, sample_indices = stratified_subsample(
                vectors=val_reps.latent_means,
                labels=val_reps.labels,
                max_samples=sample_cfg.max_samples,
                seed=sample_cfg.seed,
            )

            # Save sample indices
            import pandas as pd

            indices_df = pd.DataFrame({"sample_index": sample_indices})
            indices_path = self._run_dirs["latent_data"] / "projection_sample_indices.csv"
            indices_df.to_csv(indices_path, index=False)

            if "pca" in proj_config.methods:
                pca = PCAProjection(n_components=2)
                pca_result = pca.fit_transform(sample_vecs)
                pca_result.save_explained_variance_csv(
                    self._run_dirs["tables"] / "pca_explained_variance.csv"
                )
                plot_pca_projection(
                    pca_result=pca_result,
                    labels=sample_labels,
                    class_names=splits.class_names,
                    output_dir=figures_latent,
                    config=plot_config,
                    dataset_name=dataset_name,
                    is_quick_mode=is_quick,
                )
                plot_pca_explained_variance(
                    pca_result=pca_result,
                    output_dir=figures_latent,
                    config=plot_config,
                    dataset_name=dataset_name,
                    is_quick_mode=is_quick,
                )

            if "umap" in proj_config.methods:
                umap_proj = UMAPProjection(proj_config.umap)
                umap_result = umap_proj.fit_transform(sample_vecs)
                plot_umap_projection(
                    umap_result=umap_result,
                    labels=sample_labels,
                    class_names=splits.class_names,
                    output_dir=figures_latent,
                    config=plot_config,
                    dataset_name=dataset_name,
                    is_quick_mode=is_quick,
                )

            if "tsne" in proj_config.methods:
                tsne_proj = TSNEProjection(proj_config.tsne)
                tsne_result = tsne_proj.fit_transform(sample_vecs)
                plot_tsne_projection(
                    tsne_result=tsne_result,
                    labels=sample_labels,
                    class_names=splits.class_names,
                    output_dir=figures_latent,
                    config=plot_config,
                    dataset_name=dataset_name,
                    is_quick_mode=is_quick,
                )

        # Subsidiary metrics
        metrics_config = self.config.metrics
        clustering_result = compute_clustering_metrics(
            latent_vectors=val_reps.latent_means,
            labels=val_reps.labels,
            space_name="val_latent_original",
            num_classes=splits.num_classes,
            max_samples_silhouette=metrics_config.silhouette_sample_size,
        )

        # Save metrics table
        import pandas as pd

        metrics_df = pd.DataFrame([clustering_result.as_dict()])
        metrics_path = self._run_dirs["tables"] / "projection_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Subsidiary metrics saved: %s", metrics_path)
