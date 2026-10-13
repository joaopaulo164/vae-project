"""VAE training loop with early stopping, checkpointing, and history recording.

Implements the standard train/validate/checkpoint loop for VAE experiments.
Supports optional supervised classification head.

Data leakage prevention:
- Checkpoint selection uses validation loss only.
- Test set is never passed to this trainer.
- Supervised labels are only used when supervised_component.enabled = True.
"""

from __future__ import annotations

import logging
import time

import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from vae_project.config.loader import config_to_dict
from vae_project.config.models import (
    ExperimentConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from vae_project.models.supervised_head import SupervisedClassificationHead
from vae_project.models.vae import VariationalAutoencoder
from vae_project.training.checkpointing import (
    CheckpointMetadata,
    ModelCheckpointer,
    ResumeState,
)
from vae_project.training.early_stopping import EarlyStopping
from vae_project.training.history import EpochRecord, TrainingHistory
from vae_project.training.losses import VAELossOutput, compute_vae_loss

logger = logging.getLogger(__name__)


def _build_optimizer(
    model_parameters: list[Tensor],
    optimizer_config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Build optimizer from configuration.

    Args:
        model_parameters: List of parameter tensors to optimize.
        optimizer_config: Optimizer configuration.

    Returns:
        Configured optimizer instance.
    """
    if optimizer_config.name.lower() == "adam":
        return optim.Adam(
            model_parameters,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            betas=optimizer_config.betas,
        )
    if optimizer_config.name.lower() == "adamw":
        return optim.AdamW(
            model_parameters,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            betas=optimizer_config.betas,
        )
    if optimizer_config.name.lower() == "sgd":
        return optim.SGD(
            model_parameters,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            momentum=0.9,
        )
    raise ValueError(f"Unknown optimizer: {optimizer_config.name}. " "Supported: adam, adamw, sgd.")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: SchedulerConfig,
) -> torch.optim.lr_scheduler._LRScheduler | None:  # type: ignore[name-defined]
    """Build LR scheduler from configuration.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_config: Scheduler configuration.

    Returns:
        Scheduler instance, or None if disabled.
    """
    if not scheduler_config.enabled:
        return None
    if scheduler_config.name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(  # type: ignore[return-value]
            optimizer,
            mode="min",
            patience=scheduler_config.patience,
            factor=scheduler_config.factor,
            min_lr=scheduler_config.min_lr,
        )
    raise ValueError(
        f"Unknown scheduler: {scheduler_config.name}. " "Supported: reduce_on_plateau."
    )


class VAETrainer:
    """Trains a VariationalAutoencoder with optional supervised head.

    Handles the complete training loop including:
    - Mini-batch gradient updates
    - Epoch-level validation
    - Early stopping
    - Best-model checkpointing
    - Training history recording

    Args:
        model: VAE model to train.
        config: Full experiment configuration.
        device: Compute device.
        checkpointer: ModelCheckpointer for saving best model.
        history: TrainingHistory for recording metrics.
        supervised_head: Optional supervised classification head.
    """

    def __init__(
        self,
        model: VariationalAutoencoder,
        config: ExperimentConfig,
        device: torch.device,
        checkpointer: ModelCheckpointer,
        history: TrainingHistory,
        supervised_head: SupervisedClassificationHead | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.checkpointer = checkpointer
        self.history = history
        self.supervised_head = supervised_head

        # Build optimizer over all trainable parameters
        all_parameters = list(model.parameters())
        if supervised_head is not None:
            all_parameters += list(supervised_head.parameters())

        self.optimizer = _build_optimizer(
            [p for p in all_parameters if p.requires_grad],
            config.optimizer,
        )
        self.scheduler = _build_scheduler(self.optimizer, config.scheduler)
        self.early_stopping = EarlyStopping(config.early_stopping)

        self._log_config()

    def _log_config(self) -> None:
        """Log training configuration."""
        if self.config.loss.beta != 1.0:
            logger.warning(
                "beta=%.4f (non-standard VAE; standard is 1.0). "
                "This deviation will be recorded in results.",
                self.config.loss.beta,
            )

        logger.info(
            "Trainer initialized: max_epochs=%d, batch_size=%d, lr=%.6f, " "supervised=%s",
            self.config.training.max_epochs,
            self.config.training.batch_size,
            self.config.optimizer.learning_rate,
            self.config.supervised_component.enabled,
        )

    def _train_epoch(
        self,
        train_loader: DataLoader,  # type: ignore[type-arg]
        epoch: int,
    ) -> VAELossOutput:
        """Run one training epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch index (0-based).

        Returns:
            Average VAELossOutput for the epoch.
        """
        self.model.train()
        if self.supervised_head is not None:
            self.supervised_head.train()

        total_reconstruction = 0.0
        total_kl = 0.0
        total_classification = 0.0
        total_loss_sum = 0.0
        num_batches = 0

        supervised_enabled = self.config.supervised_component.enabled
        classification_weight = (
            self.config.supervised_component.classification_loss_weight
            if supervised_enabled
            else 0.0
        )

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()

            vae_output = self.model(batch_images)

            class_logits: Tensor | None = None
            if supervised_enabled and self.supervised_head is not None:
                latent_input = (
                    vae_output.latent_mean
                    if self.config.supervised_component.use_latent_mean
                    else vae_output.latent_sample
                )
                class_logits = self.supervised_head(latent_input)

            loss_output = compute_vae_loss(
                reconstruction=vae_output.reconstruction,
                original_input=batch_images,
                latent_mean=vae_output.latent_mean,
                latent_log_variance=vae_output.latent_log_variance,
                loss_config=self.config.loss,
                class_logits=class_logits if supervised_enabled else None,
                class_labels=batch_labels if supervised_enabled else None,
                classification_loss_weight=classification_weight,
            )

            loss_output.total_loss.backward()
            self.optimizer.step()

            total_reconstruction += loss_output.reconstruction_loss.item()
            total_kl += loss_output.kl_divergence_loss.item()
            total_classification += loss_output.supervised_classification_loss.item()
            total_loss_sum += loss_output.total_loss.item()
            num_batches += 1

            if num_batches % self.config.training.log_interval_batches == 0:
                logger.debug(
                    "Epoch %d, batch %d: total=%.4f, recon=%.4f, kl=%.4f",
                    epoch,
                    num_batches,
                    loss_output.total_loss.item(),
                    loss_output.reconstruction_loss.item(),
                    loss_output.kl_divergence_loss.item(),
                )

        num_batches = max(num_batches, 1)
        # Return average losses using zero tensors as placeholders
        device = self.device
        return VAELossOutput(
            reconstruction_loss=torch.tensor(total_reconstruction / num_batches, device=device),
            kl_divergence_loss=torch.tensor(total_kl / num_batches, device=device),
            supervised_classification_loss=torch.tensor(
                total_classification / num_batches, device=device
            ),
            total_loss=torch.tensor(total_loss_sum / num_batches, device=device),
            beta=self.config.loss.beta,
            classification_loss_weight=classification_weight,
        )

    @torch.no_grad()
    def _validate_epoch(
        self,
        val_loader: DataLoader,  # type: ignore[type-arg]
    ) -> VAELossOutput:
        """Run validation on the validation set.

        Args:
            val_loader: Validation data loader.

        Returns:
            Average VAELossOutput over the validation set.
        """
        self.model.eval()
        if self.supervised_head is not None:
            self.supervised_head.eval()

        total_reconstruction = 0.0
        total_kl = 0.0
        total_classification = 0.0
        total_loss_sum = 0.0
        num_batches = 0

        supervised_enabled = self.config.supervised_component.enabled
        classification_weight = (
            self.config.supervised_component.classification_loss_weight
            if supervised_enabled
            else 0.0
        )

        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)

            vae_output = self.model(batch_images)

            class_logits = None
            if supervised_enabled and self.supervised_head is not None:
                latent_input = (
                    vae_output.latent_mean
                    if self.config.supervised_component.use_latent_mean
                    else vae_output.latent_sample
                )
                class_logits = self.supervised_head(latent_input)

            loss_output = compute_vae_loss(
                reconstruction=vae_output.reconstruction,
                original_input=batch_images,
                latent_mean=vae_output.latent_mean,
                latent_log_variance=vae_output.latent_log_variance,
                loss_config=self.config.loss,
                class_logits=class_logits if supervised_enabled else None,
                class_labels=batch_labels if supervised_enabled else None,
                classification_loss_weight=classification_weight,
            )

            total_reconstruction += loss_output.reconstruction_loss.item()
            total_kl += loss_output.kl_divergence_loss.item()
            total_classification += loss_output.supervised_classification_loss.item()
            total_loss_sum += loss_output.total_loss.item()
            num_batches += 1

        num_batches = max(num_batches, 1)
        device = self.device
        return VAELossOutput(
            reconstruction_loss=torch.tensor(total_reconstruction / num_batches, device=device),
            kl_divergence_loss=torch.tensor(total_kl / num_batches, device=device),
            supervised_classification_loss=torch.tensor(
                total_classification / num_batches, device=device
            ),
            total_loss=torch.tensor(total_loss_sum / num_batches, device=device),
            beta=self.config.loss.beta,
            classification_loss_weight=classification_weight,
        )

    def resume_from_checkpoint(self) -> ResumeState:
        """Restore model, optimizer, scheduler and early stopping from the best checkpoint.

        The checkpoint stores the state of the best epoch, so training continues
        at ``best_epoch + 1``; epochs run after the best one are not recovered.

        Returns:
            ResumeState; pass ``start_epoch=state.next_epoch`` to :meth:`fit`.

        Raises:
            CheckpointError: If the checkpoint lacks the state required to resume.
        """
        return self.checkpointer.load_for_resume(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            scheduler=self.scheduler,
            early_stopping=self.early_stopping,
            supervised_head=self.supervised_head,
        )

    def fit(
        self,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader,  # type: ignore[type-arg]
        dataset_name: str,
        is_quick_mode: bool = False,
        start_epoch: int = 0,
    ) -> TrainingHistory:
        """Train the VAE for the configured number of epochs.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            dataset_name: Dataset name for checkpoint metadata.
            is_quick_mode: Whether this is a quick smoke test run.
            start_epoch: 0-based epoch at which to start (non-zero after
                :meth:`resume_from_checkpoint`).

        Returns:
            The completed TrainingHistory.
        """
        logger.info(
            "Starting training: max_epochs=%d, start_epoch=%d, dataset=%s, quick_mode=%s",
            self.config.training.max_epochs,
            start_epoch,
            dataset_name,
            is_quick_mode,
        )
        resolved_config = config_to_dict(self.config)

        progress_bar = tqdm(
            range(start_epoch, self.config.training.max_epochs),
            desc="Training",
            unit="epoch",
        )

        for epoch in progress_bar:
            epoch_start_time = time.perf_counter()

            # Training pass
            train_losses = self._train_epoch(train_loader, epoch)

            # Validation pass
            val_losses = self._validate_epoch(val_loader)

            epoch_duration = time.perf_counter() - epoch_start_time

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update early stopping and LR scheduler *before* checkpointing so that
            # the stored optimizer/scheduler/early-stopping state is the one to
            # continue from at the next epoch.
            val_total = val_losses.total_loss.item()
            should_stop = self.early_stopping.step(val_total, epoch)
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_total)
                else:
                    self.scheduler.step()

            # Check if this is the best epoch
            is_best = self.checkpointer.save_if_best(
                model=self.model,
                val_loss=val_total,
                metadata=CheckpointMetadata(
                    epoch=epoch,
                    val_total_loss=val_total,
                    val_reconstruction_loss=val_losses.reconstruction_loss.item(),
                    val_kl_divergence_loss=val_losses.kl_divergence_loss.item(),
                    val_classification_loss=(val_losses.supervised_classification_loss.item()),
                    topology_name=self.model.topology_name,
                    hidden_dimensions=self.model.hidden_dimensions,
                    latent_dimension=self.model.latent_dimension,
                    dataset_name=dataset_name,
                    is_quick_mode=is_quick_mode,
                    run_id=self.checkpointer.run_id,
                    seed=self.config.reproducibility.global_seed,
                    experiment_type=self.config.experiment_type,
                    learning_rate=current_lr,
                    epochs_without_improvement=self.early_stopping.patience_counter,
                ),
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
                supervised_head=self.supervised_head,
                resolved_config=resolved_config,
            )

            # Record epoch
            record = EpochRecord(
                epoch=epoch,
                train_reconstruction_loss=train_losses.reconstruction_loss.item(),
                train_kl_divergence_loss=train_losses.kl_divergence_loss.item(),
                train_total_loss=train_losses.total_loss.item(),
                val_reconstruction_loss=val_losses.reconstruction_loss.item(),
                val_kl_divergence_loss=val_losses.kl_divergence_loss.item(),
                val_total_loss=val_total,
                train_classification_loss=train_losses.supervised_classification_loss.item(),
                val_classification_loss=val_losses.supervised_classification_loss.item(),
                learning_rate=current_lr,
                epoch_duration_seconds=epoch_duration,
                is_best_epoch=is_best,
                early_stopping_counter=self.early_stopping.patience_counter,
            )
            self.history.append(record)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "train_loss": f"{train_losses.total_loss.item():.4f}",
                    "val_loss": f"{val_total:.4f}",
                    "best": "✓" if is_best else "",
                }
            )

            logger.info(
                "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, "
                "train_recon=%.4f, train_kl=%.4f, "
                "val_recon=%.4f, val_kl=%.4f, lr=%.6f, duration=%.1fs",
                epoch + 1,
                self.config.training.max_epochs,
                train_losses.total_loss.item(),
                val_total,
                train_losses.reconstruction_loss.item(),
                train_losses.kl_divergence_loss.item(),
                val_losses.reconstruction_loss.item(),
                val_losses.kl_divergence_loss.item(),
                current_lr,
                epoch_duration,
            )

            # Early stopping check (state already updated above)
            if should_stop:
                logger.info("Training stopped early at epoch %d.", epoch)
                break

        logger.info(
            "Training complete: %d epochs, best epoch=%d, best_val_loss=%.6f",
            self.history.num_epochs_completed,
            self.early_stopping.best_epoch,
            self.early_stopping.best_value,
        )

        return self.history
