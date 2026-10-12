"""Simple linear classifier over latent representations (subsidiary metric).

Trains a logistic regression on latent_mean representations to provide
a quantitative measure of linear separability in the latent space.

This is a SUBSIDIARY metric only. High classification accuracy does not
prove that natural clusters exist; it only shows that a linear boundary
can separate the classes in the latent space.

Protocol:
- Train ONLY on training split representations.
- Hyperparameter selection uses validation split only.
- Test set evaluation occurs only after all model/config decisions are made.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class LinearClassifierResult:
    """Results from linear classifier evaluation on latent representations.

    Attributes:
        train_accuracy: Accuracy on the training set representations.
        val_accuracy: Accuracy on the validation set representations.
        test_accuracy: Accuracy on the test set (only after final model selection).
        confusion_matrix_array: Confusion matrix (test set when available, else val).
        classification_report_str: Per-class precision/recall/F1 string.
        class_names: Class name list for labeling.
        space_name: Identifier for the latent space used.
        is_test_evaluation: Whether test accuracy was computed.
    """

    train_accuracy: float
    val_accuracy: float
    test_accuracy: float | None
    confusion_matrix_array: np.ndarray
    classification_report_str: str
    class_names: list[str]
    space_name: str
    is_test_evaluation: bool

    def as_dict(self) -> dict[str, object]:
        """Return scalar metrics as a dictionary."""
        return {
            "space_name": self.space_name,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "test_accuracy": self.test_accuracy,
            "is_test_evaluation": self.is_test_evaluation,
        }


def train_linear_classifier(
    train_latents: np.ndarray,
    train_labels: np.ndarray,
    val_latents: np.ndarray,
    val_labels: np.ndarray,
    class_names: list[str],
    space_name: str = "latent_space",
    test_latents: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LinearClassifierResult:
    """Train a logistic regression classifier on latent representations.

    Args:
        train_latents: Training latent vectors, shape (N_train, D).
        train_labels: Training integer labels, shape (N_train,).
        val_latents: Validation latent vectors, shape (N_val, D).
        val_labels: Validation integer labels, shape (N_val,).
        class_names: List of class name strings.
        space_name: Identifier for the latent space.
        test_latents: Optional test latent vectors (final evaluation only).
        test_labels: Optional test labels.
        max_iter: Maximum iterations for logistic regression.
        random_state: Random state for reproducibility.

    Returns:
        LinearClassifierResult with accuracy metrics and confusion matrix.
    """
    # Standardize features for logistic regression stability
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_latents)
    val_scaled = scaler.transform(val_latents)

    classifier = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        multi_class="multinomial",
        solver="lbfgs",
    )
    classifier.fit(train_scaled, train_labels)

    train_accuracy = float(accuracy_score(train_labels, classifier.predict(train_scaled)))
    val_accuracy = float(accuracy_score(val_labels, classifier.predict(val_scaled)))

    logger.info(
        "Linear classifier (%s): train_acc=%.4f, val_acc=%.4f",
        space_name,
        train_accuracy,
        val_accuracy,
    )

    # Test evaluation (only after final selection — caller's responsibility)
    test_accuracy: float | None = None
    eval_latents = val_scaled
    eval_labels = val_labels
    is_test = False

    if test_latents is not None and test_labels is not None:
        test_scaled = scaler.transform(test_latents)
        test_accuracy = float(accuracy_score(test_labels, classifier.predict(test_scaled)))
        eval_latents = test_scaled
        eval_labels = test_labels
        is_test = True
        logger.info(
            "Linear classifier (%s) TEST accuracy: %.4f",
            space_name,
            test_accuracy,
        )

    # Confusion matrix and report on evaluation set
    predictions = classifier.predict(eval_latents)
    conf_matrix = confusion_matrix(eval_labels, predictions)
    report = classification_report(
        eval_labels, predictions, target_names=class_names, zero_division=0
    )

    return LinearClassifierResult(
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        test_accuracy=test_accuracy,
        confusion_matrix_array=conf_matrix,
        classification_report_str=report,
        class_names=class_names,
        space_name=space_name,
        is_test_evaluation=is_test,
    )


def save_confusion_matrix_csv(
    result: LinearClassifierResult,
    output_path: Path,
) -> None:
    """Save the confusion matrix as a labeled CSV.

    Args:
        result: LinearClassifierResult with confusion matrix.
        output_path: Path for the output CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        result.confusion_matrix_array,
        index=[f"true_{name}" for name in result.class_names],
        columns=[f"pred_{name}" for name in result.class_names],
    )
    df.to_csv(output_path)
    logger.info("Confusion matrix saved: %s", output_path)
