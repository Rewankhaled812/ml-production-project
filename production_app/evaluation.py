"""Model evaluation module.

Computes, formats, and logs standard classification metrics including
accuracy, precision, recall, F1-score, and confusion matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from production_app.config import CLASS_NAMES
from production_app.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate fitted pipeline against held-out test data.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        X_test: Test features DataFrame.
        y_test: True test target Series.

    Returns:
        Dictionary containing metrics, confusion matrix, and classification report.
    """
    logger.info("Evaluating pipeline on %d test instances...", len(X_test))

    y_pred: np.ndarray = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    cm: np.ndarray = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report_dict: dict = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )
    report_str: str = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "classification_report_str": report_str,
    }

    logger.info("Evaluation Results:")
    logger.info(" - Accuracy:  %.4f", acc)
    logger.info(" - Precision: %.4f", prec)
    logger.info(" - Recall:    %.4f", rec)
    logger.info(" - F1-Score:  %.4f", f1)
    logger.info("Confusion Matrix:\n%s", cm)

    return metrics


def format_evaluation_summary(metrics: dict[str, Any]) -> str:
    """Format evaluation metrics into a human-readable text table.

    Args:
        metrics: Dictionary produced by evaluate_model.

    Returns:
        Formatted multi-line string report.
    """
    cm = np.array(metrics["confusion_matrix"])
    lines = [
        "=" * 48,
        "          MODEL EVALUATION REPORT",
        "=" * 48,
        f"  Accuracy:  {metrics['accuracy']:.4f}",
        f"  Precision: {metrics['precision']:.4f}",
        f"  Recall:    {metrics['recall']:.4f}",
        f"  F1-Score:  {metrics['f1_score']:.4f}",
        "-" * 48,
        "  Confusion Matrix (Rows: True, Cols: Pred):",
        f"    Benign (0):    TN={cm[0, 0]:<3}  FP={cm[0, 1]:<3}",
        f"    Malignant (1): FN={cm[1, 0]:<3}  TP={cm[1, 1]:<3}",
        "-" * 48,
        "  Detailed Classification Report:",
        metrics.get("classification_report_str", "").strip(),
        "=" * 48,
    ]
    return "\n".join(lines)
