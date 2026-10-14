"""Model persistence and inference service module.

Handles joblib serialization, artifact loading, input validation, and inference
without retraining.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from production_app.config import FEATURE_NAMES, get_config
from production_app.logging_config import get_logger
from production_app.preprocessing import validate_features

logger = get_logger(__name__)


def save_model(pipeline: Pipeline, destination_path: str | Path) -> Path:
    """Serialize and save the fitted Pipeline artifact using joblib.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        destination_path: Target filesystem path for the artifact.

    Returns:
        Resolved Path where the model was written.

    Raises:
        TypeError: If the object is not a Pipeline or estimator.
        ValueError: If the estimator is not fitted.
        OSError: If destination directory cannot be created.
    """
    if not isinstance(pipeline, Pipeline):
        logger.error("Expected Pipeline instance, got %s", type(pipeline))
        msg = f"Object to save must be a Pipeline, got {type(pipeline).__name__}"
        raise TypeError(msg)

    # Validate that pipeline is fitted
    try:
        check_is_fitted(pipeline.named_steps["classifier"])
    except Exception as exc:
        logger.error("Cannot save unfitted pipeline: %s", exc)
        raise ValueError(f"Pipeline is not fitted: {exc}") from exc

    dest = Path(destination_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Serializing and saving pipeline to '%s'...", dest)
    joblib.dump(pipeline, dest)

    size_bytes = os.path.getsize(dest)
    logger.info("Successfully persisted model artifact (%d bytes).", size_bytes)
    return dest


def load_model(artifact_path: str | Path) -> Pipeline:
    """Load a serialized Pipeline artifact from disk without retraining.

    Args:
        artifact_path: Path to the joblib artifact.

    Returns:
        Loaded scikit-learn Pipeline ready for inference.

    Raises:
        FileNotFoundError: If the specified artifact path does not exist.
        ValueError: If the loaded artifact is not a valid Pipeline.
    """
    path = Path(artifact_path).resolve()
    if not path.is_file():
        error_msg = (
            f"Model artifact not found at '{path}'. "
            "Please train and save the pipeline first before predicting."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("Loading model artifact from '%s'...", path)
    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        logger.error("Failed to deserialize artifact from '%s': %s", path, exc)
        raise ValueError(f"Corrupted or invalid model artifact: {exc}") from exc

    if not hasattr(pipeline, "predict"):
        msg = f"Loaded object {type(pipeline)} does not implement predict()."
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Successfully loaded model artifact.")
    return pipeline


def prepare_inference_features(
    input_data: pd.DataFrame | np.ndarray | dict[str, Any] | list[Any],
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Convert various input formats into a validated feature DataFrame.

    Args:
        input_data: Features as DataFrame, 2D ndarray, dict, or list of numbers.
        feature_names: Expected list of feature names.

    Returns:
        Validated pd.DataFrame conforming to the expected schema.

    Raises:
        ValueError: If input format is invalid or dimensions do not match.
    """
    expected = feature_names or FEATURE_NAMES

    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    elif isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        if len(input_data) == 0:
            raise ValueError("Input list is empty.")
        if isinstance(input_data[0], dict):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data[0], int | float):
            if len(input_data) != len(expected):
                msg = (
                    f"Feature vector length {len(input_data)} does not match "
                    f"expected {len(expected)} features."
                )
                raise ValueError(msg)
            df = pd.DataFrame([input_data], columns=expected)
        elif isinstance(input_data[0], list | np.ndarray):
            df = pd.DataFrame(input_data, columns=expected)
        else:
            raise ValueError(f"Unsupported list item type: {type(input_data[0])}")
    elif isinstance(input_data, np.ndarray):
        if input_data.ndim == 1:
            if len(input_data) != len(expected):
                msg = (
                    f"1D array length {len(input_data)} does not match "
                    f"{len(expected)} expected features."
                )
                raise ValueError(msg)
            df = pd.DataFrame([input_data], columns=expected)
        elif input_data.ndim == 2:
            if input_data.shape[1] != len(expected):
                msg = (
                    f"2D array column count {input_data.shape[1]} does not "
                    f"match {len(expected)} expected features."
                )
                raise ValueError(msg)
            df = pd.DataFrame(input_data, columns=expected)
        else:
            raise ValueError(f"Unsupported array dimensionality: {input_data.ndim}D")
    else:
        raise ValueError(f"Unsupported input type for inference: {type(input_data)}")

    return validate_features(df, expected)


def predict(
    pipeline: Pipeline,
    input_data: pd.DataFrame | np.ndarray | dict[str, Any] | list[Any],
) -> dict[str, Any]:
    """Execute inference on input features using a loaded pipeline.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        input_data: Features in any supported container format.

    Returns:
        Dictionary containing predictions, class labels, and probabilities.
    """
    cfg = get_config()
    features_df = prepare_inference_features(input_data, cfg.feature_names)

    logger.info("Executing prediction for %d sample(s)...", len(features_df))
    preds = pipeline.predict(features_df)
    labels = [cfg.class_names.get(int(p), str(p)) for p in preds]

    result: dict[str, Any] = {
        "predictions": [int(p) for p in preds],
        "class_labels": labels,
    }

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features_df)
        result["probabilities"] = probabilities.tolist()
        for idx, (lbl, prob) in enumerate(zip(labels, probabilities)):
            logger.info(
                "Sample %d: Predicted '%s' (P(Benign)=%.4f, P(Malignant)=%.4f)",
                idx,
                lbl,
                prob[0],
                prob[1],
            )
    else:
        for idx, lbl in enumerate(labels):
            logger.info("Sample %d: Predicted '%s'", idx, lbl)

    return result


def predict_from_artifact(
    artifact_path: str | Path,
    input_data: pd.DataFrame | np.ndarray | dict[str, Any] | list[Any],
) -> dict[str, Any]:
    """Load model artifact from disk and perform prediction in one call.

    Args:
        artifact_path: Path to serialized model artifact.
        input_data: Features to predict on.

    Returns:
        Inference result dictionary.
    """
    pipeline = load_model(artifact_path)
    return predict(pipeline, input_data)
