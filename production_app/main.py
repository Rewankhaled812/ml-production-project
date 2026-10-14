"""Workflow orchestration module.

Coordinates data ingestion, pipeline training, evaluation, artifact persistence,
and inference by delegating implementation details to domain-specific modules.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from production_app.config import Config, get_config
from production_app.data import load_dataset, separate_features_and_target, split_data
from production_app.evaluation import evaluate_model, format_evaluation_summary
from production_app.logging_config import get_logger, setup_logging
from production_app.model_service import load_model, predict, save_model

logger = get_logger(__name__)


def train_workflow(config: Config | None = None) -> dict[str, Any]:
    """Execute the full end-to-end model training workflow.

    Steps:
        1. Initialize logging and load configuration.
        2. Ingest dataset.
        3. Separate features and target.
        4. Split into train and test sets (preventing data leakage).
        5. Build and train pipeline (preprocessing + classifier).
        6. Evaluate on held-out test data.
        7. Persist trained pipeline artifact to disk.

    Args:
        config: Optional custom application configuration.

    Returns:
        Dictionary summarizing execution status, metrics, and saved artifact location.
    """
    cfg = config or get_config()
    setup_logging(cfg.log_level)

    logger.info("==================================================")
    logger.info("   STARTING END-TO-END TRAINING WORKFLOW")
    logger.info("==================================================")

    # 1. Load data
    raw_df = load_dataset(cfg)

    # 2. Separate features and target
    X, y = separate_features_and_target(raw_df, cfg.target_column, cfg.id_column)

    # 3. Train/Test split
    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=True,
    )

    # 4. Train pipeline
    from production_app.training import train_pipeline

    pipeline = train_pipeline(X_train=X_train, y_train=y_train, config=cfg)

    # 5. Evaluate pipeline
    eval_metrics = evaluate_model(pipeline=pipeline, X_test=X_test, y_test=y_test)
    summary_report = format_evaluation_summary(eval_metrics)
    logger.info("\n%s", summary_report)

    # 6. Save artifact
    artifact_path = cfg.resolve_model_path()
    saved_path = save_model(pipeline=pipeline, destination_path=artifact_path)

    logger.info("==================================================")
    logger.info("   TRAINING WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("   Model Artifact: %s", saved_path)
    logger.info("==================================================")

    return {
        "status": "success",
        "artifact_path": str(saved_path),
        "metrics": eval_metrics,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


def predict_workflow(
    input_data: pd.DataFrame | dict[str, Any] | list | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """Execute model prediction workflow using saved pipeline artifact.

    If input_data is not provided, loads a sample from the test split to demonstrate
    prediction on unseen data.

    Args:
        input_data: Optional features to predict on.
        config: Optional application configuration.

    Returns:
        Dictionary containing prediction results, probabilities, and labels.
    """
    cfg = config or get_config()
    setup_logging(cfg.log_level)

    logger.info("==================================================")
    logger.info("   STARTING INFERENCE WORKFLOW")
    logger.info("==================================================")

    model_path = cfg.resolve_model_path()
    pipeline = load_model(model_path)

    # If no input provided, use a representative test sample from the dataset
    if input_data is None:
        logger.info(
            "No inference payload provided. Fetching sample from evaluation dataset..."
        )
        raw_df = load_dataset(cfg)
        X, _ = separate_features_and_target(raw_df, cfg.target_column, cfg.id_column)
        sample = X.iloc[[0]]
    else:
        sample = input_data

    result = predict(pipeline=pipeline, input_data=sample)

    logger.info("==================================================")
    logger.info("   INFERENCE WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("==================================================")

    return result
