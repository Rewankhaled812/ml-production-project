"""Model training and pipeline assembly module.

Combines feature preprocessing with classification estimators into a single
scikit-learn Pipeline and executes training on partitioned data.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from production_app.config import Config, get_config
from production_app.logging_config import get_logger
from production_app.preprocessing import create_preprocessor, validate_features

logger = get_logger(__name__)


def create_classifier(
    max_iter: int | None = None,
    c_param: float | None = None,
    random_state: int | None = None,
) -> LogisticRegression:
    """Create a LogisticRegression classification estimator.

    Args:
        max_iter: Maximum solver iterations.
        c_param: Inverse of regularization strength.
        random_state: Random state seed.

    Returns:
        Configured LogisticRegression instance.
    """
    cfg = get_config()
    iterations = max_iter if max_iter is not None else cfg.model_max_iter
    reg_c = c_param if c_param is not None else cfg.model_c
    seed = random_state if random_state is not None else cfg.random_state

    logger.debug(
        "Initializing LogisticRegression (max_iter=%d, C=%.2f, random_state=%d)",
        iterations,
        reg_c,
        seed,
    )
    return LogisticRegression(
        max_iter=iterations,
        C=reg_c,
        random_state=seed,
    )


def build_pipeline(config: Config | None = None) -> Pipeline:
    """Assemble the end-to-end ML Pipeline.

    Combines the StandardScaler preprocessor with the LogisticRegression classifier.
    Ensures that when fit is called, scaling parameters are learned strictly from
    the training set and faithfully applied at inference time.

    Args:
        config: Application configuration.

    Returns:
        Unfitted scikit-learn Pipeline.
    """
    cfg = config or get_config()
    preprocessor = create_preprocessor()
    classifier = create_classifier(
        max_iter=cfg.model_max_iter,
        c_param=cfg.model_c,
        random_state=cfg.random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", preprocessor),
            ("classifier", classifier),
        ]
    )
    logger.debug(
        "Assembled scikit-learn Pipeline with steps: %s", pipeline.named_steps.keys()
    )
    return pipeline


def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config | None = None,
) -> Pipeline:
    """Fit the machine learning pipeline on training data.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target labels.
        config: Optional configuration.

    Returns:
        Fitted scikit-learn Pipeline.

    Raises:
        ValueError: If input feature validation fails or dataset is empty.
    """
    cfg = config or get_config()

    logger.info("Validating training features schema...")
    validated_X = validate_features(X_train, cfg.feature_names)

    if len(validated_X) == 0:
        logger.error("Training feature set is empty.")
        raise ValueError("Cannot train pipeline on empty dataset.")

    pipeline = build_pipeline(cfg)

    logger.info(
        "Starting model training on %d samples with %d features...",
        len(validated_X),
        validated_X.shape[1],
    )

    pipeline.fit(validated_X, y_train)

    logger.info(
        "Training successfully completed. Fitted classifier: %s",
        type(pipeline.named_steps["classifier"]).__name__,
    )
    return pipeline
