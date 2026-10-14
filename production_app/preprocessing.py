"""Preprocessing module for feature validation and scaling.

Encapsulates data transformations and schema enforcement to prevent
data leakage. Transformations are fitted exclusively on training sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from production_app.config import FEATURE_NAMES
from production_app.logging_config import get_logger

logger = get_logger(__name__)


def create_preprocessor() -> StandardScaler:
    """Create a new, unfitted feature scaler transformer.

    Returns:
        Unfitted StandardScaler instance.
    """
    logger.debug("Creating StandardScaler preprocessor instance.")
    return StandardScaler()


def validate_features(
    data: pd.DataFrame,
    expected_features: list[str] | None = None,
) -> pd.DataFrame:
    """Validate feature DataFrame schema and integrity before processing.

    Args:
        data: Input DataFrame containing features.
        expected_features: List of required feature names in order.

    Returns:
        Reordered and validated pd.DataFrame.

    Raises:
        ValueError: If columns are missing, nulls exist, or types are non-numeric.
    """
    features = expected_features or FEATURE_NAMES

    missing = [col for col in features if col not in data.columns]
    if missing:
        error_msg = f"Missing required feature columns: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Reorder columns to ensure strictly consistent ordering
    ordered_df = data[features].copy()

    # Check for missing values
    null_counts = ordered_df.isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        error_msg = f"Feature values contain nulls: {cols_with_nulls}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Verify all feature columns are numeric
    non_numeric = [
        col for col in features if not pd.api.types.is_numeric_dtype(ordered_df[col])
    ]
    if non_numeric:
        error_msg = f"Non-numeric feature columns detected: {non_numeric}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Verify all values are finite
    if not np.isfinite(ordered_df.to_numpy()).all():
        error_msg = "Non-finite values (inf / -inf) found in features."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("Feature schema validation passed for %d records.", len(ordered_df))
    return ordered_df
