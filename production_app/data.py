"""Data loading, validation, and splitting module.

Ensures reliable dataset retrieval, feature/target separation,
and reproducible train/test splits.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from production_app.config import Config, get_config
from production_app.logging_config import get_logger

logger = get_logger(__name__)


def load_from_url(
    url: str,
    feature_names: list[str],
    target_name: str,
    id_name: str,
) -> pd.DataFrame:
    """Load the Breast Cancer Wisconsin dataset directly from an HTTP URL.

    Args:
        url: Remote CSV / data URL.
        feature_names: Ordered list of feature column names.
        target_name: Name of target diagnosis column.
        id_name: Name of identifier column.

    Returns:
        pd.DataFrame containing all columns.

    Raises:
        urllib.error.URLError: If network or SSL connection fails.
    """
    logger.info("Fetching dataset from URL: %s", url)
    columns = [id_name, target_name] + feature_names
    return pd.read_csv(url, header=None, names=columns)


def load_from_sklearn(feature_names: list[str], target_name: str) -> pd.DataFrame:
    """Load Breast Cancer Wisconsin dataset from scikit-learn bundled data.

    Maps labels to match the WDBC convention:
    0 = Benign (B), 1 = Malignant (M).

    Args:
        feature_names: Expected feature column names.
        target_name: Name of target column.

    Returns:
        pd.DataFrame with standardized feature and target columns.
    """
    logger.info("Loading dataset from bundled scikit-learn repository...")
    raw_bunch = load_breast_cancer(as_frame=True)
    df = raw_bunch.data.copy()

    # Align scikit-learn feature column names to standard WDBC names
    if len(df.columns) == len(feature_names):
        df.columns = feature_names
    else:
        msg = (
            f"Feature count mismatch: expected {len(feature_names)}, "
            f"got {len(df.columns)}"
        )
        raise ValueError(msg)

    # In scikit-learn: 0 is malignant, 1 is benign.
    # In WDBC convention (notebook): 0 is Benign, 1 is Malignant.
    # We map 1 -> 0 (Benign) and 0 -> 1 (Malignant).
    sklearn_target = raw_bunch.target
    df[target_name] = 1 - sklearn_target

    return df


def load_dataset(config: Config | None = None) -> pd.DataFrame:
    """Load the dataset based on application configuration.

    Supports 'sklearn', 'url', or local file path sources with graceful fallback.

    Args:
        config: Application configuration instance.

    Returns:
        Standardized pd.DataFrame.
    """
    cfg = config or get_config()
    source = cfg.data_source

    logger.info("Initiating dataset loading using source strategy: '%s'", source)

    if source == "url":
        try:
            raw_df = load_from_url(
                url=cfg.data_url,
                feature_names=cfg.feature_names,
                target_name=cfg.target_column,
                id_name=cfg.id_column,
            )
            # Encode diagnosis if strings 'B'/'M'
            if raw_df[cfg.target_column].dtype == object or isinstance(
                raw_df[cfg.target_column].iloc[0], str
            ):
                raw_df[cfg.target_column] = raw_df[cfg.target_column].map(
                    cfg.label_mapping
                )
            logger.info("Successfully fetched %d records from URL.", len(raw_df))
            return raw_df
        except (urllib.error.URLError, ConnectionError, OSError) as err:
            logger.warning(
                "Network retrieval failed (%s). Falling back to scikit-learn.",
                err,
            )
            return load_from_sklearn(cfg.feature_names, cfg.target_column)

    if Path(source).is_file():
        logger.info("Loading dataset from local file: %s", source)
        local_df = pd.read_csv(source)
        if cfg.target_column in local_df.columns and (
            local_df[cfg.target_column].dtype == object
        ):
            local_df[cfg.target_column] = local_df[cfg.target_column].map(
                cfg.label_mapping
            )
        return local_df

    # Default to scikit-learn dataset
    df = load_from_sklearn(cfg.feature_names, cfg.target_column)
    logger.info("Successfully loaded %d records from scikit-learn.", len(df))
    return df


def separate_features_and_target(
    df: pd.DataFrame,
    target_column: str | None = None,
    id_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features matrix and target vector from raw dataframe.

    Args:
        df: Input dataframe.
        target_column: Name of the target column.
        id_column: Optional identifier column to drop.

    Returns:
        Tuple of (X, y) where X is feature DataFrame and y is target Series.

    Raises:
        ValueError: If target column is missing or contains invalid/null values.
    """
    cfg = get_config()
    target_col = target_column or cfg.target_column
    id_col = id_column or cfg.id_column

    if target_col not in df.columns:
        logger.error(
            "Target column '%s' not found in DataFrame columns: %s",
            target_col,
            df.columns.tolist(),
        )
        raise ValueError(f"Target column '{target_col}' missing from dataset.")

    data_copy = df.copy()
    if id_col and id_col in data_copy.columns:
        logger.debug("Dropping identifier column '%s'", id_col)
        data_copy = data_copy.drop(columns=[id_col])

    if data_copy[target_col].isnull().any():
        logger.error("Target column contains null values.")
        raise ValueError("Null values detected in target column.")

    X = data_copy.drop(columns=[target_col])
    y = data_copy[target_col].astype(int)

    if X.isnull().sum().sum() > 0:
        logger.error("Feature matrix contains missing values.")
        raise ValueError("Null values detected in feature matrix.")

    logger.info(
        "Features and target separated. Features shape: %s, Target shape: %s",
        X.shape,
        y.shape,
    )
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    random_state: int | None = None,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into reproducible train and test sets.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of dataset to include in test split (e.g. 0.2).
        random_state: Seed for random generator to guarantee reproducibility.
        stratify: Whether to maintain class proportions in splits.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    cfg = get_config()
    effective_test_size = test_size if test_size is not None else cfg.test_size
    effective_random_state = (
        random_state if random_state is not None else cfg.random_state
    )

    stratify_target = y if stratify else None

    logger.info(
        "Splitting data into train/test (test_size=%.2f, random_state=%d)...",
        effective_test_size,
        effective_random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=effective_test_size,
        random_state=effective_random_state,
        stratify=stratify_target,
    )

    logger.info(
        "Split complete. Training set: %s, Test set: %s",
        X_train.shape,
        X_test.shape,
    )
    return X_train, X_test, y_train, y_test
