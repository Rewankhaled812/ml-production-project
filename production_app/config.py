"""Configuration management for the ML production application.

Loads environment variables and application defaults to avoid hardcoding
runtime parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

# Automatically load variables from .env if present
load_dotenv()

FEATURE_NAMES: list[str] = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

TARGET_COLUMN: str = "Diagnosis"
ID_COLUMN: str = "ID"

LABEL_MAPPING: dict[str, int] = {
    "B": 0,
    "M": 1,
}

CLASS_NAMES: dict[int, str] = {
    0: "Benign (B)",
    1: "Malignant (M)",
}


@dataclass(frozen=True)
class Config:
    """Application configuration container."""

    # Project metadata
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    app_env: str = field(
        default_factory=lambda: os.getenv("APP_ENV", "development").strip().lower()
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").strip().upper()
    )

    # Data configuration
    data_source: str = field(
        default_factory=lambda: os.getenv("DATA_SOURCE", "sklearn").strip().lower()
    )
    data_url: str = field(
        default_factory=lambda: os.getenv(
            "DATA_URL",
            (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "breast-cancer-wisconsin/wdbc.data"
            ),
        ).strip()
    )
    test_size: float = field(
        default_factory=lambda: float(os.getenv("TEST_SIZE", "0.2"))
    )
    random_state: int = field(
        default_factory=lambda: int(os.getenv("RANDOM_STATE", "42"))
    )

    # Model training configuration
    model_max_iter: int = field(
        default_factory=lambda: int(os.getenv("MODEL_MAX_ITER", "1000"))
    )
    model_c: float = field(default_factory=lambda: float(os.getenv("MODEL_C", "1.0")))
    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_PATH", "models/pipeline.joblib"))
    )

    # Schema configuration
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    target_column: str = TARGET_COLUMN
    id_column: str = ID_COLUMN
    label_mapping: ClassVar[dict[str, int]] = LABEL_MAPPING
    class_names: ClassVar[dict[int, str]] = CLASS_NAMES

    def resolve_model_path(self) -> Path:
        """Resolve model artifact path against project root if relative."""
        if self.model_path.is_absolute():
            return self.model_path
        return (self.project_root / self.model_path).resolve()


def get_config() -> Config:
    """Retrieve an initialized application configuration instance."""
    return Config()
