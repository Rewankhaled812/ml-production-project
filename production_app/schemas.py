"""Pydantic schemas for API request validation and structured responses.

Defines typed data models conforming to the Breast Cancer Wisconsin Diagnostic
feature schema and diagnostic classification outputs.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from production_app.config import FEATURE_NAMES


class PredictionRequest(BaseModel):
    """Observation schema containing all 30 required diagnostic cell features.

    All features represent continuous non-negative physical measurements
    extracted from digitized cell nucleus images.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "example": {
                "radius_mean": 17.99,
                "texture_mean": 10.38,
                "perimeter_mean": 122.8,
                "area_mean": 1001.0,
                "smoothness_mean": 0.1184,
                "compactness_mean": 0.2776,
                "concavity_mean": 0.3001,
                "concave_points_mean": 0.1471,
                "symmetry_mean": 0.2419,
                "fractal_dimension_mean": 0.07871,
                "radius_se": 1.095,
                "texture_se": 0.9053,
                "perimeter_se": 8.589,
                "area_se": 153.4,
                "smoothness_se": 0.006399,
                "compactness_se": 0.04904,
                "concavity_se": 0.05373,
                "concave_points_se": 0.01587,
                "symmetry_se": 0.03003,
                "fractal_dimension_se": 0.006193,
                "radius_worst": 25.38,
                "texture_worst": 17.33,
                "perimeter_worst": 184.6,
                "area_worst": 2019.0,
                "smoothness_worst": 0.1622,
                "compactness_worst": 0.6656,
                "concavity_worst": 0.7119,
                "concave_points_worst": 0.2654,
                "symmetry_worst": 0.4601,
                "fractal_dimension_worst": 0.1189,
            }
        },
    )

    # 1. Mean nuclear measurements (10 features)
    radius_mean: float = Field(
        ...,
        description="Mean of distances from center to points on perimeter.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[17.99],
    )
    texture_mean: float = Field(
        ...,
        description="Standard deviation of gray-scale values (mean).",
        ge=0.0,
        allow_inf_nan=False,
        examples=[10.38],
    )
    perimeter_mean: float = Field(
        ...,
        description="Mean size of the core tumor perimeter.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[122.8],
    )
    area_mean: float = Field(
        ...,
        description="Mean area of the cell nuclei.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[1001.0],
    )
    smoothness_mean: float = Field(
        ...,
        description="Mean of local variation in radius lengths.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.1184],
    )
    compactness_mean: float = Field(
        ...,
        description="Mean of (perimeter^2 / area - 1.0).",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.2776],
    )
    concavity_mean: float = Field(
        ...,
        description="Mean severity of concave portions of contour.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.3001],
    )
    concave_points_mean: float = Field(
        ...,
        description="Mean number of concave portions of contour.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.1471],
    )
    symmetry_mean: float = Field(
        ...,
        description="Mean symmetry score of nuclei.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.2419],
    )
    fractal_dimension_mean: float = Field(
        ...,
        description="Mean 'coastline approximation' - 1.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.07871],
    )

    # 2. Standard error nuclear measurements (10 features)
    radius_se: float = Field(
        ...,
        description="Standard error for mean of distances from center to perimeter.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[1.095],
    )
    texture_se: float = Field(
        ...,
        description="Standard error for standard deviation of gray-scale values.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.9053],
    )
    perimeter_se: float = Field(
        ...,
        description="Standard error for perimeter size.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[8.589],
    )
    area_se: float = Field(
        ...,
        description="Standard error for nuclear area.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[153.4],
    )
    smoothness_se: float = Field(
        ...,
        description="Standard error for local variation in radius lengths.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.006399],
    )
    compactness_se: float = Field(
        ...,
        description="Standard error for compactness.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.04904],
    )
    concavity_se: float = Field(
        ...,
        description="Standard error for severity of concave contour portions.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.05373],
    )
    concave_points_se: float = Field(
        ...,
        description="Standard error for number of concave contour portions.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.01587],
    )
    symmetry_se: float = Field(
        ...,
        description="Standard error for nuclear symmetry.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.03003],
    )
    fractal_dimension_se: float = Field(
        ...,
        description="Standard error for 'coastline approximation' - 1.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.006193],
    )

    # 3. Largest ('worst') nuclear measurements (10 features)
    radius_worst: float = Field(
        ...,
        description="Largest ('worst') mean value for distances to perimeter.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[25.38],
    )
    texture_worst: float = Field(
        ...,
        description="Largest ('worst') mean value for gray-scale standard deviation.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[17.33],
    )
    perimeter_worst: float = Field(
        ...,
        description="Largest ('worst') mean value for perimeter.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[184.6],
    )
    area_worst: float = Field(
        ...,
        description="Largest ('worst') mean value for nuclear area.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[2019.0],
    )
    smoothness_worst: float = Field(
        ...,
        description="Largest ('worst') mean value for radius local variation.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.1622],
    )
    compactness_worst: float = Field(
        ...,
        description="Largest ('worst') mean value for compactness.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.6656],
    )
    concavity_worst: float = Field(
        ...,
        description="Largest ('worst') mean value for concavity severity.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.7119],
    )
    concave_points_worst: float = Field(
        ...,
        description="Largest ('worst') mean value for concave contour points.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.2654],
    )
    symmetry_worst: float = Field(
        ...,
        description="Largest ('worst') mean value for nuclear symmetry.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.4601],
    )
    fractal_dimension_worst: float = Field(
        ...,
        description="Largest ('worst') mean value for fractal dimension.",
        ge=0.0,
        allow_inf_nan=False,
        examples=[0.1189],
    )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert validated request data into a strictly ordered pandas DataFrame.

        Returns:
            DataFrame containing exactly the model features in their expected order.
        """
        data = self.model_dump()
        return pd.DataFrame([[data[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)


class PredictionResponse(BaseModel):
    """Response model for a successful diagnostic prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_class": 1,
                "class_label": "Malignant (M)",
                "probabilities": {
                    "Benign (B)": 0.0001,
                    "Malignant (M)": 0.9999,
                },
            }
        }
    )

    predicted_class: int = Field(
        ...,
        description="Original predicted class index (0 for Benign, 1 for Malignant).",
        examples=[1],
    )
    class_label: str = Field(
        ...,
        description="Human-readable diagnostic class label.",
        examples=["Malignant (M)"],
    )
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Class membership probabilities mapped by readable class label.",
        examples=[{"Benign (B)": 0.0001, "Malignant (M)": 0.9999}],
    )


class HealthResponse(BaseModel):
    """Response model describing service and model health."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_path": "models/pipeline.joblib",
                "detail": "Model artifact is loaded and ready for inference.",
            }
        }
    )

    status: str = Field(
        ...,
        description="Operational status of the service ('healthy' or 'unhealthy').",
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the trained pipeline is loaded in memory and usable.",
        examples=[True],
    )
    model_path: str | None = Field(
        default=None,
        description="Configured model artifact storage path.",
        examples=["models/pipeline.joblib"],
    )
    detail: str | None = Field(
        default=None,
        description="Operational context or health details.",
        examples=["Model artifact is loaded and ready for inference."],
    )
