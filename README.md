# Production Machine Learning Classification: Breast Cancer Wisconsin

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: ruff](https://img.shields.io/badge/linter-ruff-red.svg)](https://github.com/astral-sh/ruff)
[![Environment: uv](https://img.shields.io/badge/packaging-uv-purple.svg)](https://docs.astral.sh/uv/)

From experimentation notebook to a modular, reproducible, production-ready Python application and validated REST ML API.

---

## 1. Project Overview & Scope

This repository bridges the gap between experimental data science and robust, production-grade software engineering through two iterative stages:

- **Assignment 01 — "From Notebook to Production Python"**: Refactored an exploratory Jupyter notebook (`notebooks/01_exploration.ipynb`) into a decoupled, modular Python package (`production_app/`), encapsulating data loading, leakage-free preprocessing, pipeline training, evaluation, and serialized persistence (`models/pipeline.joblib`).
- **Assignment 02 — "From Saved Model to a Validated ML API"**: Extends the existing application by building a high-performance **FastAPI** service (`production_app/api.py`) backed by strict **Pydantic** validation schemas (`production_app/schemas.py`). The API loads the pre-trained scikit-learn pipeline into memory at startup and provides real-time inference with zero retraining.

### Architectural Principles
- **Separation of Concerns**: Isolated responsibilities across modules (`data`, `preprocessing`, `training`, `evaluation`, `model_service`, `schemas`, and `api`).
- **Zero Retraining on Inference**: The API never trains, refits, or adapts preprocessing on inference requests. The fitted pipeline artifact (`models/pipeline.joblib`) is reused as-is.
- **Strict Input Validation**: Pydantic models enforce non-negative physical domain constraints (`ge=0.0`), reject non-finite inputs (`NaN`, `Infinity`), forbid unknown fields, and validate strict types.
- **Defensive Error Handling**: Internal exceptions and file paths are never exposed to API clients. Controlled HTTP status codes (`200`, `422`, `503`, `500`) are guaranteed.
- **Lifecycle Efficiency**: The fitted pipeline artifact is loaded once into memory during application lifespan startup, avoiding disk I/O on individual requests.

---

## 2. Dataset & ML Problem

- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI / Scikit-learn).
- **Instances**: 569 patient samples.
- **Features**: 30 continuous numeric attributes derived from digitized cell nuclei images (mean, standard error, and "worst" or largest values across 10 nuclear characteristics).
- **Target Variable**: Binary classification:
  - `0`: Benign (B) — 357 instances (62.7%)
  - `1`: Malignant (M) — 212 instances (37.3%)
- **Data Integrity**: 0 missing values, 0 duplicate records.

---

## 3. Machine Learning Pipeline Architecture

The end-to-end pipeline is structured to eliminate data leakage and guarantee deterministic behavior across training, CLI inference, and API serving:

```text
[Raw Ingestion] (UCI URL / Scikit-learn fallback / CSV)
       │
       ▼
[Data Validation & Split] (Stratified 80/20 train/test split)
       │
       ▼
┌──────────────────────── Scikit-Learn Pipeline ────────────────────────┐
│                                                                        │
│   [StandardScaler]      ──► Fitted on X_train ONLY                     │
│          │                                                             │
│          ▼                                                             │
│   [LogisticRegression]  ──► L2 regularized, solver: lbfgs, max_iter: 1000│
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[Evaluation Engine]               [Model Persistence]
 - Accuracy, Precision, Recall, F1 - Joblib serialization to models/
 - Confusion Matrix & Reports      - Artifact reload for inference
                                         │
                                         ▼
                               [FastAPI Serving Layer]
                                - Pydantic Request Validation
                                - In-Memory Pipeline Inference
                                - Structured JSON Predictions
```

### Why Logistic Regression with StandardScaler?
- **Interpretability & Calibration**: Provides well-calibrated class probabilities for clinical decision support.
- **Leakage Prevention**: Feature scaling statistics ($\mu, \sigma$) are computed strictly on training data within the pipeline.
- **Low Latency**: Sub-millisecond inference time per observation, ideal for high-throughput production microservices.

---

## 4. Evaluation Results (Assignment 01)

Evaluated on a held-out, stratified test set of 114 samples (72 Benign, 42 Malignant):

| Metric | Score |
|---|---|
| **Accuracy** | **96.49%** |
| **Precision (Malignant)** | **97.50%** |
| **Recall (Malignant)** | **92.86%** |
| **F1-Score (Malignant)** | **95.12%** |

### Confusion Matrix
```text
                  Predicted Benign (0)    Predicted Malignant (1)
True Benign (0)            71 (TN)                 1 (FP)
True Malignant (1)          3 (FN)                39 (TP)
```

---

## 5. Folder Hierarchy & Responsibilities

```text
ml-production-project/
├── notebooks/
│   └── 01_exploration.ipynb    # Validated exploratory analysis & baseline checks
├── production_app/             # Modular application package
│   ├── __init__.py             # Package marker and shared exports
│   ├── config.py               # Centralized configuration & environment loader
│   ├── logging_config.py       # Unified logger setup and formatting
│   ├── data.py                 # Dataset ingestion, separation, and train/test splitting
│   ├── preprocessing.py        # Scaler definition and feature schema validation
│   ├── training.py             # Pipeline assembly and estimator fitting
│   ├── evaluation.py           # Classification metrics calculation & reporting
│   ├── model_service.py        # Model persistence, cached loading, and inference
│   ├── main.py                 # Workflow orchestration (train_workflow, predict_workflow)
│   ├── schemas.py              # Pydantic v2 schemas for request validation & responses
│   └── api.py                  # FastAPI application with /health and /predict endpoints
├── models/                     # Storage directory for saved pipeline artifacts
│   ├── .gitkeep
│   └── pipeline.joblib         # Serialized fitted pipeline artifact (tracked by Git)
├── run.py                      # Unified CLI entry point (train, predict, api modes)
├── .env.example                # Safe sample environment configuration
├── .gitignore                  # Git exclusions (.venv, .env, caches, checkpoints)
├── .pre-commit-config.yaml     # Pre-commit hook definitions (black, ruff, sanity checks)
├── pyproject.toml              # Declarative dependencies & tool configurations
├── uv.lock                     # Deterministic dependency lockfile
└── README.md                   # Complete architectural and operational documentation
```

---

## 6. Installation & Environment Setup

### Prerequisites
- **Python**: `>= 3.12`
- **uv**: Fast Python package and environment manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/)).
  ```bash
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Git**: Installed and available in PATH.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Rewankhaled812/ml-production-project.git
cd ml-production-project
```

### Step 2: Install Dependencies with uv
Using `uv`, synchronize and lock all dependencies into `.venv`:
```bash
uv sync
```
*(Alternatively, you can activate `.venv` directly:*
- *Windows: `.venv\Scripts\activate`*
- *Linux/macOS: `source .venv/bin/activate`)*

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```
Default configuration values:
```ini
# Application Environment (development, testing, production)
APP_ENV=development

# Logging Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Dataset Source: "url", "sklearn", or path to a local CSV file
DATA_SOURCE=sklearn
DATA_URL=https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

# Split and Reproducibility Configuration
TEST_SIZE=0.2
RANDOM_STATE=42

# Model Hyperparameters
MODEL_MAX_ITER=1000
MODEL_C=1.0

# Model Artifact Storage Path
MODEL_PATH=models/pipeline.joblib

# API Server Configuration
API_HOST=127.0.0.1
API_PORT=8000
```

---

## 7. Model Artifact Lifecycle & Preparation

The trained model artifact `models/pipeline.joblib` is already included and tracked in this repository.

If the artifact is ever missing or if you wish to retrain it from scratch, execute:
```bash
uv run python run.py --mode train
```
This fits the `StandardScaler` and `LogisticRegression` pipeline on the training split, logs evaluation metrics, and persists the serialized artifact to `models/pipeline.joblib`.

---

## 8. Execution Guide

You can run CLI workflows or launch the REST API using `uv run`.

### 1. Assignment 01: Train and Persist
Fits and evaluates the pipeline, then saves it to disk:
```bash
uv run python run.py --mode train
```

### 2. Assignment 01: CLI Prediction (No Retraining)
Loads `models/pipeline.joblib` and evaluates an unseen sample:
```bash
uv run python run.py --mode predict
```

### 3. Assignment 02: Start the FastAPI Application
Start the API using Uvicorn:
```bash
uv run uvicorn production_app.api:app --host 127.0.0.1 --port 8000
```
Or start via the unified CLI entry point:
```bash
uv run python run.py --mode api
```

### 4. Stopping the API
To terminate the server, press `Ctrl+C` in your terminal.

---

## 9. API Architecture & Request-to-Prediction Flow

When a client sends an HTTP request to the API:

```text
HTTP Client
    │
    ▼
[FastAPI Router]
    │
    ├─► GET /health  ──► Checks in-memory pipeline status
    │                    ├─► Model Loaded?  ──► HTTP 200 (healthy)
    │                    └─► Not Loaded?    ──► HTTP 503 (unhealthy)
    │
    └─► POST /predict
          │
          ▼
    [Pydantic Schema Validation (PredictionRequest)]
          ├─► Missing feature?         ──► HTTP 422 Unprocessable Entity
          ├─► Non-numeric / NaN / Inf? ──► HTTP 422 Unprocessable Entity
          ├─► Negative value (< 0)?    ──► HTTP 422 Unprocessable Entity
          └─► Extra unknown fields?    ──► HTTP 422 Unprocessable Entity
          │
          ▼ (Validated Features)
    [Ordered DataFrame Conversion]
          │
          ▼
    [In-Memory Pipeline (models/pipeline.joblib)]
          ├── StandardScaler.transform() (learned training stats)
          └── LogisticRegression.predict() & predict_proba()
          │
          ▼
    [Structured JSON Response (PredictionResponse)] ──► HTTP 200 OK
```

---

## 10. API Endpoints Reference

### Interactive Documentation
Once the server is running, access Swagger UI or ReDoc in any browser:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Root redirect**: Visiting [http://127.0.0.1:8000/](http://127.0.0.1:8000/) automatically redirects to `/docs`.

---

### A. GET `/health`

Inspects service operational readiness and confirms that the model pipeline is loaded and ready for inference.

#### Healthy Response (HTTP 200 OK)
When the model artifact is loaded into memory:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/pipeline.joblib",
  "detail": "Model artifact loaded and ready for inference."
}
```

#### Unhealthy Response (HTTP 503 Service Unavailable)
If the model artifact is missing, corrupted, or cannot be loaded:
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "model_path": "models/pipeline.joblib",
  "detail": "Model artifact is unavailable or cannot be loaded."
}
```

---

### B. POST `/predict`

Performs diagnostic inference on exactly one observation matching the 30 cell nuclei features.

#### Required Feature Schema
Every request must supply exactly these 30 numerical features (`float`, non-negative, finite):

| Feature Name | Type | Constraint | Description |
|---|---|---|---|
| `radius_mean` | `float` | $\ge 0.0$ | Mean of distances from center to points on perimeter |
| `texture_mean` | `float` | $\ge 0.0$ | Standard deviation of gray-scale values (mean) |
| `perimeter_mean` | `float` | $\ge 0.0$ | Mean size of core tumor perimeter |
| `area_mean` | `float` | $\ge 0.0$ | Mean area of cell nuclei |
| `smoothness_mean` | `float` | $\ge 0.0$ | Mean of local variation in radius lengths |
| `compactness_mean` | `float` | $\ge 0.0$ | Mean of $(perimeter^2 / area - 1.0)$ |
| `concavity_mean` | `float` | $\ge 0.0$ | Mean severity of concave portions of contour |
| `concave_points_mean` | `float` | $\ge 0.0$ | Mean number of concave portions of contour |
| `symmetry_mean` | `float` | $\ge 0.0$ | Mean symmetry score of nuclei |
| `fractal_dimension_mean` | `float` | $\ge 0.0$ | Mean "coastline approximation" - 1 |
| `radius_se` | `float` | $\ge 0.0$ | Standard error for radius mean |
| `texture_se` | `float` | $\ge 0.0$ | Standard error for gray-scale standard deviation |
| `perimeter_se` | `float` | $\ge 0.0$ | Standard error for perimeter size |
| `area_se` | `float` | $\ge 0.0$ | Standard error for nuclear area |
| `smoothness_se` | `float` | $\ge 0.0$ | Standard error for local variation in radius |
| `compactness_se` | `float` | $\ge 0.0$ | Standard error for compactness |
| `concavity_se` | `float` | $\ge 0.0$ | Standard error for concavity |
| `concave_points_se` | `float` | $\ge 0.0$ | Standard error for concave portions of contour |
| `symmetry_se` | `float` | $\ge 0.0$ | Standard error for symmetry |
| `fractal_dimension_se` | `float` | $\ge 0.0$ | Standard error for fractal dimension |
| `radius_worst` | `float` | $\ge 0.0$ | Largest mean value for radius |
| `texture_worst` | `float` | $\ge 0.0$ | Largest mean value for gray-scale standard deviation |
| `perimeter_worst` | `float` | $\ge 0.0$ | Largest mean value for perimeter |
| `area_worst` | `float` | $\ge 0.0$ | Largest mean value for nuclear area |
| `smoothness_worst` | `float` | $\ge 0.0$ | Largest mean value for radius local variation |
| `compactness_worst` | `float` | $\ge 0.0$ | Largest mean value for compactness |
| `concavity_worst` | `float` | $\ge 0.0$ | Largest mean value for concavity severity |
| `concave_points_worst` | `float` | $\ge 0.0$ | Largest mean value for concave points |
| `symmetry_worst` | `float` | $\ge 0.0$ | Largest mean value for symmetry |
| `fractal_dimension_worst` | `float` | $\ge 0.0$ | Largest mean value for fractal dimension |

---

#### 1. Valid Request Example (Malignant Observation)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
       "fractal_dimension_worst": 0.1189
     }'
```

#### Successful Response (HTTP 200 OK)
```json
{
  "predicted_class": 1,
  "class_label": "Malignant (M)",
  "probabilities": {
    "Benign (B)": 0.0,
    "Malignant (M)": 1.0
  }
}
```

---

#### 2. Valid Request Example (Benign Observation)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "radius_mean": 13.54,
       "texture_mean": 14.36,
       "perimeter_mean": 87.46,
       "area_mean": 566.3,
       "smoothness_mean": 0.09779,
       "compactness_mean": 0.08129,
       "concavity_mean": 0.06664,
       "concave_points_mean": 0.04781,
       "symmetry_mean": 0.1885,
       "fractal_dimension_mean": 0.05766,
       "radius_se": 0.2699,
       "texture_se": 0.7886,
       "perimeter_se": 2.058,
       "area_se": 23.56,
       "smoothness_se": 0.008462,
       "compactness_se": 0.0146,
       "concavity_se": 0.02387,
       "concave_points_se": 0.01315,
       "symmetry_se": 0.0198,
       "fractal_dimension_se": 0.0023,
       "radius_worst": 15.11,
       "texture_worst": 19.26,
       "perimeter_worst": 99.7,
       "area_worst": 711.2,
       "smoothness_worst": 0.144,
       "compactness_worst": 0.1773,
       "concavity_worst": 0.239,
       "concave_points_worst": 0.1288,
       "symmetry_worst": 0.2977,
       "fractal_dimension_worst": 0.07259
     }'
```

#### Successful Response (HTTP 200 OK)
```json
{
  "predicted_class": 0,
  "class_label": "Benign (B)",
  "probabilities": {
    "Benign (B)": 0.936214,
    "Malignant (M)": 0.063786
  }
}
```

---

#### 3. Validation Failure Examples (HTTP 422 Unprocessable Entity)

- **Missing Required Feature** (e.g. `radius_mean` omitted):
  ```json
  {
    "detail": [
      {
        "type": "missing",
        "loc": ["body", "radius_mean"],
        "msg": "Field required"
      }
    ]
  }
  ```

- **Invalid Type** (e.g. `"texture_mean": "abc"`):
  ```json
  {
    "detail": [
      {
        "type": "float_type",
        "loc": ["body", "texture_mean"],
        "msg": "Input should be a valid number",
        "input": "abc"
      }
    ]
  }
  ```

- **Negative Domain Violation** (e.g. `"radius_mean": -5.0`):
  ```json
  {
    "detail": [
      {
        "type": "greater_than_equal",
        "loc": ["body", "radius_mean"],
        "msg": "Input should be greater than or equal to 0",
        "input": -5.0,
        "ctx": { "ge": 0.0 }
      }
    ]
  }
  ```

- **Non-Finite Values** (e.g. `"radius_mean": NaN` or `Infinity`):
  ```json
  {
    "detail": [
      {
        "type": "finite_number",
        "loc": ["body", "radius_mean"],
        "msg": "Input should be a finite number",
        "input": "nan"
      }
    ]
  }
  ```

---

#### 4. Model Unavailable Failure (HTTP 503 Service Unavailable)
If the model file is removed or inaccessible when a prediction is requested:
```json
{
  "detail": "Model service is currently unavailable. Ensure the model artifact is trained and accessible."
}
```
*Note*: Internal stack traces and server filesystem paths are never returned to the client.

---

## 11. Code Quality & Pre-commit Hooks

Run the configured formatters, linters, and pre-commit checks:
```bash
# Run Ruff linting
uv run ruff check .

# Check code formatting with Black
uv run black --check .

# Run pre-commit checks across all tracked files
uv run pre-commit run --all-files
```

---

## 12. Verification & Acceptance Checklist

- [x] **Exploration Record**: `notebooks/01_exploration.ipynb` preserves exploratory analysis and baseline validation.
- [x] **Refactored Architecture**: Reusable Python modules cleanly decoupled across `production_app/`.
- [x] **Zero Retraining on Inference**: Pre-trained scikit-learn pipeline loaded once and reused across all API requests.
- [x] **Pydantic Validation**: Strict schema enforcement for all 30 features, catching missing keys, invalid types, non-finite values, and negative inputs.
- [x] **Health Check**: `GET /health` distinguishes between healthy (HTTP 200) and unavailable (HTTP 503) model states.
- [x] **Inference Endpoint**: `POST /predict` returns original predicted class (`0`/`1`), readable label, and probability distribution.
- [x] **Defensive Error Handling**: Controlled HTTP error responses without leaking file paths or exception tracebacks.
- [x] **Interactive Documentation**: Swagger UI available at `/docs` with interactive testing capabilities.
- [x] **Packaging & Consistency**: Managed via `uv` (`pyproject.toml`, `uv.lock`) with clean pre-commit passing (`ruff`, `black`).
