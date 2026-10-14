# Production Machine Learning Classification: Breast Cancer Wisconsin

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: ruff](https://img.shields.io/badge/linter-ruff-red.svg)](https://github.com/astral-sh/ruff)
[![Environment: uv](https://img.shields.io/badge/packaging-uv-purple.svg)](https://docs.astral.sh/uv/)

From experimentation notebook to a modular, reproducible, production-ready Python application.

---

## 1. Project Overview & Goal

The primary objective of this project is to bridge the gap between experimental data science (Jupyter Notebooks) and robust software engineering (production Python applications).

Rather than optimizing for incremental accuracy tweaks, this project emphasizes:
- **Separation of Concerns**: Splitting monolithic notebook scripts into clean, decoupled, single-responsibility modules (`data`, `preprocessing`, `training`, `evaluation`, `model_service`, and `main`).
- **Reproducibility**: Managing dependencies and locking environments with `uv` (`pyproject.toml` and `uv.lock`).
- **Leakage Prevention**: Utilizing scikit-learn's `Pipeline` to ensure all feature scaling and transformations fit exclusively on training data and carry over seamlessly into inference.
- **Model Persistence & Serving**: Serializing the trained pipeline with `joblib` so that inference runs in production without retraining.
- **Software Quality**: Strict type hinting, structured logging (no `print()` in production code), defensive exception handling, and automated code-quality hooks (`ruff`, `black`, `pre-commit`).

---

## 2. Dataset & ML Problem

- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI / Scikit-learn).
- **Instances**: 569 patient samples.
- **Features**: 30 continuous numeric attributes derived from digitized cell nuclei images (e.g., `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `concavity_mean`, etc.).
- **Target Variable**: Binary classification:
  - `0`: Benign (B) — 357 instances (62.7%)
  - `1`: Malignant (M) — 212 instances (37.3%)
- **Data Integrity**: 0 missing values, 0 duplicate records.

---

## 3. Machine Learning Pipeline Architecture

The end-to-end pipeline is structured to eliminate data leakage and guarantee deterministic behavior across training and serving:

```
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
```

### Why Logistic Regression?
Logistic Regression with standard scaling provides:
- High interpretability and fast convergence.
- Probabilistic outputs for clinical risk assessment (`predict_proba`).
- Low compute footprint suitable for real-time edge or API microservices.

---

## 4. Evaluation Results

Evaluated on a held-out, stratified test set of 114 samples (72 Benign, 42 Malignant):

| Metric | Score |
|---|---|
| **Accuracy** | **96.49%** |
| **Precision (Malignant)** | **97.50%** |
| **Recall (Malignant)** | **92.86%** |
| **F1-Score (Malignant)** | **95.12%** |

### Confusion Matrix
```
                  Predicted Benign (0)    Predicted Malignant (1)
True Benign (0)            71 (TN)                 1 (FP)
True Malignant (1)          3 (FN)                39 (TP)
```
These results are based on a single train/test split and are intended for educational evaluation, not clinical diagnosis.
---

## 5. Folder Hierarchy & Responsibilities

The codebase follows the required architecture:

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
│   ├── model_service.py        # Joblib persistence, artifact loading, and inference
│   └── main.py                 # High-level workflow orchestration
├── models/                     # Storage directory for saved pipeline artifacts
│   └── pipeline.joblib         # Serialized fitted pipeline artifact
├── run.py                      # CLI entry point for training and inference
├── .env.example                # Safe sample environment configuration
├── .gitignore                  # Git exclusions (.venv, .env, caches, checkpoints)
├── .pre-commit-config.yaml     # Pre-commit hook definitions (black, ruff, sanity checks)
├── pyproject.toml              # Declarative dependencies & tool configurations
├── uv.lock                     # Deterministic dependency lockfile
└── README.md                   # Complete architectural and operational documentation
```

---

## 6. Installation & Setup Guide

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
git clone <[https://github.com/Rewankhaled812/ml-production-project]>
cd ml-production-project
```

### Step 2: Create Environment and Install Dependencies
Using `uv`, synchronize and lock all dependencies into `.venv`:
```bash
# Sync dependencies declared in pyproject.toml and uv.lock
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
Default `.env` configuration:
```ini
APP_ENV=development
LOG_LEVEL=INFO
DATA_SOURCE=sklearn
DATA_URL=https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
TEST_SIZE=0.2
RANDOM_STATE=42
MODEL_MAX_ITER=1000
MODEL_C=1.0
MODEL_PATH=models/pipeline.joblib
```

---

## 7. Execution Guide

You can run workflows using `uv run` or by executing Python inside the virtual environment.

### 1. Train and Persist the Pipeline
Fits the pipeline on the training set, computes evaluation metrics on the test set, and serializes the fitted pipeline to `models/pipeline.joblib`:
```bash
uv run python run.py --mode train
```

### 2. Predict Using Saved Artifact (No Retraining)
Loads the persisted `models/pipeline.joblib`, prepares the input schema, and performs inference with class probabilities:
```bash
uv run python run.py --mode predict
```

### 3. Optional Command-Line Overrides
```bash
# Train with custom artifact destination and debug logs
uv run python run.py --mode train --model-path models/custom_model.joblib --log-level DEBUG

# Predict with a custom model
uv run python run.py --mode predict --model-path models/custom_model.joblib
```

---

## 8. Code Quality & Pre-commit

Run the configured linters, formatters, and git hooks before committing:
```bash
# Lint with Ruff
uv run ruff check .

# Check formatting with Black
uv run black --check .

# Run pre-commit across all files
uv run pre-commit run --all-files
```

---

## 9. Verification & Acceptance Checklist

- [x] **Exploration Record**: `notebooks/01_exploration.ipynb` demonstrates verified data checks and baseline validation.
- [x] **Refactored Architecture**: Reusable Python logic cleanly decoupled across `production_app/` modules.
- [x] **Data Leakage Prevented**: `StandardScaler` is wrapped in a `Pipeline` fitted strictly on training data.
- [x] **Model Persistence**: Serializes to `models/pipeline.joblib`; prediction loads the artifact with zero retraining.
- [x] **Configuration**: Settings loaded via `config.py` from environment variables, documented in `.env.example`.
- [x] **Logging**: Informative operational logs throughout; zero `print()` statements in production modules.
- [x] **Defensive Engineering**: Input validation, custom error handling, and type annotations across all functions.
- [x] **Tooling & Git**: `uv` environment tracking (`pyproject.toml`, `uv.lock`), `.gitignore`, and passing pre-commit hooks (`ruff`, `black`).
