<div align="center">

# AQI Predictor

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6600?logo=xgboost&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE?logo=apacheairflow&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Tuning-4B0082)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Azure Blob](https://img.shields.io/badge/Azure%20Blob-Storage-0078D4?logo=microsoftazure&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-Database-4479A1?logo=mysql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)
![Terraform](https://img.shields.io/badge/Terraform-IaC-844FBA?logo=terraform&logoColor=white)

[![ML Tests & CML Report](https://github.com/YubikNakarmi/Aqi_predictor/actions/workflows/test.yml/badge.svg)](https://github.com/YubikNakarmi/Aqi_predictor/actions/workflows/test.yml)
[![Push to Docker Hub](https://github.com/YubikNakarmi/Aqi_predictor/actions/workflows/push_images.yml/badge.svg)](https://github.com/YubikNakarmi/Aqi_predictor/actions/workflows/push_images.yml)

[![CD to Azure](https://github.com/YubikNakarmi/Aqi_predictor/actions/workflows/azure-deploy.yml/badge.svg?branch=main)](https://github.com/YubikNakarmi/Aqi_predictor/actions/workflows/azure-deploy.yml)

**An end-to-end Air Quality Index (AQI) forecasting system that predicts hourly PM2.5 levels up to 24 hours ahead using XGBoost models, orchestrated by Apache Airflow and served via MLflow.**


</div>

---

<details>
<summary><b>Table of Contents</b></summary>

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Project Layout](#project-layout)
- [Airflow DAGs](#airflow-dags)
- [Stack](#stack)
- [Setup](#setup)
- [Port Reference](#port-reference)
- [MLflow](#mlflow)
- [Training Pipeline](#training-pipeline)
- [Prediction Pipeline](#prediction-pipeline)
- [Frontend](#frontend)
- [Azure Deployment](#azure-deployment)
- [Development](#development-editable-install)
- [Notebooks](#notebooks)
- [Secrets](#secrets)

</details>

## Project Overview

This project builds and operates a complete AQI forecasting system:

- **Ingests** AQI and weather data from the OpenWeather API (historical and live).
- **Preprocesses** raw data with cleaning, imputation, segmentation, and feature engineering.
- **Tunes** hyperparameters per forecast horizon using Optuna.
- **Trains** 24 independent XGBoost models (one per hour ahead, 1–24h).
- **Evaluates** models against MAE/RMSE thresholds and promotes passing models to staging.
- **Serves** all 24 models behind a single MLflow `pyfunc` endpoint.
- **Predicts** daily by ingesting live data, preprocessing, and running inference.
- **Displays** predictions in a Streamlit dashboard.
- **Orchestrates** training (monthly) and prediction (daily) via Airflow DAGs running Dockerized tasks.

> [!IMPORTANT]
> Since open source data was used, it is very poor quality for AQI prediction. Thus, the model is not accurate as it should be. For future use good qualty data with additional weather data and with less missing values. 


---

## Architecture

<div align="center">

![Pipeline Architecture](.github_utils/pipeline-Page-1.drawio.png)

</div>

---

## Data Flow

### Training (monthly)
1. Raw hourly CSVs are read from `data/raw/static/hourly/`.
2. `DataCleaner` handles missing values, outliers, segmentation, and feature engineering.
3. Processed data is split into train/val/test (70/15/15) and saved as Parquet under `data/processed/hourly/`.
4. Optuna tunes hyperparameters; best params saved to `data/artifacts/best_params.json`.
5. 24 XGBoost models are trained (one per horizon) and registered in MLflow Model Registry.
6. Models passing evaluation thresholds are promoted to staging.
7. A composite `ServeModel` wrapping all 24 models is registered as `hourly_pm25_24h_service`.

### Prediction (daily)
1. Live AQI + weather data is fetched from OpenWeather and saved as CSV in `data/raw/pred_ingestion/`.
2. Preprocessing applies the same transformations as training and outputs Parquet to `data/processed/pred_ingestion/`.
3. The latest `hourly_pm25_24h_service` model produces 24-hour forecasts.
4. Predictions are saved to `data/predictions/hourly/` and optionally written to MySQL.

---

---

## Project Layout

<details>
<summary><b>Click to expand full directory tree</b></summary>

```
├── dags/                        # Airflow DAG definitions
│   ├── pred_dag.py              #   Daily prediction pipeline
│   ├── train_dag.py             #   Monthly training pipeline
│   └── monitor_dag.py           #   Monitoring (placeholder)
├── hourly_train_pipeline/       # Training pipeline scripts
│   ├── preprocess.py            #   Raw → processed Parquet
│   ├── tune.py                  #   Optuna hyperparameter search
│   ├── train.py                 #   Train 24 XGBoost models
│   ├── eval.py                  #   Evaluate & promote models
│   └── serve.py                 #   Register composite serving model
├── hourly_pred_pipeline/        # Prediction pipeline scripts
│   ├── ingest.py                #   Fetch live data from OpenWeather
│   ├── preprocess.py            #   Clean & engineer features
│   └── pred.py                  #   Run inference & save results
├── frontend/
│   └── app.py                   # Streamlit dashboard
├── src/
│   ├── modules/                 # Shared library modules
│   └── scripts/                 # CLI utilities & core logic
├── docker/                      # Dockerfiles
│   ├── dockerfile.airflow       #   Airflow image
│   ├── Dockerfile.mlflow        #   MLflow tracking server
│   ├── Dockerfile.trainer       #   Training container
│   ├── dockerfile.pred          #   Prediction container
│   └── Dockerfile.streamlit     #   Streamlit frontend
├── azure-deploy/                # Terraform infrastructure (Azure)
│   ├── main.tf
│   ├── variable.tf
│   ├── terraform.tfvars
│   └── versions.tf
├── config/
│   └── airflow.cfg              # Airflow configuration
├── data/                        # Runtime data (git-ignored)
│   ├── raw/                     #   Raw CSVs from ingestion
│   ├── processed/               #   Cleaned Parquet files
│   ├── predictions/             #   Model output
│   ├── artifacts/               #   Best params, plots
│   └── metadata/                #   Pipeline run metadata
├── db/                          # Local SQLite backends (git-ignored)
├── notebooks/                   # Exploration & visualization
├── docker-compose.yaml          # Full service orchestration
├── setup.cfg                    # Package metadata & entry points
├── pyproject.toml               # Build system config
├── requirements.txt             # Python dependencies
└── setup_data_structure.py      # Creates data directory tree
```

</details>

---

## Airflow DAGs

| DAG | Schedule | Tasks | Description |
|-----|----------|-------|-------------|
| `aqi_pred_dag` | `@daily` | `fetch_data` → `preprocess_data` → `predict` | Ingests live data, preprocesses, and runs inference using the `aqi-prediction` Docker image |
| `aqi_train_dag` | `@monthly` | `preprocess_data` → `tune_hyperparameters` → `train_model` → `evaluate_model` | Full retraining pipeline using the `aqi-trainer` Docker image |
| `monitor_dag` | — | — | Placeholder for future monitoring tasks |

> [!NOTE]
> All tasks run as **DockerOperators**, mounting the host `data/` and `db/` directories into containers. Environment variables (API keys, MLflow URI, horizon, split ratios, target columns) are passed from the Airflow environment.

---

## Stack

| Component | Purpose |
|-----------|---------|
| **Apache Airflow** (CeleryExecutor) | DAG orchestration with PostgreSQL metadata + Redis broker |
| **MLflow** | Experiment tracking, model registry, artifact storage |
| **XGBoost** | Gradient-boosted tree models for AQI prediction |
| **Optuna** | Bayesian hyperparameter optimization |
| **DVC** | Data versioning with Azure Blob Storage remote |
| **Docker Compose** | Multi-service container orchestration |
| **Streamlit** | Interactive prediction dashboard |
| **MySQL** | MLflow backend store and prediction logging |
| **Azure Blob Storage** | MLflow artifact store and DVC remote |
| **Terraform** | Azure infrastructure provisioning |

---

## Setup

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Runtime |
| Docker & Compose | Latest | Container orchestration |
| MySQL | 8.x | MLflow backend store |
| Git | Latest | Version control |
| DVC | 3.x | Data versioning (Azure Blob) |

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YubikNakarmi/Aqi_predictor.git
cd Aqi_predictor
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Editable install for local src/ modules
pip install -e .
```

### 3. Create required directories

Several directories are excluded from Git (via `.gitignore`) but are required at runtime. Create them after cloning:

```bash
# Data directory tree
python setup_data_structure.py

# Directories ignored by Git but needed by services
mkdir -p logs db secrets
```

| Directory | Purpose | In `.gitignore` |
|-----------|---------|:---------------:|
| `data/` | Raw, processed, predictions, artifacts | Yes |
| `logs/` | Airflow scheduler & task logs | Yes |
| `db/` | MLflow & Optuna SQLite backends | Yes |
| `secrets/` | Credential files (e.g. Azure keys) | Yes |

Data directory tree created by `setup_data_structure.py`:
```
data/
├── artifacts/
├── plots/
│   ├── daily/
│   └── hourly/
├── predictions/
│   ├── daily/
│   └── hourly/
├── processed/
│   ├── daily/
│   ├── hourly/
│   └── pred_ingestion/
└── raw/
    ├── dynamic/
    ├── pred_ingestion/
    └── static/
        ├── daily/
        │   ├── aqi/
        │   └── weather/
        └── hourly/
```

### 4. Environment configuration

Copy the template and fill in your values:
```bash
cp .env.template .env
```

> [!NOTE]
> All variables below must be set before running any service.

<details>
<summary><b>Required variables in <code>.env</code></b></summary>

| Variable | Example / Description |
|----------|----------------------|
| `OPEN_WEATHER_API_KEY` | Your OpenWeather API key |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob connection string for DVC & MLflow artifacts |
| `MLFLOW_ARTIFACT_ROOT` | `wasbs://container@account.blob.core.windows.net/` or a local path |
| `MYSQL_HOST` | `host.docker.internal` (local) or remote host |
| `MYSQL_PW` | MySQL password |
| `MYSQL_USER` | `root` or your MySQL user |
| `MYSQLURL` | `mysql+pymysql://root:password@host.docker.internal:3306/mlflow_backend` |
| `AIRFLOW_PROJ_DIR` | Absolute path to this repo on the host |
| `HORIZON` | `24` (forecast horizon in hours) |
| `TARGET_COL` | `pm25` (target column for training) |
| `TARGET_COLS` | `pm25,o3` (comma-separated list for preprocessing) |
| `STATION_NAME` | `us_paro_hourly` |
| `VALUE_MIN` | `0` (minimum valid AQI value) |
| `VALUE_MAX` | `500` (maximum valid AQI value) |

</details>

### 5. MySQL setup

Create the required databases on your MySQL server:
```sql
CREATE DATABASE IF NOT EXISTS mlflow_backend;
CREATE DATABASE IF NOT EXISTS pypipeline_predictions;
```

### 6. DVC setup

Configure the remote in `.dvc/config` (example):
```ini
[core]
  remote = azureremote
  autostage = true
[remote "azureremote"]
  url = azure://aqi/static
```

Pull tracked data:
```sh
dvc pull
```

Tracked by DVC:
- `data/raw/`
- `data/processed/`
- `db/mlflow.db`
- `db/optuna_study.db`

### 7. Build and run with Docker Compose

```bash
docker compose up --build
```

This starts the core services: **Airflow** (API server + scheduler + DAG processor + worker + triggerer), **MLflow**, **Optuna Dashboard**, **PostgreSQL** (Airflow metadata), and **Redis** (Celery broker).

<details>
<summary><b>Run individual or optional services</b></summary>

```bash
docker compose up mlflow                       # MLflow only
docker compose up optuna-server                # Optuna dashboard only
docker compose --profile manual up trainer     # Training container
docker compose --profile manual up prediction  # Prediction container
docker compose --profile manual up streamlit   # Streamlit dashboard
docker compose --profile flower up flower      # Celery Flower monitoring
```

</details>

---

## Port Reference

| Port | Service | Description |
|------|---------|-------------|
| 5000 | MLflow Tracking | Tracking server UI and API |
| 5050 | MLflow Model Serving | Model serving REST endpoint (`/invocations`) |
| 5555 | Flower | Celery worker monitoring (optional, `--profile flower`) |
| 8080 | Airflow API Server | Airflow web UI and REST API |
| 8081 | Optuna Dashboard | Hyperparameter tuning dashboard |
| 8501 | Streamlit | AQI prediction dashboard (optional, `--profile manual`) |

> [!WARNING]
> **Cloud inbound rules** — Open ports **22** (SSH), **5000** (MLflow), **5050** (Model Serving), **8080** (Airflow), **8081** (Optuna), and **8501** (Streamlit) in your security group.

---

## MLflow

The Docker setup uses MySQL for the backend store and Azure Blob Storage for artifacts. The tracking server is exposed at http://localhost:5000.

> [!TIP]
> For local development, you can switch to a SQLite backend by changing `MYSQLURL` to a local `.sqlite` path and `MLFLOW_ARTIFACT_ROOT` to a local directory in `.env`.

MLflow stores:
- Parameters and metrics for each training run
- Optuna tuning results (via `MLflowCallback`)
- Registered models per horizon (`pm25_plus_{h}h_model`)
- Composite serving model (`hourly_pm25_24h_service`)

---

## Libraries
Local libararies were specifically build for this project to reduce code redundancy and to make the code more modular and reusable. The libraries are located in `src/modules/` and `src/scripts/`. They include:
### data_cleaner.py
  data preprocessing utilities (missing value handling, outlier detection, feature engineering). This requires the following features to work properly. There are differnt function which can be called to run this module in differnt ways: 

#### `run()`:
 main entry point for preprocessing raw data. Applies all transformations and saves processed Parquet files. Furthermore, this steps also removes unecessary columns .Expected input features:
 ```python
 ["locationId", "location", "city", "country", "utc", "local", "parameter", "value", "unit", "latitude", "longitude"]
 ```
 [!NOTE]
 This is hard coded, need to make it more flexible to accomodate different datasets.

#### `run_clean()`:
This step cleans the raw data with major steps such as: 
- Extracting pm and o3 columns from the raw data
- Clean data by handling ouliers, setting frequency of data to hourly.
- Adding time features such as hour of day, day of week, is_weekend, and is_night.
- Add missing values flags. 
#### `run_feature_engineering()`:
This step adds lag features, rolling statistics, and shifted target columns for all horizons. It also applies the segmentation logic to prevent data leakage across missing-value gaps. Expected input features:
```python
["pm25", "o3", "hour","date","was_imputed","day_of_week","is_weekend","
is_night","hour_sin","hour_cos","pm25_missing","o3_missing","was_imputed",
"pm25_gap_length","o3_gap_length","segment_id",]
```
#### Installation
```sh
pip install -e .
```

This exposes CLI entry points defined in `setup.cfg`:
- `aqi-ingest`
- `aqi-train`
- `data_hourly_preprocessing`

To undo:
```sh
pip uninstall pypipeline
```

---


## Training Pipeline

<div align="center">

![Pipeline Architecture](.github_utils/train.drawio.png)

</div>

Hourly training runs per horizon (1–24) with separate models and tuning. Pipeline scripts live under `hourly_train_pipeline/`.

Key environment variables (set in `.env`):
```bash
TARGET_COLS="pm25,o3"    # comma-separated list for preprocessing
TARGET_COL="pm25"        # target for training
VALUE_MIN=0
VALUE_MAX=500
HORIZON=24
```

### Preprocessing
`preprocess.py` reads raw CSVs, applies transformations via `DataCleaner`, and writes processed Parquet to `data/processed/hourly/`. Key steps:
- Handling missing values (imputation or dropping)
- Segmentation by missing-value gaps to avoid data leakage
- Outlier detection and capping
- Feature engineering (lag features, rolling statistics, time-based features)
- Train/val/test split (70/15/15 by default)
- Shifted target column creation for all horizons (1–24h ahead)

Output: `train_processed.parquet`, `val_processed.parquet`, `test_processed.parquet`

<details>
<summary><b>Data Segmentation — how it works (click to expand)</b></summary>

#### Gap Classification

```
Gap Size       │ Strategy        │ Confidence
───────────────┼─────────────────┼───────────
≤ 6 hours      │ Interpolation   │ high
6–12 hours     │ KNN (k=6)       │ medium
12–24 hours    │ Tag only        │ low
> 24 hours     │ Segment break   │ —
```

#### Before Segmentation — raw hourly data with gaps

```
 Row │ Timestamp          │  pm25  │  o3   │ missing │ gap_len │ Notes
─────┼────────────────────┼────────┼───────┼─────────┼─────────┼──────────────────
  0  │ 2025-01-01 00:00   │  35.0  │ 28.0  │    0    │    0    │
  1  │ 2025-01-01 01:00   │  38.0  │ 30.0  │    0    │    0    │
  2  │ 2025-01-01 02:00   │   -    │   -   │    1    │    2    │ ← small gap
  3  │ 2025-01-01 03:00   │   -    │   -   │    1    │    2    │ ← small gap
  4  │ 2025-01-01 04:00   │  40.0  │ 32.0  │    0    │    0    │
  5  │ 2025-01-01 05:00   │  42.0  │ 34.0  │    0    │    0    │
  6  │ 2025-01-01 06:00   │   -    │   -   │    1    │   30    │ ┐
  7  │ 2025-01-01 07:00   │   -    │   -   │    1    │   30    │ │
  8  │ 2025-01-01 08:00   │   -    │   -   │    1    │   30    │ │ very large
  .  │        ...         │   -    │   -   │    1    │   30    │ │ gap (30h)
  .  │        ...         │   -    │   -   │    1    │   30    │ │ NOT imputed
 35  │ 2025-01-02 11:00   │   -    │   -   │    1    │   30    │ ┘
 36  │ 2025-01-02 12:00   │  55.0  │ 40.0  │    0    │    0    │
 37  │ 2025-01-02 13:00   │  58.0  │ 42.0  │    0    │    0    │
 38  │ 2025-01-02 14:00   │  60.0  │ 44.0  │    0    │    0    │
```

#### After Imputation + Segmentation

```
 Row │ Timestamp          │  pm25  │  o3   │ imputed │ confidence │ gap_len │ seg_id │ Notes
─────┼────────────────────┼────────┼───────┼─────────┼────────────┼─────────┼────────┼────────────
  0  │ 2025-01-01 00:00   │  35.0  │ 28.0  │    0    │   none     │    0    │   0    │
  1  │ 2025-01-01 01:00   │  38.0  │ 30.0  │    0    │   none     │    0    │   0    │
  2  │ 2025-01-01 02:00   │ (38.7) │(30.7) │    1    │   high     │    2    │   0    │ interpolated
  3  │ 2025-01-01 03:00   │ (39.3) │(31.3) │    1    │   high     │    2    │   0    │ interpolated
  4  │ 2025-01-01 04:00   │  40.0  │ 32.0  │    0    │   none     │    0    │   0    │
  5  │ 2025-01-01 05:00   │  42.0  │ 34.0  │    0    │   none     │    0    │   0    │
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  SEGMENT BREAK (gap 30h > 24h)  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  6  │ 2025-01-01 06:00   │   -    │   -   │    0    │     -      │   30    │   0    │ ┐ dropped
  .  │        ...         │   -    │   -   │    0    │     -      │   30    │   0    │ │ at train
 35  │ 2025-01-02 11:00   │   -    │   -   │    0    │     -      │   30    │   0    │ ┘ time
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
 36  │ 2025-01-02 12:00   │  55.0  │ 40.0  │    0    │   none     │    0    │   1    │ ← new seg
 37  │ 2025-01-02 13:00   │  58.0  │ 42.0  │    0    │   none     │    0    │   1    │
 38  │ 2025-01-02 14:00   │  60.0  │ 44.0  │    0    │   none     │    0    │   1    │

  ( ) = imputed value       - = NaN / missing
```

#### Effect on Lag Features — computed within each `segment_id`

```
 Row │ Timestamp          │  pm25  │ seg_id │ lag_1  │ lag_2  │ roll_3 │ Why
─────┼────────────────────┼────────┼────────┼────────┼────────┼────────┼───────────────────────
  4  │ 2025-01-01 04:00   │  40.0  │   0    │  39.3  │  38.7  │  38.7  │ lags from same segment
  5  │ 2025-01-01 05:00   │  42.0  │   0    │  40.0  │  39.3  │  39.4  │ lags from same segment
     │                    │        │        │        │        │        │
 36  │ 2025-01-02 12:00   │  55.0  │   1    │  NaN   │  NaN   │  NaN   │ start of new segment
 37  │ 2025-01-02 13:00   │  58.0  │   1    │  55.0  │  NaN   │  55.0  │ lag within segment 1
 38  │ 2025-01-02 14:00   │  60.0  │   1    │  58.0  │  55.0  │  56.5  │ lag within segment 1

  Row 36: lag_1 = NaN (NOT 42.0 from row 5)
  Without segmentation it would be 42.0 — a value from 30h ago → DATA LEAKAGE
```

</details>

### Tuning
`tune.py` runs Optuna (default 60 trials) to find optimal XGBoost hyperparameters. Results are logged to the `xgb_aqi_hourly_tuning` MLflow experiment via `MLflowCallback`. The best parameters are saved to `data/artifacts/best_params.json`. Optuna study state is persisted in a SQLite database (`db/optuna.db`).

### Training
`train.py` trains 24 XGBoost models using the best hyperparameters from tuning. Each model is trained for a specific forecast horizon (1–24h ahead) and:
- Logged as a separate MLflow run with parameters and metrics
- Registered in MLflow Model Registry as `pm25_plus_{h}h_model`

### Evaluation
`eval.py` evaluates all 24 trained models on the holdout test set using `mlflow.models.evaluate()`. Models passing the following thresholds are promoted to **staging**:
- MAE < 25
- RMSE < 40

Combined predictions are saved to `data/predictions/hourly/`. Per-horizon and aggregate metrics are logged to MLflow and recorded in pipeline metadata.

### Serving
`serve.py` registers a composite **custom `pyfunc` model** (`ServeModel`) that wraps all 24 horizon models into a single MLflow endpoint. On inference, it runs all 24 models and returns predictions in columns `pm25_plus_1h_pred` through `pm25_plus_24h_pred`.

The model is registered as `hourly_pm25_24h_service` in MLflow Model Registry.

By default, `docker-compose.yaml` starts the model serving via the `model-serve` service on port **5050**. The endpoint is `http://localhost:5050/invocations`.

Start serving locally via CLI:
```bash
mlflow models serve --model-uri models:/hourly_pm25_24h_service/latest --host 0.0.0.0 --port 5050 --no-conda
```

Example pipeline run (local):
```bash
python hourly_train_pipeline/preprocess.py
python hourly_train_pipeline/tune.py
python hourly_train_pipeline/train.py
python hourly_train_pipeline/eval.py
python hourly_train_pipeline/serve.py
```

#### Usage example (from processed data)

```python
import pandas as pd
import requests

df = pd.read_parquet("data/processed/hourly/us_paro_hourly/train_processed.parquet")

# Drop target/excluded columns to get just features
exclude = (
    [f"pm25_plus_{i}h" for i in range(1, 25)]
    + [f"o3_plus_{i}h" for i in range(1, 25)]
    + ["segment_id", "imputation_confidence", "pm25_target", "o3_target"]
)
features = df.drop(columns=[c for c in exclude if c in df.columns])

# Send 1 row
sample = features.iloc[[0]]
payload = {"dataframe_split": sample.to_dict(orient="split")}

resp = requests.post("http://localhost:5050/invocations", json=payload)
print(resp.json())
```

#### Usage example (manual payload)

```bash
curl -X POST http://localhost:5050/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["pm25","o3","hour","day_of_week","is_weekend","is_night",
        "pm25_missing","o3_missing","pm25_gap_length","o3_gap_length",
        "was_imputed","hour_sin","hour_cos",
        "pm25_lag_1","o3_lag_1","pm25_lag_2","o3_lag_2",
        "...remaining lag, roll, slope, and ratio columns..."],
      "data": [[35.0, 28.0, 14, 2, 0, 0, 0, 0, 0, 0, 0.0, 0.99, -0.10, "...values..."]]
    }
  }'
```

The response returns predictions for all 24 horizons:
```json
{"pm25_plus_1h_pred": [42.1], "pm25_plus_2h_pred": [43.5], "...": "..."}
```

> [!TIP]
> The Python approach using processed data is recommended — it guarantees columns and types match the model signature exactly.

---

## Prediction Pipeline

<div align="center">

![Pipeline Architecture](.github_utils/pred.drawio.png)

</div>


The daily prediction pipeline runs as the `aqi_pred_dag` Airflow DAG and produces live AQI forecasts. Scripts live under `hourly_pred_pipeline/`.

### Ingestion
`ingest.py` fetches current AQI data from the OpenWeather API with a configurable lookback (default 24h). Data is saved as a dated CSV under `data/raw/pred_ingestion/`. Pipeline metadata (success/failure) is recorded to `data/metadata/ingestion.json`.

### Preprocessing
`preprocess.py` reads the ingested CSV and applies the same `DataCleaner` transformations used during training: time features, missing-value flags, gap length computation, segmentation, and feature engineering. Output is saved as Parquet in `data/processed/pred_ingestion/`.

### Prediction
`pred.py` loads the latest version of `hourly_pm25_24h_service` from MLflow Model Registry, runs inference on the most recent preprocessed row, and produces 24-hour forecasts. Predictions are:
- Saved as CSV to `data/predictions/hourly/`
- Optionally written to the `hourly_predictions` MySQL table
- Logged to the `xgb_aqi_hourly_prediction` MLflow experiment

Example pipeline run (local):
```bash
python hourly_pred_pipeline/ingest.py
python hourly_pred_pipeline/preprocess.py
python hourly_pred_pipeline/pred.py
```

## Frontend

A **Streamlit** dashboard (`frontend/app.py`) provides a visual interface for AQI predictions:
- **Horizon slider** — select forecast horizon (1–24h)
- **AQI metric display** — shows the predicted value
- **Trend chart** — visualizes prediction trends over time

Run locally:
```bash
streamlit run frontend/app.py
```

Or via Docker Compose:
```bash
docker compose --profile manual up streamlit
```

The dashboard is available at http://localhost:8501.

---

## Azure Deployment

Infrastructure-as-code for Azure is in `azure-deploy/` using Terraform:
- **Resource group** and **Virtual Network** provisioning
- Configured via `terraform.tfvars` and `variable.tf`
- Uses Azure CLI for provider authentication

```bash
cd azure-deploy
terraform init
terraform plan
terraform apply
```

---
## CI/CD



## Notebooks

Exploration and visualization notebooks live under `notebooks/`:

| Notebook | Purpose |
|----------|---------|
| `datapreprocess(limited).ipynb` | Data preprocessing exploration |
| `houlry_train.ipynb` | Hourly model training experiments |
| `aqi_calculation.ipynb` | AQI calculation methodology |
| `plots.ipynb` | Model performance plots |
| `pred.ipynb` | Prediction experimentation |
| `visuals.ipynb` | Data visualizations |

---

## Secrets

> [!CAUTION]
> Never commit secrets to version control. All credential files are git-ignored.

Store all secrets in `.env` (git-ignored). Required credentials:
```
OPEN_WEATHER_API_KEY=...
AZURE_STORAGE_CONNECTION_STRING=...
MYSQL_PW=...
```

For Azure Blob Storage, the connection string can also be placed in `secrets/azure_storage_connection_string.txt` for services that read from file.

---
# Improvements
- Use a more robust data for training, currently only aqi data with large gaps is used, which is not ideal for training a good model. A more complete dataset with fewer gaps would improve model performance. Additonally use weather data as features, which are currently not used but could provide valuable information for AQI prediction.
- Make cenral modules in /src more robust and customizable, such as DataCleaner, to handle different datasets and feature engineering needs without code changes.
- Implement monitoring using NannyML or custom Airflow tasks to track data quality, model performance, and pipeline health over time.
# Credits
- [Training data source](https://opendatanepal.com/datasets/air-quality-data-in-kathmandu)
