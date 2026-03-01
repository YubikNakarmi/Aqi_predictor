# AQI predictor with Airflow

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6600?logo=xgboost&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE?logo=apacheairflow&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Tuning-4B0082)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Azure Blob](https://img.shields.io/badge/Azure%20Blob-Storage-0078D4?logo=microsoftazure&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-Database-4479A1?logo=mysql&logoColor=white)

[![Pylint](https://github.com/YubikNakarmi/Aqi_predictor/actions/workflows/pylint.yml/badge.svg?branch=test)](https://github.com/YubikNakarmi/Aqi_predictor/actions/workflows/pylint.yml)



This project predicts hourly AQI using historical and live CSV data. Data stays in CSV files (no external DB required for core data).

## Contents
- [AQI predictor with Airflow](#aqi-predictor-with-airflow)
  - [Contents](#contents)
  - [Project overview](#project-overview)
  - [Data flow](#data-flow)
  - [Project layout](#project-layout)
  - [Airflow DAGs](#airflow-dags)
  - [Stack](#stack)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [1. Clone and create virtual environment](#1-clone-and-create-virtual-environment)
    - [2. Install dependencies](#2-install-dependencies)
    - [3. Create data directory structure](#3-create-data-directory-structure)
    - [4. Environment configuration](#4-environment-configuration)
    - [5. MySQL setup](#5-mysql-setup)
    - [6. DVC setup](#6-dvc-setup)
    - [7. Build and run with Docker Compose](#7-build-and-run-with-docker-compose)
  - [Port Rules](#port-rules)
  - [MLflow](#mlflow)
  - [Training Pipeline](#training-pipeline)
    - [Preprocessing](#preprocessing)
    - [Tuning](#tuning)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Serving](#serving)
      - [Usage example (from processed data)](#usage-example-from-processed-data)
      - [Usage example (manual payload)](#usage-example-manual-payload)
  - [Development (editable install)](#development-editable-install)
  - [Notebooks](#notebooks)
  - [Secrets](#secrets)

## Project overview

This repo runs an end-to-end AQI pipeline that:
- Ingests AQI and weather history from OpenWeather APIs and writes CSVs.
- Preprocesses and trains horizon-based models (1-24 hours ahead).
- Tracks experiments and models in MLflow.
- Serves a selected model via MLflow model serving (Docker).

## Data flow

1. Ingestion pulls AQI + weather history and merges by timestamp.
2. Cleaned CSVs are written under `data/raw` and `data/processed`.
3. Training generates models per horizon and logs to MLflow.
4. Artifacts and predictions are stored in `data/artifacts` and `data/predictions`.

## Project layout

- `dags/`: Airflow DAGs for ingestion and training.
- `hourly_train_pipeline/`: Training pipeline scripts (preprocess, tune, train, eval).
- `src/scripts/`: CLI utilities such as ingestion.
- `data/`: Raw, processed, predictions, and artifacts.
- `db/`: MLflow and Optuna SQLite backends when used locally.
- `docker/`: Dockerfiles for Airflow, MLflow, and trainer images.

## Airflow DAGs

- `aqi_data_dag` (daily): pulls AQI data from OpenWeather and writes CSVs.
- `aqi_train_dag` (monthly): preprocesses, tunes, trains, and evaluates models in Docker.

## Stack
- Airflow for orchestration
- MLflow for model tracking
- DVC for data versioning (Azure Blob Storage)
- Optuna for hyperparameter tuning
- XGBoost for model training

## Setup

### Prerequisites

| Requirement        | Version   | Purpose                          |
|--------------------|-----------|----------------------------------|
| Python             | 3.10+     | Runtime                          |
| Docker & Compose   | Latest    | Container orchestration          |
| MySQL              | 8.x       | MLflow backend store             |
| Git                | Latest    | Version control                  |
| DVC                | 3.x       | Data versioning (Azure Blob)     |

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

### 3. Create data directory structure

```bash
python setup_data_structure.py
```

This creates the required directory tree:
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
│   └── hourly/
└── raw/
    ├── dynamic/
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

Required variables in `.env`:

| Variable                          | Example / Description                                                        |
|-----------------------------------|------------------------------------------------------------------------------|
| `OPEN_WEATHER_API_KEY`            | Your OpenWeather API key                                                     |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob connection string for DVC & MLflow artifacts                      |
| `MLFLOW_ARTIFACT_ROOT`            | `wasbs://container@account.blob.core.windows.net/` or a local path           |
| `MYSQL_HOST`                      | `host.docker.internal` (local) or remote host                               |
| `MYSQL_PW`                        | MySQL password                                                               |
| `MYSQL_USER`                      | `root` or your MySQL user                                                    |
| `MYSQLURL`                        | `mysql+pymysql://root:password@host.docker.internal:3306/mlflow_backend`     |
| `AIRFLOW_PROJ_DIR`                | Absolute path to this repo on the host                                       |
| `HORIZON`                         | `24` (forecast horizon in hours)                                             |
| `TARGET_COL`                      | `pm25` (target column for training)                                          |
| `STATION_NAME`                    | `us_paro_hourly`                                                             |

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
- data/raw/
- data/processed/
- db/mlflow.db
- db/optuna_study.db

### 7. Build and run with Docker Compose

```bash
docker compose up --build
```

This starts: **Airflow** (webserver + scheduler + worker), **MLflow**, **Optuna Dashboard**, **PostgreSQL** (Airflow metadata), and **Redis** (Celery broker).

Run individual services:
```bash
docker compose up mlflow           # MLflow only
docker compose up optuna-server    # Optuna dashboard only
```

## Port Rules

| Port | Service              | Description                              |
|------|----------------------|------------------------------------------|
| 5000 | MLflow Tracking      | MLflow tracking server UI and API        |
| 5050 | MLflow Model Serving | MLflow model serving REST endpoint       |
| 8080 | Airflow API Server   | Airflow web UI and REST API              |
| 8081 | Optuna Dashboard     | Optuna hyperparameter tuning dashboard   |

> **Cloud inbound rules**: Open ports **22** (SSH), **5000** (MLflow), **8080** (Airflow), **8081** (Optuna), and **5050** (Model Serving) in your security group.

## MLflow

The Docker setup uses MySQL for the backend store and Azure Blob Storage for artifacts. The tracking server is exposed at http://localhost:5000. You can set up mlflow locally for development with a local SQLite backend by changing the `MLFLOW_ARTIFACT_ROOT` to local path and `MYSQLURI` to a .SQLITE db locally in `.env` and running the mlflow server locally.

MLflow stores:
- Parameters and metrics for each training run
- Optuna tuning results
- Models registered per horizon

## Training Pipeline

Hourly training runs per horizon (1-24) with separate models and tuning. Pipeline scripts live under `hourly_train_pipeline/`.

Key environment variables:
```bash
export TARGET_COLS="pm25,o3,pm10"  # comma-separated list for preprocessing
export TARGET_COL="pm25"           # target for training
export VALUE_MIN=0
export VALUE_MAX=500
export HORIZON=24
```
should be set on `.env` file

### Preprocessing
`.preprocess.py` reads raw CSVs, applies transformations, and writes processed parquet on `/data/processed/..`.
This utilizes local libraries under `src/` for data cleaning and feature engineering, more on dependancy graph. The key preprocessing steps include:
- Handling missing values (imputation or dropping)
- segmentation by missing values to avoid leakage
- outlier detection and capping
- Feature engineering (lag features, rolling stats, time-based features)
more on detail on  `/notebooks/preprocessing.ipynb`

### Tuning
`.tune.py` runs Optuna to find the best hyperparameters for the XGBoost model per horizon. Results are logged to MLflow. This utilizes the Optuna integration with MLflow to track tuning experiments. `/src/train_houly.py` contains the training logic and is called by both tuning and training scripts, while `tune` is the wrapper. The best hyperparameters are saved to a JSON file under `data/artifacts/` for use in the training step.

### Training

`.train.py` trains the final model using the best hyperparameters found in tuning. The trained model is logged to MLflow and saved as a pickle file under `data/artifacts/`. After training the model is registered in MLflow Model Registry under a name like `aqi_model_horizon_{HORIZON}`.




### Evaluation
`.eval.py` evaluates the trained model on a holdout test set and logs metrics to MLflow. Evaluation results are also saved as JSON under `data/artifacts/` and predictoin on `data/predictions`. The mlflow is modeled to log the evaluation metrics under the same run as training, so you can see the full picture of training and evaluation in one place. The follwoing are the requirements for promotion:
- MAE < 20
- RMESE <30

Example runs:
```bash
python hourly_train_pipeline/preprocess.py 
python hourly_train_pipeline/tune.py
python hourly_train_pipeline/train.py
python hourly_train_pipeline/eval.py
python hourly_train_pipeline/serve.py
```

### Serving
`serve.py` serves the selected model using MLflow's built-in model serving capabilities. The model is loaded from MLflow Model Registry and exposed via a REST API endpoint. This utilize custom pyfunc model that combines all the 24 horizon models into one endpoint, where you can specify the horizon as a query parameter. 

By default `docker-compose.yaml` automatically starts the model serving through `serve` service. The defualt port is set to 5050, and end points is `http://Localhost:5050/invocations`. 

Or Start the serving endpoint locally through CLI:
```bash
mlflow models serve --model-uri models:/hourly_pm25_24h_service/latest --host 0.0.0.0 --port 5050 --no-conda
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

> **Tip**: The Python approach using processed data is recommended — it guarantees columns and types match the model signature exactly.



## Development (editable install)

```sh
.venv/scripts/python.exe -m pip install -e .
```

To undo:
```sh
python -m pip uninstall pypipeline
```

## Notebooks

Experiments and visuals live under `/notebooks`.

## Secrets

Store secrets in `.env`. Example:
```
OPEN_WEATHER_API_KEY=...
```

You may also need Azure Blob Storage credentials for DVC and MLflow artifacts.