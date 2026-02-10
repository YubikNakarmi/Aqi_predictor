# AQI predictor with Airflow

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
    - [Environment configuration](#environment-configuration)
    - [DVC setup](#dvc-setup)
  - [Run](#run)
  - [MLflow](#mlflow)
  - [Training](#training)
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

### Environment configuration

Required in `.env` (example names):
- `OPEN_WEATHER_API_KEY`
- `AZURE_STORAGE_CONNECTION_STRING`
- `MYSQLURL`
- `MLFLOW_TRACKING_URI`

Local MySQL on host (Windows/Mac):
```dotenv
MYSQL_HOST=host.docker.internal
MYSQLURL=mysql+pymysql://root:your_password@host.docker.internal:3306/mlflow_backend
```

### DVC setup

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

## Run

```sh
cp .env.template .env
dvc pull
docker compose up
```

Run only MLflow:
```sh
docker compose up mlflow
```

## MLflow

The Docker setup uses MySQL for the backend store and Azure Blob Storage for artifacts. The tracking server is exposed at http://localhost:5000.

MLflow stores:
- Parameters and metrics for each training run
- Optuna tuning results
- Models registered per horizon

## Training

Hourly training runs per horizon (1-24) with separate models and tuning. Pipeline scripts live under `hourly_train_pipeline/`.

Key environment variables:
```bash
export TARGET_COLS="pm25,o3,pm10"  # comma-separated list for preprocessing
export TARGET_COL="pm25"           # target for training
export VALUE_MIN=0
export VALUE_MAX=500
export HORIZON=24
```

Example runs:
```bash
python hourly_train_pipeline/preprocess.py
python hourly_train_pipeline/tune.py
python hourly_train_pipeline/train.py
python hourly_train_pipeline/eval.py
```


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