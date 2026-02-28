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
  - [Training Pipeline](#training-pipeline)
    - [Preprocessing](#preprocessing)
    - [Tuning](#tuning)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Serving](#serving)
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

### Serving
`serve.py` serves the selected model using MLflow's built-in model serving capabilities. The model is loaded from MLflow Model Registry and exposed via a REST API endpoint. This utilize custom pyfunc model that combines all the 24 horizon models into one endpoint, where you can specify the horizon as a query parameter.


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