# AQI predictor with Airflow

This project predicts hourly AQI for the Shankapark station (Kathmandu) using historical and live CSV data. Data is stored in CSV files (no external database required).

## Table of Contents
- [Libraries used](#libraries-used)
- [Setting up](#setting-up)
  - [Environment Configuration](#environment-configuration)
  - [DVC setup](#dvc-setup)
  - [Building docker images](#building-docker-images)
  - [Configuring docker-compose](#configuring-docker-compose)
- [Development (editable install)](#development-editable-install)
- [Running](#running)
- [MLflow](#mlflow)
  - [Setup](#setup)
  - [Running MLflow server](#running-mlflow-server)
  - [What mlflow stores](#what-mlflow-stores)
- [Notebooks](#notebooks)
- [Training Strategy](#training-strategy)
  - [Customizable Target Training](#customizable-target-training)
  - [Environment Variables for Customization](#environment-variables-for-customization)
  - [Quick Usage Examples](#quick-usage-examples)
  - [Python API Usage](#python-api-usage)
  - [Common Value Ranges by Pollutant](#common-value-ranges-by-pollutant)
  - [Model Naming Convention](#model-naming-convention)
  - [Running from CLI](#running-from-cli)
- [Secrets](#secrets)

## Libraries used
- DVC for data versioning and storage in Azure blob storage
- Airflow for orchestration
- MLflow for model tracking
- Optuna for hyperparameter tuning
- XGBoost for model training

## Setting up

### Environment Configuration

1. Copy the environment template to create your `.env` file:
   ```sh
   cp .env.template .env
   ```

2. Edit `.env` and fill in your own values:
   - **API Keys**: `OPEN_WEATHER_API_KEY` for OpenWeather data
   - **Azure Storage**: `AZURE_STORAGE_CONNECTION_STRING` and `MLFLOW_ARTIFACT_ROOT`
   - **Database**: `MYSQL_HOST`, `MYSQL_PW`, `MYSQLURL`, and `OPTUNA_PATH`
   - **Paths**: Adjust `DATA_PROCESSED_PATH` and `PREDICTIONS_DIR` as needed

3. Example for local setup with MySQL on host:
   ```dotenv
   MYSQL_HOST=host.docker.internal
   MYSQLURL=mysql+pymysql://root:your_password@host.docker.internal:3306/mlflow_backend
   ```

   **Note**: Use `host.docker.internal` on Windows/Mac to access services on your host machine, or use the service name if MySQL is in a container.

### DVC setup
config remote path at /.dvc/config
``` ini
[[core]
    remote = azureremote
    autostage = true
['remote "azureremote"']
    url = azure://aqi/static
```

DVC is set up with Azure Blob Storage.
 ``` sh
dvc pull 
dvc get
 ```
 #### Tracked files with DVC
- data/raw/
- data/processed/
- db/mlflow.db
- db/optuna_study.db



### Building docker images

Build the Docker images used by the pipeline:

``` sh
docker compose -f .\docker-compose.train.yaml build mlflow
docker compose -f .\docker-compose.train.yaml up -d mlflow

```

### Configuring docker-compose
In `data_train_dag.py`, set the root data directory (example):
```python
host_path = "D:/pypipeline/data"
```

## Development (editable install) 

To make the `modules` package importable while you develop it, install the project in editable mode:

```sh
# from the project root
.venv/scripts/python.exe -m pip install -e .
```

What this does:
- Installs the package in "editable" mode so Python uses the project source directly.
- Changes you make to files (e.g., `modules/`) are available immediately without reinstalling.

To undo:

```sh
python -m pip uninstall pypipeline
```

Notes:
- Ensure your virtual environment is active before running the commands.
- If you use a notebook kernel, restart the kernel after installation.

## Running
Run the following command on root directory
``` sh 
docker compose up
```

## MLflow 

### Setup
MLflow is set up with a SQLite backend and Azure Blob Storage for artifacts. The configuration (in `docker-compose.yaml`) mounts the `db/` folder as the backend store (after running `dvc pull`), and uses the `DEFAULT_ARTIFACT_ROOT` environment variable for artifact storage.

``` python
artifacts_uri = os.environ.get("DEFAULT_ARTIFACT_ROOT")  # must be set so MLflow can register artifacts/models
tracking = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")  # default to local server
mlflow.set_tracking_uri(tracking)
```

### Running MLflow server

``` sh
docker compose up mlflow
```
Then access mlflow at http://localhost:5000

Or in CLI if you want to run without docker
``` sh
mlflow server --backend-store-uri sqlite:///db/mlflow.db --default-artifact-root wasbs://mlflow@yubikaqidata.blob.core.windows.net --port 5000
```
### What mlflow stores
- Parameters and metrics of each training run
- Parameters and metrics of model tuning with Optuna
- Models registered in model registry (all 24 horizons)
  
## Secrets
.env file was used for storing environment variable for api key

```
OPEN_WEATHER_API_KEY = ----
```
Additionally, you may need the Azure Blob storage key locally for DVC configuration.

## Notebooks
Notebooks can be found on /notebooks directory, which contains all the experiments and steps for training, datacleaning and visuals.

## Training Strategy
The training is done for each horizon from 1 to 24 hours separately. Each horizon has its own model trained and logged in mlflow model registry. The hyperparameter tuning is done with optuna for each horizon separately as well. The training script can be found on /src/scripts/train_hourly.py

### Customizable Target Training
The library supports training models for any pollutant target (PM2.5, O3, PM10, NO2, etc.) with configurable value ranges.

#### Environment Variables for Customization
```bash
# For preprocessing - create targets for multiple pollutants
export TARGET_COLS="pm25,o3,pm10"  # Comma-separated list
export HORIZON=24
export DATA_PROCESSED_PATH="data/processed/hourly/..."

# For training - train models for a specific pollutant
export TARGET_COL="pm25"           # Single target: pm25, o3, pm10, etc.
export VALUE_MIN=0                 # Minimum valid value
export VALUE_MAX=500               # Maximum valid value
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

#### Quick Usage Examples

**Train PM2.5 Models (default):**
```bash
export TARGET_COL="pm25"
export VALUE_MIN=0
export VALUE_MAX=500
python hourly_pipeline/train.py
```

**Train O3 (Ozone) Models:**
```bash
export TARGET_COL="o3"
export VALUE_MIN=0
export VALUE_MAX=300
python hourly_pipeline/train.py
```

**Train PM10 Models:**
```bash
export TARGET_COL="pm10"
export VALUE_MIN=0
export VALUE_MAX=600
python hourly_pipeline/train.py
```

#### Python API Usage
```python
from scripts.train_hourly import train, tune, clean_and_target

# Prepare data for multiple targets
train_df, val_df, test_df = clean_and_target(
    horizon=24,
    train=train_raw,
    val=val_raw,
    test=test_raw,
    target_cols=["pm25", "o3"]  # Create targets for both pollutants
)

# Tune hyperparameters for specific target
best_params = tune(
    df_train=train_df,
    df_val=val_df,
    trials=60,
    target_col="pm25",
    value_range=(0, 500),
    horizons=24
)

# Train models for specific target
models, metrics, sig = train(
    horizons=24,
    best_params=best_params,
    df_train=train_df,
    df_val=val_df,
    target_col="pm25",
    value_range=(0, 500)
)
```

#### Common Value Ranges by Pollutant
| Pollutant | Range | Unit | Notes |
|-----------|-------|------|-------|
| PM2.5 | (0, 500) | AQI | Fine particles - default |
| PM10 | (0, 600) | AQI | Coarser particles |
| O3 | (0, 300) | ppb | Ozone concentration |
| NO2 | (0, 200) | ppb | Nitrogen dioxide |
| SO2 | (0, 200) | ppb | Sulfur dioxide |
| CO | (0, 50) | ppm | Carbon monoxide |

#### Model Naming Convention
Models are automatically named with the target pollutant:
- PM2.5: `pm25_plus_1h_model`, `pm25_plus_2h_model`, ..., `pm25_plus_24h_model`
- O3: `o3_plus_1h_model`, `o3_plus_2h_model`, ..., `o3_plus_24h_model`
- PM10: `pm10_plus_1h_model`, ..., `pm10_plus_24h_model`

MLflow experiments are also named accordingly: `xgb_pm25_hourly_training`, `xgb_o3_hourly_training`, etc.

#### Running from CLI
``` sh
python src/scripts/train_hourly.py
```
If you want to run on CLI

## Secrets
.env file was used for storing environment variable for api key

### env variables required for project
- OPEN_WEATHER_API_KEY : API key for open weather map
- AZURE_STORAGE_CONNECTION_STRING : Azure blob storage connection string for dvc and mlflow artifacts
- MLFLOW_TRACKING_URI : mlflow tracking server uri