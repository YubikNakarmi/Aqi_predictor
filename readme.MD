# AQI predictor with Airflow

This project predicts hourly AQI for the Shankapark station (Kathmandu) using historical and live CSV data. Data is stored in CSV files (no external database required).

## Libraries used
- DVC for data versioning and storage in Azure blob storage
- Airflow for orchestration
- MLflow for model tracking
- Optuna for hyperparameter tuning
- XGBoost for model training

## Setting up

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

### Building docker images

Build the Docker images used by the pipeline:

``` sh
docker build -t data_train:v1 -f scripts/train.dockerfile .
```

``` sh
docker build -t data_predict:v1 -f scripts/predict.dockerfile .
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
python -m pip install -e .
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