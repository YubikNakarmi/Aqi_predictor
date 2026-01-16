# AQI predictor with Airflow

This project uses historical and live data to predict AQI of Shankapark station in Kathmandu. All the data is stored is csv file no exteral database is used. 

## Setting up

### DVC setup

Dvc is setup with azure blob storage
 ``` sh
dvc pull 
dvc get
 ```

### Building docker images

Build all the docker images first for running the ML scripts 

``` sh
docker build -t data_train:v1 -f scripts train.dockerfile .
```

``` sh
docker build -t data_predict:v1 -f scripts predict.dockerfile scripts\predict.dockerfile
```

### Configuring docker-compose
On data_train_dag.py setup the location of root directory. 
```python
host_path = "D:/pypipeline/data"
```

## Development (editable install) 

To make the `modules` package importable while you develop it, install the project in editable mode:

```sh
# from the project root
venv/scripts/python.exe -m pip install -e .
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
Mlflow is setup with sqlite database and azure blob storage for artifacts. It is configured in docker-compose.yaml file to mount /db folder as a backend store (need to pull from dvc first), and artifacts store is setup with environment variable DEFAULT_ARTIFACT_ROOT (Azure blob storage).

``` python
artifacts_uri = os.environ["DEFAULT_ARTIFACT_ROOT"]#need to be set for mlflow server to register artifacts/models
tracking = os.environ["MLFLOW_TRACKING_URI"] or "http://localhost:5000" #automatically logs in sqlite db locally
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
- Parameters and metrics of model tuning with optuna
- Models registered in model registry (all 24 horizons)
  
## Secrets
.env file was used for storing environment variable for api key

```
OPEN_WEATHER_API_KEY = ----
```
furthermore locally azure blob key was also used for dvc

## Notebooks
Notebooks can be found on /notebooks directory, which contains all the experiments and steps for training, datacleaning and visuals.

## Training Strategy
The training is done for each horizon from 1 to 24 hours separately. Each horizon has its own model trained and logged in mlflow model registry. The hyperparameter tuning is done with optuna for each horizon separately as well. The training script can be found on /src/scripts/train_hourly.py
``` sh
python src/scripts/train_hourly.py
```
If you want to run on CLI

