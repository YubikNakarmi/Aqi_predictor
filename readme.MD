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
Mlflow is setup with sqlite database and azure blob storage for artifacts. It is configured in docker-compose.yaml file to mount /db folder as a backend store, and artifacts store is setup with environment variable DEFAULT_ARTIFACT_ROOT (Azure blob storage).

``` python
artifacts_uri = os.environ["DEFAULT_ARTIFACT_ROOT"]
tracking = os.environ["MLFLOW_TRACKING_URI"] or "http://localhost:5000" #automatically logs in sqlite db locally
mlflow.set_tracking_uri(tracking)
```

### Running MLflow server

``` sh
docker compose up mlflow
```
Then access mlflow at http://localhost:5000

## Secrets
.env file was used for storing environment variable for api key

```
OPEN_WEATHER_API_KEY = ----
```
furthermore locally azure blob key was also used for dvc


