
from datetime import datetime, timedelta
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
import os
from scripts import ingestion_hourly
from modules.data_hourly_preprocessing import DataCleaner 
from modules.logging_utils import setup_logging
import pandas as pd

host_path = "/d/pypipeline/data" #HOST PATH TO DATA FOLDER, ADJUST AS NEEDED

OPEN_WEATHER_API_KEY = os.environ.get("OPEN_WEATHER_API_KEY", "")
PRED_INGEST_PATH = os.environ.get("PRED_INGEST_PATH", "/pypipeline/data/raw/pred_ingestion/us_paro")
PRED_PROCESSED_PATH = os.environ.get("PRED_PROCESSED_PATH", "/pypipeline/data/processed/pred_ingestion/us_paro")
DEFAULT_AQI_COLUMNS = ["date","o3","pm25"]
INGEST_LOOKBACK_HOURS = os.environ.get("INGEST_LOOKBACK_HOURS", 24)


key = os.environ.get("OPEN_WEATHER_API_KEY", "")

COLUMNS = ["date","o3","pm25","pm10"]

logger = setup_logging(__name__)


mount = Mount(source=host_path, target='/data', type='bind')
environment_vars = {
    'OPEN_WEATHER_API_KEY': OPEN_WEATHER_API_KEY,
    'PRED_INGEST_PATH': PRED_INGEST_PATH,
    'PRED_PROCESSED_PATH': PRED_PROCESSED_PATH,
    'INGEST_LOOKBACK_HOURS': INGEST_LOOKBACK_HOURS,
    "DEFAULT_AQI_COLUMNS": DEFAULT_AQI_COLUMNS
    }

with DAG(dag_id="aqi_data_dag", start_date=datetime(2025, 8, 1),schedule='@daily',catchup=False,
         ) as dag:
            

        fetch_data = DockerOperator(
        task_id='fetch_data',
        image='aqi-ingestion:latest',
                api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='pypipeline_aqi_network',
        command='python hourly_pred_pipeline/ingest.py',
        mounts=mount,
        environment = environment_vars
        )

        preprocess_data = DockerOperator(
        task_id='preprocess_data',
        image='aqi-preprocessing:latest',
                api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='pypipeline_aqi_network',
        command='python hourly_pred_pipeline/preprocess.py',
        mounts=mount,
        environment = environment_vars
        )

        predict = DockerOperator(
        task_id='predict',
        image='aqi-predict:latest',
                api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='pypipeline_aqi_network',
        command='python hourly_pred_pipeline/predict.py',
        mounts=mount,
        environment = environment_vars
        )