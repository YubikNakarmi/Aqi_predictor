
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

host_path = "/d/pypipeline/data" #environment variable
OPEN_WEATHER_API_KEY = os.environ.get("OPEN_WEATHER_API_KEY", "")
PRED_INGEST_PATH = os.environ.get("PRED_INGEST_PATH", "/pypipeline/data/raw/pred_ingestion/us_paro")
DEFAULT_AQI_COLUMNS = ["date","o3","pm25"]


key = os.environ.get("OPEN_WEATHER_API_KEY", "")

COLUMNS = ["date","o3","pm25","pm10"]

logger = setup_logging(__name__)



def ingest():
    data = ingestion_hourly.load_aqi_data(
        aqi_api_key= key,
        lookback_hours=24,
    )
    logger.info("Data preview:\n%s", data.head())\
    
    ingestion_hourly.save_data(
        data=data,
        file_path=PRED_INGEST_PATH+"/us_paro_hourly_ingest.csv",
        columns=COLUMNS
)




with DAG(dag_id="aqi_data_dag",
         start_date=datetime(2025, 8, 1),
         schedule='@daily',
         catchup=False,
         ) as dag:
            fetch_data = PythonOperator(
                task_id='fetch_and_load',
                python_callable = ingest
                # container env key
            )

            

            



