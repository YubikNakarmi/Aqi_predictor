
from datetime import datetime, timedelta
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
import os
from scripts.ingestion_hourly import save_data, load_aqi_data

host_path = "/d/pypipeline/data" #environment variable
OPEN_WEATHER_API_KEY = os.environ.get("OPEN_WEATHER_API_KEY", "")
PRED_INGEST_PATH = os.environ.get("PRED_INGEST_PATH", "data/ingestion/hourly/us_paro_hourly/")
DEFAULT_AQI_COLUMNS = ["time","o3","pm2_5","pm10"]


def ingest():
        data = load_aqi_data(aqi_api_key=OPEN_WEATHER_API_KEY)
        save_data(data = data,
                  column=DEFAULT_AQI_COLUMNS, aqi_file_path=PRED_INGEST_PATH)



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

            

            



