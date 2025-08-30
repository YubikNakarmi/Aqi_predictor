
from datetime import datetime, timedelta
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import os
import subprocess

def run_external():
    subprocess.run(['python', '/opt/airflow/scripts/ingestion.py',
                    '--weather_api_key', os.environ.get("WEATHER_API_KEY"),
                    '--aqi_api_key', os.environ.get("AQI_API_KEY")])


with DAG(dag_id="aqi_data_dag",
         start_date=datetime(2025, 8, 1),
         schedule='@hourly',
         catchup=False,
         ) as dag:
            fetch_load = PythonOperator(
                task_id='fetch_and_load',
                python_callable=run_external)

