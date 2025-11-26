
from datetime import datetime, timedelta
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
import os
import subprocess

def run_external():
    subprocess.run(['python', '/opt/airflow/scripts/ingestion_hourly.py',
                    '--api_key', os.environ.get("API_KEY")])


with DAG(dag_id="aqi_data_dag",
         start_date=datetime(2025, 8, 1),
         schedule='@daily',
         catchup=False,
         ) as dag:
            fetch_load = PythonOperator(
                task_id='fetch_and_load',
                python_callable=run_external)
            
            predict = DockerOperator()

