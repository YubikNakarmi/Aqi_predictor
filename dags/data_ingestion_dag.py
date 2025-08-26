
from datetime import datetime, timedelta
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator


with DAG(dag_id="aqi_data_dag",
         start_date=datetime(2025, 8, 1),
         schedule='@hourly',
         catchup=False,
         ) as dag:
            fetch_data = DockerOperator(
                task_id='run_data_processing',
                image='data_ingestion:latest',
                api_version='auto',
                command = 'python /opt/airflow/scripts/ingestion.py',                
            )




