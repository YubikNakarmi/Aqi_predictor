
from datetime import datetime, timedelta
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
import os
import subprocess
from modules.data_hourly_preprocessing import clean_data_hourly


with DAG(dag_id="aqi_data_dag",
         start_date=datetime(2025, 8, 1),
         schedule='@monthly',
         catchup=False,
         ) as dag:
            clean_data = PythonOperator(
                task_id='clean_data',
                python_callable=clean_data_hourly
            )
            
            predict = DockerOperator()

