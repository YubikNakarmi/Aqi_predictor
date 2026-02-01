
from datetime import datetime, timedelta
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
import os
from scripts.ingestion_hourly import load_aqi_data as ingestion_run

host_path = "/d/pypipeline/data" #environment variable
OPEN_WEATHER_API_KEY = os.environ.get("OPEN_WEATHER_API_KEY", "")

with DAG(dag_id="aqi_data_dag",
         start_date=datetime(2025, 8, 1),
         schedule='@daily',
         catchup=False,
         ) as dag:
            fetch_data = PythonOperator(
                task_id='fetch_and_load',
                python_callable=ingestion_run(aqi_api_key=OPEN_WEATHER_API_KEY,),
                # container env key
            )

            environments = {
                'OPEN_WEATHER_API_KEY': os.environ.get('OPEN_WEATHER_API_KEY'),}

            pred = DockerOperator(
                task_id='run_data_processing',
                image='data_train:v1',#built image name
                api_version='auto',
                command = 'python train.py',    
                docker_url='unix://var/run/docker.sock',#connection to host docker daemon 
                container_name='ingest',
                auto_remove="force",#auto removes container to avoid conflicts
                mounts=[Mount(source=host_path, target='/opt/airflow/data', type='bind')],#mounting external volume
                environment=environments,#setting environment variable inside container
          
            )




