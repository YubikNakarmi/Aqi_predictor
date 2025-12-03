
from datetime import datetime, timedelta
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
import os

host_path = "D:/pypipeline/data" #environment variable

with DAG(dag_id="aqi_data_dag",
         start_date=datetime(2025, 8, 1),
         schedule='@weekly',
         catchup=False,
         ) as dag:
            fetch_data = DockerOperator(
                task_id='run_data_processing',
                image='data_train:v1',#built image name
                api_version='auto',
                command = 'python train.py',    
                docker_url='unix://var/run/docker.sock',#connection to host docker daemon 
                mount_tmp_dir=False,   
                container_name='ingest',
                auto_remove="force",#auto removes container to avoid conflicts
                mounts=[Mount(source=host_path, target='/opt/airflow/data', type='bind')],#mounting external volume
                environment={"AQI_API_KEY":"88370b78ae71f620f8bf5d8ca57bdb1d8d55c4bf",
                             "WEATHER_API_KEY":"6bcbef49bee949749e6130656252808"},#setting environment variable inside container
          
            )




