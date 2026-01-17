
from datetime import datetime, timedelta
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
import os
import subprocess
from modules.data_hourly_preprocessing import clean_data_hourly


with DAG(dag_id="aqi_data_dag",
         start_date=datetime(2025, 8, 1),
         schedule='@monthly',
         catchup=False,
         ) as dag:
    
    DB_PATH= r"/db"
    DATA_PATH= r"/data"

    mounts = [
        Mount(source = DB_PATH, target = "/db", type="bind"),
        Mount(source = DATA_PATH, target = "/data", type="bind"),
    ]
    preprocess = DockerOperator(
        task_id='preprocess_data',
        image='aqi-trainer:latest',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='aqi_network',
        mounts=mounts,
        command = "python hourly_pipeline/preprocess.py",
    ),
    tune = DockerOperator(
        task_id='tune_hyperparameters',
        image='aqi-trainer:latest',
        api_version='auto',
        auto_remove=True,
        command='python -m hourly_pipeline.tune',
        docker_url='unix://var/run/docker.sock',
        network_mode='aqi_network',
        mounts=mounts,
    ),
    train = DockerOperator(
        task_id='train_model',
        image='aqi-trainer:latest',
        api_version='auto',
        auto_remove=True,
        command='python -m hourly_pipeline.train',
        docker_url='unix://var/run/docker.sock',
        network_mode='aqi_network',
        mounts=mounts,
    )
            
    

