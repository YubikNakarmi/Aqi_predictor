
from datetime import datetime, timedelta
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from docker.types import Mount
import os


with DAG(dag_id="aqi_train_dag",
         start_date=datetime(2026, 1, 1),
         schedule='@monthly',
         catchup=False,
         ) as dag:

    # On Windows with Docker Desktop, use forward slashes
    # The paths should exist on the host where Docker daemon is running
    #PROJ_DIR = os.environ.get('AIRFLOW_PROJ_DIR', '/d/pypipeline')

    PROJ_DIR = '/d/pypipeline'
    DB_PATH = f"{PROJ_DIR}/db"
    DATA_PATH = f"{PROJ_DIR}/data"

    mounts = [
        Mount(source=DB_PATH, target="/db", type="bind"),
        Mount(source=DATA_PATH, target="/data", type="bind"),
    ]
    
    # Shared environment variables for all trainer tasks
    shared_env = {
        'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
        'MYSQL_HOST': os.environ.get('MYSQL_HOST', 'host.docker.internal'),
        'MYSQL_PW': os.environ.get('MYSQL_PW', ''),
        'MYSQL_USER': os.environ.get('MYSQL_USER', 'root'),
        'MYSQLURL': os.environ.get('MYSQLURL', ''),
        'AZURE_STORAGE_CONNECTION_STRING': os.environ.get('AZURE_STORAGE_CONNECTION_STRING', ''),
        'HORIZON': os.environ.get('HORIZON', '24'),
        'TRAIN_SPLIT': os.environ.get('TRAIN_SPLIT', '0.7'),
        'VAL_SPLIT': os.environ.get('VAL_SPLIT', '0.15'),
        'TEST_SPLIT': os.environ.get('TEST_SPLIT', '0.15'),
        'PREDICTIONS_DIR': os.environ.get('PREDICTIONS_DIR', 'data/predictions/hourly/us_paro_hourly/'),
        'ARTIFACTS_PATH': os.environ.get('ARTIFACTS_PATH', 'data/artifacts/'),
        'DATA_PROCESSED_PATH': os.environ.get('DATA_PROCESSED_PATH', 'data/processed/hourly/us_paro_hourly/'),
        'STATION_NAME': os.environ.get('STATION_NAME', 'us_paro_hourly'),
        'TARGET_COLS': os.environ.get('TARGET_COLS', 'pm25,o3'),
        'TARGET_COL': os.environ.get('TARGET_COL', 'pm25'),
        'VALUE_MIN': os.environ.get('VALUE_MIN', '0'),
        'VALUE_MAX': os.environ.get('VALUE_MAX', '500'),
    }
    
    preprocess = DockerOperator(
        task_id='preprocess_data',
        image='aqi-trainer:latest',
        api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='pypipeline_aqi_network',
        mounts=mounts,
        command = "python hourly_pipeline/preprocess.py",
        environment=shared_env,
    )
    
    tune = DockerOperator(
        task_id='tune_hyperparameters',
        image='aqi-trainer:latest',
        api_version='auto',
        auto_remove='success',
        command='python hourly_pipeline/tune.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='pypipeline_aqi_network',
        mounts=mounts,
        environment={**shared_env, 'OPTUNA_PATH': 'sqlite:////db/optuna.db'},
    )
    
    train = DockerOperator(
        task_id='train_model',
        image='aqi-trainer:latest',
        api_version='auto',
        auto_remove='success',
        command='python hourly_pipeline/train.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='pypipeline_aqi_network',
        mounts=mounts,
        environment=shared_env,
    )shared_envate = DockerOperator(
        task_id='evaluate_model',
        image='aqi-trainer:latest',
        api_version='auto',
        auto_remove='success',
        command='python hourly_pipeline/eval.py',
        environment={
            'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
        },
        docker_url='unix://var/run/docker.sock',
        network_mode='pypipeline_aqi_network',
        mounts=mounts,
    )
    
    # Set task dependencies
    preprocess >> tune >> train >> evaluate

