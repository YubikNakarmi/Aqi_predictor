
from datetime import datetime, timedelta
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from docker.types import Mount



with DAG(dag_id="aqi_data_dag",
         start_date=datetime(2026, 1, 1),
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
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='aqi_network',
        mounts=mounts,
        command = "python pipelines/preprocess.py",
    )
    
    tune = DockerOperator(
        task_id='tune_hyperparameters',
        image='aqi-trainer:latest',
        api_version='auto',
        auto_remove='success',
        command='python pipelines/tune.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='aqi_network',
        mounts=mounts,
    )
    
    train = DockerOperator(
        task_id='train_model',
        image='aqi-trainer:latest',
        api_version='auto',
        auto_remove='success',
        command='python pipelines/train.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='aqi_network',
        mounts=mounts,
    )
    
    evaluate = DockerOperator(
        task_id='evaluate_model',
        image='aqi-trainer:latest',
        api_version='auto',
        auto_remove='success',
        command='python pipelines/eval.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='aqi_network',
        mounts=mounts,
    )
    
    # Set task dependencies
    preprocess >> tune >> train >> evaluate

