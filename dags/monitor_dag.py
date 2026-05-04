from airflow.operators.python import PythonOperator
from pendulum import datetime

from modules.data_hourly_preprocessing import main



with DAG(dag_id="aqi_monitor_dag",
         start_date=datetime(2026, 1, 1),
         schedule='@monthly',
         catchup=False,
         ) as dag:

    monitor = PythonOperator(
        task_id='monitor_model_performance',
        python_callable=main,
    )