import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator


def save_data(data):
    aqi_file_path = r"/opt/airflow/data/raw/shankapark_realtime.csv"
    try:
        if os.path.exists(aqi_file_path):
            # Check if file is empty
            if os.path.getsize(aqi_file_path) == 0:
                aqi = pd.DataFrame(columns=["aqi", "time"])
            else:
                aqi = pd.read_csv(aqi_file_path)
                if aqi.empty:
                    aqi = pd.DataFrame(columns=["aqi", "time"])
            aqi = pd.concat([aqi, data], ignore_index=True)
        else:
            aqi = pd.DataFrame(columns=["aqi", "time"])
            aqi = pd.concat([aqi, data], ignore_index=True)
        aqi.to_csv(aqi_file_path, index=False)
        print("Data saved successfully.")
        
    except Exception as e:
        print(f"Error saving data: {e}")
        

def load_data():
    token = "88370b78ae71f620f8bf5d8ca57bdb1d8d55c4bf"
    url = f"https://api.waqi.info/search/?keyword=nepal&token={token}"
    responese = requests.get(url)

    load = pd.DataFrame(columns=["aqi", "time"])
    if responese.status_code == 200:
        data=responese.json()

        for station in data["data"]:
            if (station["uid"]== 14866):
                aqi=station["aqi"]
                time = station["time"]["stime"]

                l = pd.Series([aqi, time], index=["aqi", "time"])
                load = pd.concat([load, l.to_frame().T], ignore_index=True)#series to df and transpose
    else:
        print("Failed to load data")
    save_data(load)

with DAG(dag_id="aqi_data_dag",
         start_date=datetime(2025, 8, 1),
         schedule='@hourly',
         catchup=False,
         ) as dag:
            
            load_aqi_data = PythonOperator(
                task_id='load_aqi_data',
                python_callable=load_data
            )




