import pandas as pd
import requests
import os
import click
import datetime as dt
import logging

DEFAULT_COLUMNS = ["time","o3","pm2_5","pm10"]


def save_data(data:pd.DataFrame)->pd.DataFrame:

    aqi_file_path = r"/opt/airflow/data/raw/shankapark_realtime.csv" #container directory

    try:
        if os.path.exists(aqi_file_path):
            # Check if file is empty
            if os.path.getsize(aqi_file_path) == 0:
                aqi = pd.DataFrame(columns=DEFAULT_COLUMNS)
            else:
                aqi = pd.read_csv(aqi_file_path)
                if aqi.empty:
                    aqi = pd.DataFrame(columns=DEFAULT_COLUMNS)
            aqi = pd.concat([aqi, data], ignore_index=True)
        else:
            aqi = pd.DataFrame(columns=DEFAULT_COLUMNS)
            aqi = pd.concat([aqi, data], ignore_index=True)
        aqi.to_csv(aqi_file_path, index=False)
        print("Data saved successfully.")
        
    except Exception as e:
        print(f"Error saving data: {e}")


def load_aqi_data(aqi_api_key:str = not None ) -> pd.DataFrame:

    now = dt.datetime.now()
    past = dt.datetime.now() - dt.timedelta(days=4)
    res_now = int(dt.datetime.timestamp(now))
    res_past = int(dt.datetime.timestamp(past))
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat=27.738065847677174&lon=85.33533094823635&start={res_past}&end={res_now}&appid={aqi_api_key}"
    weather_url = f"https://history.openweathermap.org/data/2.5/history/city?lat=27.738065847677174&lon=85.33533094823635&type=hour&start={res_past}&end={res_now}&appid={aqi_api_key}"
    
    try:
        response = requests.get(url)
        weather_response = requests.get(weather_url)

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty

    rows = []
    if response.status_code == 200:
        data=response.json()
        for data in data["list"]:
                o3=data["components"]["o3"]
                pm25=data["components"]["pm2_5"]
                pm10=data["components"]["pm10"]
                time = dt.datetime.fromtimestamp(data["dt"]).strftime('%Y-%m-%d %H:%M:%S')

                rows.append({"time": time, "o3": o3, "pm2_5": pm25, "pm10": pm10})
                
    else:
        print("Failed to aqi load data")

    load = pd.DataFrame(rows, columns=DEFAULT_COLUMNS)

    return load

   
@click.command()
@click.option('--aqi_key', default=None, help='API key for AQI data')
@click.option('--weather_key', default=None, help='API key for weather data')

def load_data(aqi_key, weather_key):
    
    load = load_aqi_data(aqi_key)
    if load.empty:
        print("No data loaded. Exiting.")
        return
    else:
        save_data(load)

if __name__ == "__main__":
    load_data()