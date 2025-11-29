import pandas as pd
import requests
import os
import click
import datetime as dt
import logging

DEFAULT_AQI_COLUMNS = ["time","o3","pm2_5","pm10"]
DEFAULT_WEATHER_COLUMNS = ["time","temp","humidity","rain1h","snowfall","windspeed"
                           ,"weather","weather_description"]

DEFAULT_MERGED_COLUMNS = DEFAULT_AQI_COLUMNS + DEFAULT_WEATHER_COLUMNS[1:]

LOG_FORMAT = '%(asctime)s %(levelname)s %(filename)s: %(lineno)d %(message)s'

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)





def save_data(data:pd.DataFrame)->pd.DataFrame:

    aqi_file_path = r"/opt/airflow/data/raw/shankapark_realtime.csv" #container directory

    try:
        if os.path.exists(aqi_file_path):
            # Check if file is empty
            if os.path.getsize(aqi_file_path) == 0:
                aqi = pd.DataFrame(columns=DEFAULT_MERGED_COLUMNS)
            else:
                aqi = pd.read_csv(aqi_file_path)
                if aqi.empty:
                    aqi = pd.DataFrame(columns=DEFAULT_MERGED_COLUMNS)
            aqi = pd.concat([aqi, data], ignore_index=True)
        else:
            aqi = pd.DataFrame(columns=DEFAULT_MERGED_COLUMNS)
            aqi = pd.concat([aqi, data], ignore_index=True)
        aqi.to_csv(aqi_file_path, index=False)
        print("Data saved successfully.")
        
    except Exception as e:
        print(f"Error saving data: {e}")



def load_weather_data(api)->pd.DataFrame:
    now = dt.datetime.now()
    past = dt.datetime.now() - dt.timedelta(days=4) #to unix time conversion for api
    res_now = int(dt.datetime.timestamp(now))
    res_past = int(dt.datetime.timestamp(past))
    
    weather_url = f"https://history.openweathermap.org/data/2.5/history/city?lat=1.5533&lon=110.3592&type=hour&start={res_past}&end={res_now}&appid={api}"
    response = requests.get(weather_url).json()
    rows = []
    if response.status_code == 200:
        for result in response["list"]:
        
            time = dt.datetime.fromtimestamp(result["dt"]).strftime('%Y-%m-%d %H:%M:%S')
            temp = result["main"]["temp"]
            humidity = result["main"]["humidity"]
            windspeed = result["wind"]["speed"]
            weather = result["weather"][0]["main"]
            weather_description = result["weather"][0]["description"]

            try:
                rain = result["rain"]["1h"]
            except KeyError:
                rain = 0.0
            try:
                snow = result["snow"]["1h"]
            except KeyError:
                snow = 0.0

            rows.append({"time": time, "temp": temp, "humidity": humidity, 
                         "windspeed": windspeed, "weather": weather, "weather_description": weather_description, 
                         "rain": rain, "snow": snow})
    else:
        print("Failed to load weather data")

    load = pd.DataFrame(rows, columns=DEFAULT_WEATHER_COLUMNS)
    return load


def load_aqi_data(aqi_api_key: str = None) -> pd.DataFrame:

    now = dt.datetime.now()
    past = dt.datetime.now() - dt.timedelta(days=4)
    res_now = int(dt.datetime.timestamp(now))
    res_past = int(dt.datetime.timestamp(past))
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat=27.738065847677174&lon=85.33533094823635&start={res_past}&end={res_now}&appid={aqi_api_key}"
    rows = []
    response = requests.get(url) 

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


    load = pd.DataFrame(rows, columns=DEFAULT_AQI_COLUMNS)

    return load


   
@click.command()
@click.option('--api_key', default=None, help='API key for AQI data and weather data')

def main(api_key):
    
    aqi_load = load_aqi_data(api_key)
    weather_load = load_weather_data(api_key)
    load = pd.merge(aqi_load, weather_load, on='time', how='inner')

    if load.empty:
        print("No data loaded. Exiting.")
        return
    else:
        save_data(load)

if __name__ == "__main__":
    main()
    api = os.environ.get("API_key")
   
