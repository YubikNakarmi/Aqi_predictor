import pandas as pd
import requests
import os
import click
import datetime as dt
from modules.logging_utils import setup_logging


DEFAULT_AQI_COLUMNS = ["date","o3","pm25","pm10"]

DEFAULT_WEATHER_COLUMNS = ["date","temp","humidity","rain1h","snowfall","windspeed"
                           ,"weather","weather_description"]

DEFAULT_MERGED_COLUMNS = ["date","o3","pm25","pm10","temp","humidity","rain1h","snowfall","windspeed"
                           ,"weather","weather_description"]

OPEN_WEATHER_API_KEY = os.environ.get("OPEN_WEATHER_API_KEY", "84737501867f684f51f05bd39eb89b6c")

logger = setup_logging(__name__)

def save_data(data:pd.DataFrame,
              file_path:str,
              column:list[str] = DEFAULT_MERGED_COLUMNS,
              aqi_file_path:str = None)->pd.DataFrame:
    #container directory
    #aqi_file_path = r"D:\pypipeline\scripts\test.csv" #local directory

    try:
        if os.path.exists(aqi_file_path):
            # Check if file is empty
            if os.path.getsize(aqi_file_path) == 0:
                aqi = pd.DataFrame(columns=column)
            else:
                aqi = pd.read_csv(aqi_file_path)
                if aqi.empty:
                    aqi = pd.DataFrame(columns=column)
            aqi = pd.concat([aqi, data], ignore_index=True)
        else:
            aqi = pd.DataFrame(columns=column)
            aqi = pd.concat([aqi, data], ignore_index=True)
        aqi.to_csv(aqi_file_path, index=False)
        logger.info("Data saved successfully.")
        
    except Exception:
        logger.exception("Error saving data")


def load_weather_data(api:str = OPEN_WEATHER_API_KEY, 
                      lat: float = 27.738065847677174, 
                      lon: float = 85.33533094823635)->pd.DataFrame:
    now = dt.datetime.now()
    past = dt.datetime.now() - dt.timedelta(days=4) #to unix time conversion for api
    res_now = int(dt.datetime.timestamp(now))
    res_past = int(dt.datetime.timestamp(past))
    
    weather_url = (
        "https://history.openweathermap.org/data/2.5/history/city"
        f"?lat={lat}&lon={lon}&type=hour&start={res_past}&end={res_now}&units=metric&appid={api}"
    )
    response = requests.get(weather_url)
    rows = []
    if response.status_code == 200:
        data = response.json()
        for result in data["list"]:
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

            rows.append({"date": time, "temp": temp, "humidity": humidity, 
                         "windspeed": windspeed, "weather": weather, "weather_description": weather_description, 
                         "rain1h": rain, "snowfall": snow})
            logger.info(f"Loaded weather data for time: {time}")
    else:
        logger.error(
            "Failed to load weather data from API response code: %s, returning empty dataframe",
            response.status_code,
        )

    load = pd.DataFrame(rows, columns=DEFAULT_WEATHER_COLUMNS)
    return load


def load_aqi_data(
    aqi_api_key: str = OPEN_WEATHER_API_KEY,
    lat: float = 27.738065847677174,
    lon: float = 85.33533094823635,
    out_path: str = None,
    lookback_hours: int | None = None,
    lookback_days: int = 4,
) -> pd.DataFrame:
    now = dt.datetime.now()
    if lookback_hours is not None:
        past = now - dt.timedelta(hours=lookback_hours)
    else:
        past = now - dt.timedelta(days=lookback_days)
    res_now = int(dt.datetime.timestamp(now))
    res_past = int(dt.datetime.timestamp(past))
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={res_past}&end={res_now}&appid={aqi_api_key}"
    rows = []
    response = requests.get(url) 

    if response.ok:
        data=response.json()
        for data in data["list"]:
            o3=data["components"]["o3"]
            pm25=data["components"]["pm2_5"]
            pm10=data["components"]["pm10"]
            time = dt.datetime.fromtimestamp(data["dt"]).strftime('%Y-%m-%d %H:%M:%S')

            rows.append({"date": time, "o3": o3, "pm25": pm25, "pm10": pm10})
            logger.info(f"Loaded AQI data for time: {time}")
                
    else:
        logger.error(
            "Failed to load AQI data from API response code: %s, returning empty dataframe",
            response.status_code,
        )

    load = pd.DataFrame(rows, columns=DEFAULT_AQI_COLUMNS)
    return load

''' module runner for airflow compose'''
def run(api_key:str,
         dest:str,
        lat: float = 27.738065847677174, 
        lon: float = 85.33533094823635,):

    aqi_load = load_aqi_data(api_key)
    weather_load = load_weather_data(api_key)
    load = pd.merge(aqi_load, weather_load, on='date', how='inner')

    if load.empty:
        logger.error("No data loaded. Exiting.")
        return
    else:
        save_data(load.fillna(0.00),aqi_file_path=dest)


''' cli entry'''
   
@click.command()
@click.option('--api_key', default=None, help='API key for AQI data and weather data')
@click.option('--lat', default=27.738065847677174, help='Latitude for AQI data and weather data')
@click.option('--lon', default=85.33533094823635, help='Longitude for AQI data and weather data')

def main(api_key,lat,lon):#cli script entry
    
    aqi_load = load_aqi_data(api_key, lat, lon)
    weather_load = load_weather_data(api_key, lat, lon)
    load = pd.merge(aqi_load, weather_load, on='date', how='inner')

    if load.empty:
        logger.error("No data loaded. Exiting.")
        return
    else:
        save_data(load.fillna(0.00))

if __name__ == "__main__":
    #api = os.environ.get("API_key")
    main()

   
