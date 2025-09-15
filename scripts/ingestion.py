import pandas as pd
import requests
import os
import click
import datetime as dt



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

def load_weather_data(weather_api_key):
    url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q=kathmandu&aqi=no"
    

def load_aqi_data(aqi_api_key):

    now = dt.datetime.now()
    past = dt.datetime.now() - dt.timedelta(days=3)
    res_now = int(dt.datetime.timestamp(now))
    res_past = int(dt.datetime.timestamp(past))
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat=27.738065847677174&lon=85.33533094823635&start={res_past}&end={res_now}&appid=db714ac9e7a0ea15469110f17a34eccf"
    response = requests.get(url)

    load = pd.DataFrame(columns=["time","o3","pm2_5","pm10"])
    if response.status_code == 200:
        data=response.json()
        for data in data["list"]:
                o3=data["components"]["o3"]
                pm2_5=data["components"]["pm2_5"]
                pm10=data["components"]["pm10"]
                time = dt.datetime.fromtimestamp(data["dt"]).strftime('%Y-%m-%d %H:%M:%S')

                l = pd.DataFrame([time,o3,pm2_5,pm10], columns=["time","o3","pm2_5","pm10"])
                load = pd.concat([load, l], ignore_index=True)#series to df and transpose
    else:
        print("Failed to aqi load data")
    return load



   
# @click.command()
# @click.option('--api_key', default=None, help='API key for AQI data')
# def load_data(weather_api_key=None, aqi_api_key=None):
#     load = load_aqi_data(aqi_api_key)
   
#     save_data(load)

# if __name__ == "__main__":
#     load_data()