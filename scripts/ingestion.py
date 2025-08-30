import pandas as pd
import requests
import os
import click



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
    url = f"https://api.waqi.info/search/?keyword=nepal&token={aqi_api_key}"
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
        print("Failed to aqi load data")
    return load
        
@click.command()
@click.option('--weather_api_key', default=None, help='API key for weather data')
@click.option('--aqi_api_key', default=None, help='API key for AQI data')

def load_data(weather_api_key=None, aqi_api_key=None):
    load = load_aqi_data(aqi_api_key)
   
    save_data(load)



if __name__ == "__main__":
    load_data()