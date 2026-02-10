import pandas as pd
import requests
import os
import json
from modules.data_hourly_preprocessing import DataCleaner
from scripts import ingestion_hourly
from modules.logging_utils import setup_logging
logger = setup_logging(__name__)


key = os.environ.get("OPEN_WEATHER_API_KEY", "")

COLUMNS = ["date","o3","pm25","pm10"]



def ingest():
    data = ingestion_hourly.load_aqi_data(
        aqi_api_key= key,
        lookback_hours=24,
    )
    logger.info("Data preview:\n%s", data.head())\
    
    ingestion_hourly.save_data(
        data=data,
        file_path=r"D:\pypipeline\data\raw\pred_ingestion\us_paro\us_paro_hourly.csv"
        ,column=COLUMNS)

def preprocess():

    df = pd.read_csv(r"D:\pypipeline\data\raw\pred_ingestion\us_paro\us_paro_hourly.csv")
    df = df.drop(columns=["pm10,"])
    cleaner = DataCleaner()
    df_1 = cleaner.add_time_features(df)
    df_2 = cleaner.add_missing_flags(df_1)
    


if __name__ == "__main__":
    ingest()


