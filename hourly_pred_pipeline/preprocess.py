import pandas as pd
import os
from modules.data_hourly_preprocessing import DataCleaner
from scripts import ingestion_hourly
from modules.logging_utils import setup_logging
logger = setup_logging(__name__)


key = os.environ.get("OPEN_WEATHER_API_KEY", "")

COLUMNS = ["date","o3","pm25","pm10"]
PRED_INGEST_PATH = os.environ.get("PRED_INGEST_PATH", "/pypipeline/data/raw/pred_ingestion/us_paro")
PRED_PROCESSED_PATH = os.environ.get("PRED_PROCESSED_PATH", "/pypipeline/data/processed/hourly/us_paro_hourly/")



def preprocess():

    df = pd.read_csv(PRED_INGEST_PATH+r"us_paro_hourly_ingest.csv")

    clean = DataCleaner()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.drop(columns=["pm10"], inplace=True)
    df_1 = clean.add_time_features(df)
    df_2 = clean.add_missing_flags(df_1)
    df_2["was_imputed"] = 0
    df_3 = clean.add_gap_length(df_2)
    df_3 = clean.add_segmentation(df_3)

    df_4 = clean.engineer_features(df_3)
    df_4.to_parquet(PRED_PROCESSED_PATH+"pred_processed.parquet")

    

if __name__ == "__main__":
    preprocess()


