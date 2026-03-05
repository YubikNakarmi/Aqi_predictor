import pandas as pd
import os
from modules.data_hourly_preprocessing import DataCleaner
from scripts import ingestion_hourly
from modules.logging_utils import setup_logging
from datetime import datetime, timedelta
from modules.runtime_metadata import update_pipeline_metadata
logger = setup_logging(__name__)


key = os.environ.get("OPEN_WEATHER_API_KEY", "")

COLUMNS = ["date","o3","pm25","pm10"]
PRED_INGEST_PATH = os.environ.get("PRED_INGEST_PATH", "data/raw/pred_ingestion/us_paro")
PRED_PROCESSED_PATH = os.environ.get("PRED_PROCESSED_PATH", "data/processed/pred_ingestion/us_paro")



def preprocess():
    metadata_file = "data/metadata/prediction.json"
    try:
        input_file = PRED_INGEST_PATH + r"/" + datetime.now().strftime("%Y-%m-%d") + ".csv"
        df = pd.read_csv(input_file)

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
        output_file = PRED_PROCESSED_PATH + r"/" + datetime.now().strftime("%Y-%m-%d") + ".csv"
        df_4.to_parquet(output_file)

        update_pipeline_metadata(
            metadata_file,
            {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "pipeline": "prediction",
                "stage": "preprocess",
                "status": "success",
                "input_file": input_file,
                "output_file": output_file,
                "rows": int(len(df_4)),
                "columns": list(df_4.columns),
            },
        )
    except Exception as exc:
        update_pipeline_metadata(
            metadata_file,
            {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "pipeline": "prediction",
                "stage": "preprocess",
                "status": "failed",
                "error": str(exc),
            },
        )
        raise

    

if __name__ == "__main__":
    preprocess()


