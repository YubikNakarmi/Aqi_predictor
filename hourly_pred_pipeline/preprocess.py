import pandas as pd
import os
import datetime
try:
    from src.modules.data_hourly_preprocessing import DataCleaner
    from src.modules.logging_utils import setup_logging
    from src.modules.runtime_metadata import update_pipeline_metadata
except ImportError:
    from modules.data_hourly_preprocessing import DataCleaner
    from modules.logging_utils import setup_logging
    from modules.runtime_metadata import update_pipeline_metadata

logger = setup_logging(__name__)


key = os.environ.get("OPEN_WEATHER_API_KEY", "")

COLUMNS = ["date","o3","pm25","pm10"]
PRED_INGEST_PATH = os.environ.get("PRED_INGEST_PATH", "data/raw/pred_ingestion/us_paro")
PRED_PROCESSED_PATH = os.environ.get("PRED_PROCESSED_PATH", "data/processed/pred_ingestion/us_paro")



def preprocess():
    metadata_file = "data/metadata/prediction.json"
    try:
        input_file = PRED_INGEST_PATH + r"/" + datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d") + ".csv"
        df = pd.read_csv(input_file)

        clean = DataCleaner(target_features=["pm25", "o3"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.drop(columns=["pm10"], inplace=True)

        df_4 = clean.run_feature_engineering(df, target_features=["pm25", "o3"])
        output_file = PRED_PROCESSED_PATH + r"/" + datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d") + ".csv"
        df_4.to_csv(output_file)

        update_pipeline_metadata(
            metadata_file,
            {
                "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
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
                "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                "pipeline": "prediction",
                "stage": "preprocess",
                "status": "failed",
                "error": str(exc),
            },
        )
        raise

    

if __name__ == "__main__":
    preprocess()


