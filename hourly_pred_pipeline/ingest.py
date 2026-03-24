from scripts import ingestion_hourly
from modules.logging_utils import setup_logging
import os
import datetime
from modules.runtime_metadata import update_pipeline_metadata

logger = setup_logging(__name__)

key = os.environ.get("OPEN_WEATHER_API_KEY", "")
COLUMNS = ["date","o3","pm25","pm10"]
PRED_INGEST_PATH = os.environ.get("PRED_INGEST_PATH", "data/raw/pred_ingestion/us_paro")
INGEST_LOOKBACK_HOURS = int(os.environ.get("INGEST_LOOKBACK_HOURS", 24))
INGEST_METADATA_FILE = os.environ.get("INGEST_METADATA_FILE", "data/metadata/ingestion.json")



def ingest():
    metadata_file = INGEST_METADATA_FILE
    try:
        data = ingestion_hourly.load_aqi_data(#load data using scripts/ingestion_hourly.py
            aqi_api_key= key,
            lookback_hours=INGEST_LOOKBACK_HOURS,
        )
        logger.info("Data preview:\n%s", data.head())

        output_file = PRED_INGEST_PATH + r"/" + datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d") + ".csv"
        ingestion_hourly.save_data(# save data
            data=data,
            file_path=output_file,
            column=COLUMNS
        )
        logger.info("Data saved to %s", output_file)

        update_pipeline_metadata(#metadata update 
            metadata_file,
            {
                "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                "pipeline": "ingestion",
                "stage": "ingest",
                "status": "success",
                "lookback_hours": INGEST_LOOKBACK_HOURS,
                "rows": int(len(data)),
                "output_file": output_file,
                "columns": COLUMNS,
            },
        )
    except Exception as exc:
        update_pipeline_metadata(#failed metadata update
            metadata_file,
            {
                "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                "pipeline": "ingestion",
                "stage": "ingest",
                "status": "failed",
                "lookback_hours": INGEST_LOOKBACK_HOURS,
                "error": str(exc),
            },
        )
        raise
    

if __name__ == "__main__":
    ingest()