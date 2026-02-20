from scripts import ingestion_hourly
from modules.logging_utils import setup_logging
import os

logger = setup_logging(__name__)

key = os.environ.get("OPEN_WEATHER_API_KEY", "")
COLUMNS = ["date","o3","pm25","pm10"]
PRED_INGEST_PATH = os.environ.get("PRED_INGEST_PATH", "data/raw/pred_ingestion/us_paro")



def ingest():
    data = ingestion_hourly.load_aqi_data(
        aqi_api_key= key,
        lookback_hours=24,
    )
    logger.info("Data preview:\n%s", data.head())\
    
    ingestion_hourly.save_data(
        data=data,
        file_path=PRED_INGEST_PATH+r"/us_paro_hourly_ingest.csv",
        column=COLUMNS
)
    

if __name__ == "__main__":
    ingest()