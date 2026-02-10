import pandas as pd
import requests
import os
import json
from modules.data_hourly_preprocessing import DataCleaner
from scripts import ingestion_hourly
from modules.logging_utils import setup_logging
logger = setup_logging(__name__)


key = os.environ.get("OPEN_WEATHER_API_KEY", "")


def ingest():
    data = ingestion_hourly.load_aqi_data(
        aqi_api_key= key,
        lookback_hours=24,
    )
    logger.info("Data preview:\n%s", data.head())\



if __name__ == "__main__":
    ingest()


