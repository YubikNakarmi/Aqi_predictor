import mlflow
from modules import logging_utils
import os
import pandas as pd


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PRED_PROCESSED_PATH = os.environ.get("PRED_PROCESSED_PATH", "/data/processed/hourly/us_paro_hourly/")
PRED_OUTPUT_PATH = os.environ.get("PRED_OUTPUT_PATH", "/data/predictions/hourly/us_paro_hourly/prod")
PERD_OUTPUT_MYSQLURI = os.environ.get("PERD_OUTPUT_MYSQLURI", "")

logger = logging_utils.setup_logging(__name__)


def mlflow_sanity_check()->bool:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.set_experiment("sanity_check")
        with mlflow.start_run(run_name="sanity_check_run"):
            mlflow.log_param("sanity_check", "passed")
        logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)
        return True
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MLflow tracking server at {MLFLOW_TRACKING_URI}: {e}")
    

def pred():

    if not mlflow_sanity_check():
        return
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("xgb_aqi_hourly_prediction")

    with mlflow.start_run(run_name="hourly_prediction_run"):
        model = mlflow.pyfunc.load_model("models:/hourly_pm25_24h_service/latest")
        data = pd.read_parquet(PRED_PROCESSED_PATH+"pred_processed.parquet")
        logger.info("Data preview:\n%s", data.head())
        
        expected_cols = [col.name for col in model.metadata.get_input_schema().inputs]
        logger.info("Expected columns for prediction: %s", expected_cols)
        prediction = model.predict(data.iloc[[1]][expected_cols])
        pred_df = pd.DataFrame(prediction)
        logger.info("Prediction result:\n%s", pred_df)
        pred_df.to_csv(PRED_OUTPUT_PATH+"/predicted_pm25.csv", index=False)
        logger.info("Prediction saved to %s/predicted_pm25.csv", PRED_OUTPUT_PATH)