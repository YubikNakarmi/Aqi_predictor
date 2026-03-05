import mlflow
from modules import logging_utils
import os
import pandas as pd
from modules.mysql_utils import MySQLUtils
import datetime
from modules.runtime_metadata import update_pipeline_metadata


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PRED_PROCESSED_PATH = os.environ.get("PRED_PROCESSED_PATH", "data/processed/pred_ingestion/us_paro")
PRED_OUTPUT_PATH = os.environ.get("PRED_OUTPUT_PATH", "data/predictions/hourly/us_paro_hourly/prod")
PRED_MYSQLURI = os.environ.get("PRED_MYSQLURI", "mysql+pymysql://root:yubik123@localhost:3306/pypipeline_predictions")
PREDICTION_METADATA_FILE = os.environ.get("PREDICTION_METADATA_FILE", "data/metadata/prediction.json")
logger = logging_utils.setup_logging(__name__)


def mlflow_sanity_check()->bool:#check mlflow connection
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.set_experiment("sanity_check")
        with mlflow.start_run(run_name="sanity_check_run"):
            mlflow.log_param("sanity_check", "passed")
        logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)
        return True
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MLflow tracking server at {MLFLOW_TRACKING_URI}: {e}")
    
def path_sanity_check()->bool:#check if path exists
    if not os.path.exists(PRED_PROCESSED_PATH):
        raise FileNotFoundError(f"Processed data path {PRED_PROCESSED_PATH} does not exist")
    if not os.path.exists(PRED_OUTPUT_PATH):
        raise FileNotFoundError(f"Prediction output path {PRED_OUTPUT_PATH} does not exist")
    logger.info("Data paths verified: %s, %s", PRED_PROCESSED_PATH, PRED_OUTPUT_PATH)
    return True
    

def pred():
    metadata_file = PREDICTION_METADATA_FILE

    try:
        if not mlflow_sanity_check():
            return
        
        if not path_sanity_check():
            return
        
        mysql_utils = MySQLUtils(PRED_MYSQLURI)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("xgb_aqi_hourly_prediction")

        with mlflow.start_run(run_name="hourly_prediction_run"):#set prediction expermiment
            run_id = mlflow.active_run().info.run_id
            model_name = "hourly_pm25_24h_service"
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")#load model from registry
            data = pd.read_parquet(PRED_PROCESSED_PATH + r"/" + datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d") + ".csv")
            logger.info("Data preview:\n%s", data.head())
            
            expected_cols = [col.name for col in model.metadata.get_input_schema().inputs]#get expected columns from model signature
            logger.info("Expected columns for prediction: %s", expected_cols)
            prediction = model.predict(data.iloc[[1]][expected_cols])#only using latest aqi fal for proedictoin

            now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")# get current time
            prediction["timestamp"] = now
            logger.info("Prediction result:\n%s", prediction)

            output_file = PRED_OUTPUT_PATH + r"/" + datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d") + ".csv"
            prediction.to_csv(output_file, index=False)
            logger.info("Prediction saved to %s", output_file)

            if PRED_MYSQLURI:#check if MySQL URI is provided, if yes write to MySQL
                mysql_utils.write_dataframe_to_mysql(prediction, table_name="hourly_predictions", if_exists="append")
                logger.info("Prediction written to MySQL table hourly_predictions")

            update_pipeline_metadata(
                metadata_file,
                {
                    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                    "pipeline": "prediction",
                    "stage": "predict",
                    "status": "success",
                    "mlflow_run_id": run_id,
                    "registered_model": model_name,
                    "output_file": output_file,
                    "prediction_rows": int(len(prediction)),
                    "latest_prediction": float(prediction["prediction"].iloc[0]),
                },
            )
    except Exception as exc:
        update_pipeline_metadata(
            metadata_file,
            {
                "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "pipeline": "prediction",
                "stage": "predict",
                "status": "failed",
                "error": str(exc),
            },
        )
        raise

if __name__ == "__main__":
    pred()