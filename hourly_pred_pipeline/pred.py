import mlflow
from modules import logging_utils
import os
import pandas as pd
from modules.mysql_utils import MySQLUtils
import datetime


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PRED_PROCESSED_PATH = os.environ.get("PRED_PROCESSED_PATH", "data/processed/pred_ingestion/us_paro")
PRED_OUTPUT_PATH = os.environ.get("PRED_OUTPUT_PATH", "data/predictions/hourly/us_paro_hourly/prod")
PRED_MYSQLURI = os.environ.get("PRED_MYSQLURI", "mysql+pymysql://root:yubik123@localhost:3306/pypipeline_predictions")

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
    
def path_sanity_check()->bool:
    if not os.path.exists(PRED_PROCESSED_PATH):
        raise FileNotFoundError(f"Processed data path {PRED_PROCESSED_PATH} does not exist")
    if not os.path.exists(PRED_OUTPUT_PATH):
        raise FileNotFoundError(f"Prediction output path {PRED_OUTPUT_PATH} does not exist")
    logger.info("Data paths verified: %s, %s", PRED_PROCESSED_PATH, PRED_OUTPUT_PATH)
    return True
    

def pred():

    if not mlflow_sanity_check():
        return
    
    if not path_sanity_check():
        return
    
    mysql_utils = MySQLUtils(PRED_MYSQLURI)


    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("xgb_aqi_hourly_prediction")

    with mlflow.start_run(run_name="hourly_prediction_run"):
        model = mlflow.pyfunc.load_model("models:/hourly_pm25_24h_service/latest")
        data = pd.read_parquet(PRED_PROCESSED_PATH+r"/"+datetime.datetime.now().strftime("%Y-%m-%d")+".csv")
        logger.info("Data preview:\n%s", data.head())
        
        expected_cols = [col.name for col in model.metadata.get_input_schema().inputs] 
        logger.info("Expected columns for prediction: %s", expected_cols)
        prediction = model.predict(data.iloc[[1]][expected_cols])#filter using expected columns 

        pred_df = pd.DataFrame(prediction)
        logger.info("Prediction result:\n%s", pred_df)
        pred_df.to_csv(PRED_OUTPUT_PATH+r"/"+datetime.datetime.now().strftime("%Y-%m-%d")+".csv", index=False)
        logger.info("Prediction saved to %s/%s.csv", PRED_OUTPUT_PATH, datetime.datetime.now().strftime("%Y-%m-%d"))

        if PRED_MYSQLURI:
           mysql_utils.write_dataframe_to_mysql(pred_df, table_name="hourly_predictions", if_exists="append")
           logger.info("Prediction written to MySQL table hourly_predictions")

if __name__ == "__main__":
    pred()