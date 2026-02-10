import pandas as pd
import xgboost as xgb
import mlflow
import os
from scripts.train_hourly import tune
import json
from optuna.integration.mlflow import MLflowCallback
from modules.logging_utils import setup_logging

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
VAL_PREDICTIONS_PATH = os.getenv("VAL_PREDICTIONS_PATH", "val_predictions.csv")
DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly/")
OPTUNA_PATH = os.getenv("OPTUNA_PATH", r"sqlite:///db/optuna.db")
TRIALS = int(os.getenv("TRIALS", 60))
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "data/artifacts")
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"

logger = setup_logging(__name__)


def mlflow_sanity_check():

    try:
        mlflow.set_experiment("sanity_check")
        with mlflow.start_run(run_name="sanity_check_run"):
            mlflow.log_param("sanity_check", "passed")
        logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)
        return True
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MLflow tracking server at {MLFLOW_TRACKING_URI}: {e}")
    
def main():

     # optuna callback for mlflow
    opt_tracker = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), 
        metric_name="mae", #auto logs runs
    )
    
    mlflow_sanity_check()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    val_df = pd.read_parquet(f"{DATA_PROCESSED_PATH}/val_processed.parquet")
    train_df = pd.read_parquet(f"{DATA_PROCESSED_PATH}/train_processed.parquet")
    logger.info("Data loaded for tuning")

    mlflow.set_experiment("xgb_aqi_hourly_tuning")
    best_params = tune(df_train=train_df, df_val=val_df,trials=TRIALS, 
                       optuna_path=OPTUNA_PATH, callback=opt_tracker)
    logger.info("Best hyperparameters found: %s", best_params)

    with open(f"{ARTIFACTS_PATH}/best_params.json", "w") as f:
        if not os.path.exists(ARTIFACTS_PATH):
            os.makedirs(ARTIFACTS_PATH)
        json.dump(best_params, f)
    logger.info("Tuning complete")

if __name__ == "__main__":
    main()
