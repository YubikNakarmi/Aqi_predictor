import os
import json
from hourly_pipeline.tune import ARTIFACTS_PATH
from scripts.train_hourly import train
import pandas as pd
import mlflow

DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly/")
PREDICTIONS_DIR = os.getenv("PREDICTIONS_PATH", r"data/predictions/hourly/us_paro_hourly/test/")
HORIZON = int(os.getenv("HORIZON", 24))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "/data/artifacts/")




def main():
    test_df = pd.read_parquet(f"{DATA_PROCESSED_DIR}/test_processed.parquet")
    val_df = pd.read_parquet(f"{DATA_PROCESSED_DIR}/val_processed.parquet")
    print("Test data loaded\n")

    with open(f"{ARTIFACTS_PATH}/best_params.json", "r") as f:
        best_params = json.load(f)
    print("Best hyperparameters loaded\n")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("xgb_aqi_hourly_training")

    model,metrics = train(df_val=val_df, df_test=test_df, horizon=HORIZON, best_params=best_params)

    for h,metric in metrics:
        print(f"Horizon {h} Test MAE: {metric['test_mae']}, RMSE: {metric['test_rmse']}")    




