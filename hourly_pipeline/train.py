import os
import json
from hourly_pipeline.tune import ARTIFACTS_PATH
from scripts.train_hourly import train
import pandas as pd
import mlflow
from mlflow.models import infer_signature

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

    model,metrics,signature = train(df_val=val_df, df_test=test_df, horizon=HORIZON, best_params=best_params)

    with mlflow.start_run(run_name="final_model_training"):
        for h in range(1, HORIZON + 1):

            model = f"pm25_plus_{h}h_model"
            mlflow.log_metric(f"val_mae_pm25_plus_{h}h", metrics[f"pm25_plus_{h}h"])
            mlflow.log_params(best_params)
            mlflow.log_params({"type": "xgboost"})#log model type

            sign = signature# for model consistency and format
            mlflow.xgboost.log_model(xgb_model= model, registered_model_name 
                                    = f"pm25_plus_{h}h_model",signature=sign,artifact_path="model")
            mlflow.end_run()


    for h,metric in metrics:
        print(f"Horizon {h} Test MAE: {metric['test_mae']}, RMSE: {metric['test_rmse']}")
        
        



