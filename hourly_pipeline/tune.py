import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from scripts.train_hourly import tune, train
import json
import click
from optuna.integration.mlflow import MLflowCallback

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://0.0.0.0:5000")
VAL_PREDICTIONS_PATH = os.getenv("VAL_PREDICTIONS_PATH", "val_predictions.csv")
DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly/")
OPTUNA_PATH = os.getenv("OPTUNA_PATH", r"sqlite:///db/optuna.db")
TRIALS = int(os.getenv("TRIALS", 5))
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "data/artifacts")




def mlflow_sanity_check():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name="sanity_check"):
        mlflow.log_param("sanity_check_param", 42)
        mlflow.log_metric("sanity_check_metric", 3.14)


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
    print("Data loaded for tuning\n")

    mlflow.set_experiment("xgb_aqi_hourly_tuning")
    best_params = tune(df_train=train_df, df_val=val_df,trials=TRIALS, 
                       optuna_path=OPTUNA_PATH, callback=opt_tracker)
    print("Best hyperparameters found: ", best_params)

    with open(f"{ARTIFACTS_PATH}/best_params.json", "w") as f:
        if not os.path.exists(ARTIFACTS_PATH):
            os.makedirs(ARTIFACTS_PATH)
        json.dump(best_params, f)
    print("tuning complete")

if __name__ == "__main__":
    main()
