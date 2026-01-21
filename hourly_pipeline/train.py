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
TARGET_COL = os.getenv("TARGET_COL", "pm25")  # Default to pm25, can be "o3" or others
VALUE_MIN = float(os.getenv("VALUE_MIN", 0))
VALUE_MAX = float(os.getenv("VALUE_MAX", 500))



def main():
    train_df = pd.read_parquet(f"{DATA_PROCESSED_DIR}/train_processed.parquet")
    val_df = pd.read_parquet(f"{DATA_PROCESSED_DIR}/val_processed.parquet")
    print("Test data loaded\n")

    with open(f"{ARTIFACTS_PATH}/best_params.json", "r") as f:
        best_params = json.load(f)
    print("Best hyperparameters loaded\n") #getting best params

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) #set mlflow tracking uri to server
    mlflow.set_experiment(f"xgb_{TARGET_COL}_hourly_training")

    models, metrics, signature = train(df_val=val_df, df_train=train_df, horizons=HORIZON, 
                                       best_params=best_params, target_col=TARGET_COL,
                                       value_range=(VALUE_MIN, VALUE_MAX))

    features_exclude = [f"{TARGET_COL}_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       [f"o3_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       [f"pm25_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       ["segment_id", "imputation_confidence"]

    with mlflow.start_run(run_name=f"final_{TARGET_COL}_training"):

        mlflow.log_params(best_params)
        mlflow.log_params({"type": "xgboost", "target": TARGET_COL})#log model type and target

        for h in range(1, HORIZON + 1):

            key = f"{TARGET_COL}_plus_{h}h"
            model = models.get(key)
            if model is None:
                continue

            val_mae = metrics.get(key)
            if val_mae is not None:
                mlflow.log_metric(f"val_mae_{key}", val_mae) #log metrics of each model
                
            val_mask = val_df[key].notnull() & (val_df[key] >= VALUE_MIN) & (val_df[key] <= VALUE_MAX)
            X_val = val_df.loc[val_mask].drop(columns=features_exclude).reset_index(drop=True)
            if X_val.empty:
                continue
                
            sign = infer_signature(X_val, model.predict(X_val))# for model consistency and format
            mlflow.xgboost.log_model(xgb_model=model, 
                                     registered_model_name=f"{key}_model",
                                     signature=sign,
                                     artifact_path="model")

    for key, mae in metrics.items():
        print(f"Horizon {key}: Val MAE = {mae}")


if __name__ == "__main__":
    main()        



