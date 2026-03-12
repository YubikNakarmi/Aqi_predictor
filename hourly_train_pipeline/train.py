import os
import json
from datetime import datetime, timezone
from scripts.train_hourly import train
import pandas as pd
import mlflow
from modules.logging_utils import setup_logging
from modules.runtime_metadata import update_pipeline_metadata

DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly/")
TRAIN_PROCESSED_FILE = os.getenv("TRAIN_PROCESSED_FILE", f"{DATA_PROCESSED_DIR}/train_processed.parquet")
VAL_PROCESSED_FILE = os.getenv("VAL_PROCESSED_FILE", f"{DATA_PROCESSED_DIR}/val_processed.parquet")

PREDICTIONS_DIR = os.getenv("PREDICTIONS_PATH", r"data/predictions/hourly/us_paro_hourly/test/")
HORIZON = int(os.getenv("HORIZON", 24))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "data/artifacts")
TARGET_COL = os.getenv("TARGET_COL", "pm25")  # Default to pm25, can be "o3" or others
VALUE_MIN = float(os.getenv("VALUE_MIN", 0))
VALUE_MAX = float(os.getenv("VALUE_MAX", 500))
TRAINING_METADATA_FILE = os.getenv("TRAINING_METADATA_FILE", "data/metadata/train.json")

logger = setup_logging(__name__)


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
        
def main():
    metadata_file = TRAINING_METADATA_FILE

    try:
        if mlflow_sanity_check() is False:
            return
        train_df = pd.read_parquet(TRAIN_PROCESSED_FILE)
        val_df = pd.read_parquet(VAL_PROCESSED_FILE)
        logger.info("Test data loaded")

        with open(f"{ARTIFACTS_PATH}/best_params.json", "r", encoding="utf-8") as f:
            best_params = json.load(f)
        logger.info("Best hyperparameters loaded")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("test")

        features_exclude = [f"{TARGET_COL}_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                           [f"o3_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                           [f"pm25_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                           ["segment_id", "imputation_confidence", "pm25_target", "o3_target"]

        models, mae_metrics, rmse_metrics, signature = train(
            df_val=val_df,
            df_train=train_df,
            horizons=HORIZON,
            best_params=best_params,
            target_col=TARGET_COL,
            value_range=(VALUE_MIN, VALUE_MAX),
            features_exclude=features_exclude,
        )

        run_summaries = []
        for h in range(1, HORIZON + 1):
            key = f"{TARGET_COL}_plus_{h}h"
            model = models.get(key)
            if model is None:
                continue

            with mlflow.start_run(run_name=f"final_{TARGET_COL}_training_{h}h"):
                run_id = mlflow.active_run().info.run_id
                mlflow.log_params(best_params)
                mlflow.log_params({"type": "xgboost", "target": TARGET_COL})
                mlflow.set_tag("horizon", f"{h}h")

                val_mae = mae_metrics.get(key)
                val_rmse = rmse_metrics.get(key)

                if val_rmse is not None:
                    mlflow.log_metric(f"val_rmse_{key}", val_rmse)
                    logger.info("Horizon %s: Val RMSE = %s", key, val_rmse)
                if val_mae is not None:
                    mlflow.log_metric(f"val_mae_{key}", val_mae)
                    logger.info("Horizon %s: Val MAE = %s", key, val_mae)

                mlflow.xgboost.log_model(
                    xgb_model=model,
                    registered_model_name=f"{key}_model",
                    signature=signature,
                    name=f"{key}_model",
                    params=best_params,
                    input_example=train_df.drop(columns=features_exclude).iloc[:5],
                )
                logger.info("Registered %s model successfully", key)

                run_summaries.append(
                    {
                        "horizon": h,
                        "target_key": key,
                        "run_id": run_id,
                        "val_mae": val_mae,
                        "val_rmse": val_rmse,
                    }
                )

        avg_mae = sum(mae_metrics.values()) / len(mae_metrics) if mae_metrics else None
        avg_rmse = sum(rmse_metrics.values()) / len(rmse_metrics) if rmse_metrics else None

        update_pipeline_metadata(
            metadata_file,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "pipeline": "training",
                "stage": "train",
                "status": "success",
                "target_col": TARGET_COL,
                "horizon": HORIZON,
                "run_count": len(run_summaries),
                "metrics_summary": {
                    "avg_val_mae": avg_mae,
                    "avg_val_rmse": avg_rmse,
                },
                "runs": run_summaries,
            },
        )

        logger.info("Training complete and models logged to MLflow.")
    except Exception as exc:
        update_pipeline_metadata(
            metadata_file,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "pipeline": "training",
                "stage": "train",
                "status": "failed",
                "target_col": TARGET_COL,
                "horizon": HORIZON,
                "error": str(exc),
            },
        )
        raise
if __name__ == "__main__":
    main()        



