import os
import datetime
from datetime import timezone

import mlflow
import pandas as pd
import xgboost as xgb
from mlflow import MlflowClient
from mlflow.models import MetricThreshold

from modules.logging_utils import setup_logging
from modules.runtime_metadata import update_pipeline_metadata
from train import HORIZON, TARGET_COL

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly")
TARGET_COL = os.getenv("TARGET_COL", "pm25")
VALUE_MIN = float(os.getenv("VALUE_MIN", 0))
VALUE_MAX = float(os.getenv("VALUE_MAX", 500))
PREDICTIONS_DIR = os.getenv("PREDICTIONS_DIR", r"data/predictions/hourly/us_paro_hourly")
TRAIN_METADATA_FILE = os.getenv("TRAIN_METADATA_FILE", "data/metadata/train.json")
logger = setup_logging(__name__)


def mlflow_sanity_check() -> bool:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.set_experiment("sanity_check")
        with mlflow.start_run(run_name="sanity_check_run"):
            mlflow.log_param("sanity_check", "passed")
        logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)
        return True
    except Exception as exc:
        raise ConnectionError(
            f"Failed to connect to MLflow tracking server at {MLFLOW_TRACKING_URI}: {exc}"
        )


def main():
    metadata_file = TRAIN_METADATA_FILE
    evaluation_metrics = []
    promoted_models = []

    try:
        if not mlflow_sanity_check():
            return

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("xgb_aqi_hourly_evaluation")

        df_test = pd.read_parquet(f"{DATA_PROCESSED_DIR}/test_processed.parquet")
        
        features_exclude = [f"{TARGET_COL}_plus_{i}h" for i in range(1, HORIZON + 1)] + [
            f"o3_plus_{i}h" for i in range(1, HORIZON + 1)
        ] + [f"pm25_plus_{i}h" for i in range(1, HORIZON + 1)] + [
            "segment_id",
            "imputation_confidence",
            "pm25_target",
            "o3_target",
        ]

        client = MlflowClient()
        horizon_predictions = []

        ''' horizon loop for each model evaluation and promotion '''

        for h in range(1, HORIZON + 1):
            with mlflow.start_run(run_name=f"final_{TARGET_COL}_evaluation_{h}h"):
                model_uri = f"models:/pm25_plus_{h}h_model/latest"#mlflow uri for model
                model = mlflow.xgboost.load_model(model_uri)
                target_key = f"{TARGET_COL}_plus_{h}h" #target pm25

                model_name = f"{target_key}_model"
                latest_versions = client.get_latest_versions(model_name, stages=["None"])#using mlflow client for loading latest trained model
                current_model_version = latest_versions[0].version if latest_versions else None

                test_mask = (
                    df_test[target_key].notnull()
                    & (df_test[target_key] >= VALUE_MIN) #filter test data based on target value range
                    & (df_test[target_key] <= VALUE_MAX)
                )

                if "date" in df_test.columns:
                    date_series = df_test.loc[:, "date"]#if date column exists, use it as date series for evaluation dataframe
                else:
                    date_series = df_test.index.to_series().rename("date")

                filtered_dates = date_series.loc[test_mask]
                X = df_test.loc[test_mask].drop(features_exclude, axis=1).copy()
                y = df_test.loc[test_mask, target_key]

                X_dmatrix = xgb.DMatrix(X)
                y_pred = model.predict(X_dmatrix)
                logger.info("Predictions for horizon %sh completed.", h)

                eval_df = pd.DataFrame({"prediction": y_pred, "target": y})
                mlflow.set_tag("horizon", f"{h}h")

                horizon_predictions.append(
                    pd.DataFrame(
                        {
                            "date": filtered_dates.values,
                            f"{target_key}_pred": y_pred,
                            f"{target_key}_true": y.values,
                        }
                    )
                )

                prediction_path = f"predictions{h}h.csv"
                prediction_df = eval_df.copy()
                prediction_df["date"] = filtered_dates.values
                prediction_df.to_csv(prediction_path, index=False)
                mlflow.log_artifact(prediction_path, artifact_path="prediction")
                os.remove(prediction_path)

                threshold = {
                    "mean_absolute_error": MetricThreshold(threshold=25.0, greater_is_better=False),
                    "root_mean_squared_error": MetricThreshold(
                        threshold=40.0, greater_is_better=False
                    ),
                }

                result = mlflow.models.evaluate(
                    predictions="prediction",
                    targets="target",
                    model_type="regressor",
                    data=eval_df,
                    evaluators="default",
                )

                try:
                    mlflow.validate_evaluation_results(
                        candidate_result=result, validation_thresholds=threshold
                    )
                    logger.info("Model evaluation for horizon %sh met thresholds.", h)

                    if current_model_version:
                        client.set_model_version_tag(
                            name=model_name,
                            version=current_model_version,
                            key="rmse",
                            value=str(result.metrics["root_mean_squared_error"]),
                        )
                        client.set_model_version_tag(
                            name=model_name,
                            version=current_model_version,
                            key="mae",
                            value=str(result.metrics["mean_absolute_error"]),
                        )
                        client.transition_model_version_stage(
                            name=model_name,
                            version=current_model_version,
                            stage="staging",
                        )
                        promoted_models.append(
                            {"model_name": model_name, "version": current_model_version}
                        )
                        logger.info("Model for horizon %sh promoted to staging.", h)
                    else:
                        logger.warning("Could not find model version for %s", model_name)
                except mlflow.exceptions.MlflowException as exc:
                    logger.info("Validation threshold check failed: %s", exc)

                evaluation_metrics.append(
                    {
                        "horizon": h,
                        "target_key": target_key,
                        "mae": result.metrics.get("mean_absolute_error"),
                        "rmse": result.metrics.get("root_mean_squared_error"),
                    }
                )

                if not X.empty:
                    feature_file = f"features_{h}h.txt"
                    with open(feature_file, "w", encoding="utf-8") as file:
                        file.write("\n".join(X.columns.tolist()))
                    mlflow.log_artifact(feature_file, artifact_path="features")
                    os.remove(feature_file)

        if horizon_predictions:
            preds_df = horizon_predictions[0].set_index("date")
            for item in horizon_predictions[1:]:
                preds_df = preds_df.join(item.set_index("date"), how="outer")
            preds_df = preds_df.reset_index()
            preds_df["date"] = pd.to_datetime(preds_df["date"])
            preds_df.sort_values("date", inplace=True)
        else:
            preds_df = pd.DataFrame()

        predictions_out_dir = os.path.join(PREDICTIONS_DIR, "test")
        os.makedirs(predictions_out_dir, exist_ok=True)
        preds_df.to_csv(
            os.path.join(predictions_out_dir, f"hourly_{TARGET_COL}_predictions.csv"),
            index=False,
        )

        avg_mae = (
            sum(item["mae"] for item in evaluation_metrics if item["mae"] is not None)
            / len(evaluation_metrics)
            if evaluation_metrics
            else None
        )
        avg_rmse = (
            sum(item["rmse"] for item in evaluation_metrics if item["rmse"] is not None)
            / len(evaluation_metrics)
            if evaluation_metrics
            else None
        )

        update_pipeline_metadata(
            metadata_file,
            {
                "timestamp_utc": datetime.datetime.now(timezone.utc).isoformat(),
                "pipeline": "training",
                "stage": "eval",
                "status": "success",
                "target_col": TARGET_COL,
                "horizon": HORIZON,
                "metrics_summary": {"avg_mae": avg_mae, "avg_rmse": avg_rmse},
                "metrics_by_horizon": evaluation_metrics,
                "promoted_models": promoted_models,
            },
        )
    except Exception as exc:
        update_pipeline_metadata(
            metadata_file,
            {
                "timestamp_utc": datetime.datetime.now(timezone.utc).isoformat(),
                "pipeline": "training",
                "stage": "eval",
                "status": "failed",
                "target_col": TARGET_COL,
                "horizon": HORIZON,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
