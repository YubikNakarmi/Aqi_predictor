import pandas as pd
from train import HORIZON, TARGET_COL
import mlflow
import os 
from mlflow import MlflowClient
from mlflow.models import MetricThreshold


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly")
TARGET_COL = os.getenv("TARGET_COL", "pm25")  # Default to pm25, can be "o3" or others
VALUE_MIN = float(os.getenv("VALUE_MIN", 0))
VALUE_MAX = float(os.getenv("VALUE_MAX", 500))
PREDICTIONS_DIR = os.getenv("PREDICTIONS_DIR", r"data/predictions/hourly/us_paro_hourly")




def main():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    mlflow.set_experiment("xgb_aqi_hourly_evaluation") 
    df_test= pd.read_parquet(f"{DATA_PROCESSED_DIR}/test_processed.parquet")

    result={}
    features_exclude = [f"{TARGET_COL}_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       [f"o3_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       [f"pm25_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       ["segment_id", "imputation_confidence"]
    
    client = MlflowClient()
    horizon_predictions = []  # collect predictions per horizon to concatenate later

    for h in range(1,HORIZON+1):
        with mlflow.start_run(run_name=f"final_{TARGET_COL}_evaluation_{h}h"):

            model_uri=f"models:/pm25_plus_{h}h_model/latest" #getting model uri form registry
            model = mlflow.xgboost.load_model(model_uri) #lodaing from uri
            target_key = f"{TARGET_COL}_plus_{h}h"

            # Get the latest model version for tagging/promotion
            model_name = f"{target_key}_model"
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            current_model_version = latest_versions[0].version if latest_versions else None

            ''' preprocessing steps'''
            test_mask = (
                df_test[target_key].notnull()
                & (df_test[target_key] >= VALUE_MIN)
                & (df_test[target_key] <= VALUE_MAX)
            )

            # keep date indexing aligned with the boolean mask
            if "date" in df_test.columns:
                date_series = df_test.loc[:, "date"]
            else:
                date_series = df_test.index.to_series().rename("date")

            filtered_dates = date_series.loc[test_mask]

            X = df_test.loc[test_mask].drop(features_exclude, axis=1).copy()

            y = df_test.loc[test_mask, target_key]
            y_pred = model.predict(X) #predict
            print(f"Predictions for horizon {h}h completed.")
            
            
            eval_df= pd.DataFrame({
                "prediction": y_pred,
                "target":y
            })

            mlflow.set_tag("horizon", f"{h}h")
          
             
            ''' creading df for pre dictions and true values and log to mlflow'''

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

            mlflow.log_artifact(prediction_path,artifact_path="prediction")
            os.remove(prediction_path) #removing after logging

            
            ''' mlflow evaluation'''

            thresold ={
               "mean_absolute_error": MetricThreshold(threshold= 25.0,greater_is_better=False),
               "root_mean_squared_error": MetricThreshold(threshold= 40.0,greater_is_better=False)
           } #setting thresholds
            
            result = mlflow.models.evaluate( #evaluation
                                            predictions="prediction",
                                            targets = "target",
                                        model_type="regressor",
                                        data=eval_df,
                                        evaluators="default",
                                        )
            # checking for thersholds and promotion

            try:
                mlflow.validate_evaluation_results(candidate_result=result,
                                               validation_thresholds=thresold)
                print(f"Model evaluation for horizon {h}h met the specified thresholds.")

                if current_model_version:
                    client.set_model_version_tag(name=model_name,
                                             version=current_model_version,
                                             key="rmse",
                                             value=str(result.metrics['root_mean_squared_error']))
                    client.set_model_version_tag(name=model_name,
                                                version=current_model_version,
                                                key="mae",
                                                value=str(result.metrics['mean_absolute_error']))
                    client.transition_model_version_stage(
                        name=model_name,
                        version=current_model_version,
                        stage="staging"
                    )
                    print(f"Model for horizon {h}h promoted to 'staging' stage.")
                else:
                    print(f"Warning: Could not find model version for {model_name}")

            except mlflow.exceptions.MlflowException as e:
                print(f"Model evaluation did not meet the specified thresholds: {e}")
                continue

            print(result.metrics["mean_absolute_error"])
        mlflow.end_run()

    if horizon_predictions:
        # align by date using join to preserve all horizons even if masks differ
        preds_df = horizon_predictions[0].set_index("date")
        for df in horizon_predictions[1:]:
            preds_df = preds_df.join(df.set_index("date"), how="outer")
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

if __name__ == "__main__":
    main()


