import pandas as pd
from train import HORIZON, TARGET_COL
from scripts.train_hourly import train
import mlflow
import os 
from mlflow import MlflowClient
from mlflow.models import MetricThreshold


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly/")
TARGET_COL = os.getenv("TARGET_COL", "pm25")  # Default to pm25, can be "o3" or others
VALUE_MIN = float(os.getenv("VALUE_MIN", 0))
VALUE_MAX = float(os.getenv("VALUE_MAX", 500))


def plot(eval_df,builtin_metrics, artifacts_dir):
    print(eval_df["prediction"].head())
   

def main():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    mlflow.set_experiment("xgb_aqi_hourly_evaluation") 
    df_test= pd.read_parquet(f"{DATA_PROCESSED_DIR}/test_processed.parquet")

    client = MlflowClient()
    result={}
    features_exclude = [f"{TARGET_COL}_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       [f"o3_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       [f"pm25_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       ["segment_id", "imputation_confidence"]
    
    predictions={}

    for h in range(1, 2):
        with mlflow.start_run(run_name=f"final_{TARGET_COL}_evaluation_{h}h"):

            model_uri=f"models:/pm25_plus_{h}h_model/latest"
            model = mlflow.xgboost.load_model(model_uri)
            target_key = f"{TARGET_COL}_plus_{h}h"

            test_mask = df_test[target_key].notnull() & (df_test[target_key] >= VALUE_MIN) \
            & (df_test[target_key] <= VALUE_MAX)

            df_test_h = df_test[test_mask].drop(features_exclude, axis=1).copy()


            y_pred = model.predict(df_test_h)
            
            eval_df= pd.DataFrame({
                "prediction": y_pred,
                "target":   df_test.loc[test_mask, target_key]
            })

            mlflow.set_tag("horizon", f"{h}h")
        
            y = df_test.loc[test_mask, target_key]
            df_test_eval = df_test_h.join(y)

            thresold ={
               "mae": MetricThreshold(threshold= 25.0,greater_is_better=False),
               "rmse": MetricThreshold(threshold= 40.0,greater_is_better=False)
           }
            
            result = mlflow.models.evaluate(
                                            predictions="prediction",
                                            targets = "target",
                                        model_type="regressor",
                                        data=eval_df,
                                        evaluators="default",
                                        evaluator_config={"log_explainer": True,
                                                            })
            
            try:
                mlflow.validate_evaluation_results(candidate_result=result,
                                               validation_thresholds=thresold)
                print(f"Model evaluation for horizon {h}h met the specified thresholds.")
                client.set_model_version_tag(name=target_key+"_model",
                                         version=result.model_version,
                                         key="rmse",
                                         value=str(result.metrics['regression_metrics']['rmse']))
                client.set_model_version_tag(name=target_key+"_model",
                                            version=result.model_version,
                                            key="mae",
                                            value=str(result.metrics['regression_metrics']['mae']))
                client.transition_model_version_stage(
                    name=target_key+"_model",
                    version=result.model_version,
                    stage="staging"
            )

            except mlflow.exceptions.MlflowException as e:
                print(f"Model evaluation did not meet the specified thresholds: {e}")
                continue
                

            
            
        mlflow.end_run()

if __name__ == "__main__":
    main()


