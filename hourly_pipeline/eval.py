import pandas as pd
from hourly_pipeline.train import HORIZON, TARGET_COL
from scripts.train_hourly import test 
import mlflow
import os 
from mlflow import MlflowClient


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly/")
TARGET_COL = os.getenv("TARGET_COL", "pm25")  # Default to pm25, can be "o3" or others
VALUE_MIN = float(os.getenv("VALUE_MIN", 0))
VALUE_MAX = float(os.getenv("VALUE_MAX", 500))


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
    

    with mlflow.start_run(run_name="hourly_model_evaluation"):

        for h in range(1, HORIZON + 1):
            model_uri=f"models:/pm25_plus_{h}h_model/latest"
            target_key = f"{TARGET_COL}_plus_{h}h"
        
            train_mask = df_test[target_key].notnull() & (df_test[target_key] >= VALUE_MIN) \
            & (df_test[target_key] <= VALUE_MAX)
            

            df_test_h = df_test[train_mask].copy()


            result = mlflow.models.evaluate(model=model_uri,
                                            targets = target_key,
                                           model_type="regressor",
                                           data=df_test_h,evaluators="default")
            
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

if __name__ == "__main__":
    main()


