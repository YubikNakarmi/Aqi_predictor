import mlflow
import os
import pandas as pd
from modules.logging_utils import setup_logging

TARGET_COL = os.getenv("TARGET_COL", "pm25")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
HORIZON = int(os.getenv("HORIZON", 24))
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly/")


logger = setup_logging(__name__)


class ServeModel(mlflow.pyfunc.PythonModel):
    
    def load_context(self,context):
        self.models = {h:mlflow.pyfunc.load_model(f"models:/{TARGET_COL}_plus_{h}h/latest") for h in range(1, 25)}

    def predict(self ,model_input,context):
        preds = {}
        for h, model in self.models.items():
            preds[f"{TARGET_COL}_plus_{h}h_pred"] = model.predict(model_input)

        return pd.DataFrame(preds)
    
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("xgb_aqi_hourly_serving")

    features_exclude = [f"{TARGET_COL}_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       [f"o3_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       [f"pm25_plus_{i}h" for i in range(1, HORIZON + 1)] + \
                       ["segment_id", "imputation_confidence","pm25_target","o3_target"]
    
    dummy_df = pd.read_parquet(f"{DATA_PROCESSED_DIR}/train_processed.parquet")

    with mlflow.start_run(run_name="hourly_aqi_model_serving"):
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ServeModel(),
            registered_model_name=f"hourly_{TARGET_COL}_24h_service",
            input_example=dummy_df.drop(columns=features_exclude).iloc[:5]

        )
        logger.info("Logged serving model to MLflow")

if __name__ == "__main__":
    main()