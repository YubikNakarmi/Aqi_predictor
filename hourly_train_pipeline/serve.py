import mlflow
import os
import pandas as pd

TARGET_COL = os.getenv("TARGET_COL", "pm25")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


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

    with mlflow.start_run(run_name="hourly_aqi_model_serving"):
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ServeModel(),
            registered_model_name=f"hourly_{TARGET_COL}_24h_service"
        )

if __name__ == "__main__":
    main()