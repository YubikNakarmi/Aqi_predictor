from scripts.train_hourly import test 
import mlflow
import os 
from mlflow import MlflowClient


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")


def main():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    mlflow.set_experiment("xgb_aqi_hourly_evaluation")
    client = MlflowClient()

    latest_run = client.search_runs(experiment_ids=["xgb_aqi_hourly_training"],)
    for i in range(1,25):
        model_uri = f"models:/pm25_plus_{i}h_model/Production"
        result = mlflow.models.evaluate(model=model_uri,)



