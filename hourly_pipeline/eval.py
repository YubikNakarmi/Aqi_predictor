from scripts.train_hourly import test 
import mlflow
import mlflow.client as mlflow_client
import os 


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")


def main():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    mlflow.set_experiment("xgb_aqi_hourly_evaluation")




