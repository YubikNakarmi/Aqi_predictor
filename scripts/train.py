import xgboost as xgb
import pandas as pd
import optuna
import mlflow
from optuna.integration import MLflowCallback
from sklearn.metrics import mean_absolute_error
import os


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1
    }
     
    xgb_model = xgb.XGBRegressor(**params)
    

     

def tune():
    #only tuning 1 or 2 model with 24 horizon  or 1h, using validation set maE as metric
    mlflow.set_experiment("Hourly_AQI_Tuning_Experiment")

    opt_tracker = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), 
        metric_name="mae",
        
    )

    storage = f"sqlite:///{os.path.abspath 
                                         (os.path.join(os.path.dirname(__file__), '../db/optuna.db'))}"

    study = optuna.create_study(direction="minimize",storage=storage, study_name="xgb_aqi_study", 
                                load_if_exists=True)
    
    study.optimize(objective, n_trials=60,callbacks=[opt_tracker],)

    mlflow.set_experiment("Hourly_AQI_Tuning_Experiment")




def train():
    pass # train the best model on the whole dataset

def test():
    pass


def main():
    #set mlfluw tracking using relative path
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath 
                                         (os.path.join(os.path.dirname(__file__), '../db/mlflow.db'))}") 
    
    tune()
    mlflow.set_experiment("Hourly_AQI_Experiment")

    