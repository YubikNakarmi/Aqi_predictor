from flask import json
import xgboost as xgb
import pandas as pd
import optuna
import mlflow
from optuna.integration import MLflowCallback
from sklearn.metrics import mean_absolute_error
from modules.data_hourly_preprocessing import DataCleaner as clean
import os
import numpy as np
from functools import partial

def clean_and_target(horizon: int= 24,train: pd.DataFrame=None,val: pd.DataFrame = None, 
                     test: pd.DataFrame = None ):
    # cleaning the data with feature engineering after splitting the data
    train_df_cleaned = clean().run_from_cleaned(train)
    val_df_cleaned = clean().run_from_cleaned(val)
    test_df_cleaned = clean().run_from_cleaned(test)

    # creating targets horizons for pm25 and o3 using targets from df
    for targets in range(1,horizon+1):
        train_df_cleaned[f"pm25_plus_{targets}h"] = (train_df_cleaned.groupby("segment_id")["pm25_target"].shift(-targets))
        train_df_cleaned[f"o3_plus_{targets}h"] = train_df_cleaned.groupby("segment_id")["o3_target"].shift(-targets)
        val_df_cleaned[f"pm25_plus_{targets}h"] = (val_df_cleaned.groupby("segment_id")["pm25_target"].shift(-targets))
        val_df_cleaned[f"o3_plus_{targets}h"] = val_df_cleaned.groupby("segment_id")["o3_target"].shift(-targets)
        test_df_cleaned[f"pm25_plus_{targets}h"] = (test_df_cleaned.groupby("segment_id")["pm25_target"].shift(-targets))
        test_df_cleaned[f"o3_plus_{targets}h"] = test_df_cleaned.groupby("segment_id")["o3_target"].shift(-targets)
    
    return train_df_cleaned, val_df_cleaned, test_df_cleaned



def split(train_size:float = 0.7, val_size:float = 0.15, test_size:float = 0.15, df:pd.DataFrame=None):

    main_data = clean()
    cleaned = main_data.run_clean(df)

    #calculating the end of index based on number of samples and ratios
    train_end = int(len(cleaned) * train_size)
    val_end = int(len(cleaned) * (train_size + val_size))

    #splitting the data based on the ratios
    df_train = cleaned.iloc[:train_end]
    df_val = cleaned.iloc[train_end:val_end]
    df_test = cleaned.iloc[val_end:]

    #printing the data ranges and shapes
    print("=================================")
    print("Train start:", df_train.index.min(), "end:", df_train.index.max())
    print("Validation start:", df_val.index.min(), "end:", df_val.index.max())
    print("Test start:", df_test.index.min(), "end:", df_test.index.max())
    print("Test shape:", df_test.shape, " Val shape", df_val.shape, " Test shape:", df_test.shape)
    print("===================================\n")

    return df_train, df_val, df_test

def objective(trial, df_train=None, df_val=None,features_exclude
              =[f"pm25_plus_{i}h" for i in range(1,25)] + \
              [f"o3_plus_{i}h" for i in range(1,25)] + \
                    ["segment_id",]):
    #Preparing data for 1 hour ahead prediction
    #Tuning hyperparamaters

    h=1 #tuning for 1 hour horizon

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "objective": "reg:absoluteerror",
        "random_state": 42,
        "n_jobs": -1,
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),  # L1
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),  # L2
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "eval_metric": ["mae", "rmse"],
        "tree_method": "hist",
        "enable_categorical": True,
    }

    # setting categorical to category dtype
    if "imputation_confidence" in df_train.columns and df_train["imputation_confidence"].dtype == object:
        df_train["imputation_confidence"] = df_train["imputation_confidence"].astype("category")

    if "imputation_confidence" in df_val.columns and df_val["imputation_confidence"].dtype == object:
        df_val["imputation_confidence"] = df_val["imputation_confidence"].astype("category")

    #using mask to filter null data and extremes
    train_mask = df_train[f"pm25_plus_{h}h"].notnull() & (df_train[f"pm25_plus_{h}h"] >=0) \
    & (df_train[f"pm25_plus_{h}h"] <=500)

    eval_mask = df_val[f"pm25_plus_{h}h"].notnull() & (df_val[f"pm25_plus_{h}h"] >=0) \
    & (df_val[f"pm25_plus_{h}h"] <=500)

    # setting targets and features, by using train mask and dropping index
    y = df_train.loc[train_mask, f"pm25_plus_{h}h"].reset_index(drop=True).astype(float)
    X = df_train.loc[train_mask].drop(columns=features_exclude).reset_index(drop=True)

    #same with val set
    y_val = df_val.loc[eval_mask, f"pm25_plus_{h}h"].reset_index(drop=True).astype(float)
    X_val = df_val.loc[eval_mask].drop(columns=features_exclude).reset_index(drop=True)

   #fitting model using regression and eveluation
    model = xgb.XGBRegressor(**params)
    model.fit(
        X,
        y,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)

    trial.set_user_attr("best_iteration", getattr(model, "best_iteration", None))
    return mae
    

def tune(main_df:pd.DataFrame=None):
    #only tuning 1 or 2 model with 24 horizon  or 1h, using validation set maE as metric
    mlflow.set_experiment("Hourly_AQI_Tuning_Experiment")

    features_exclude =[f"pm25_plus_{i}h" for i in range(1,25)] + [f"o3_plus_{i}h" for i in range(1,25)] + \
                    ["segment_id",]

    # optuna callback for mlflow
    opt_tracker = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), 
        metric_name="mae",
        
    )

    ''' need to configure for airflow container path '''
    storage = f"sqlite:///{os.path.abspath 
                                         (os.path.join(os.path.dirname(__file__), '../db/optuna.db'))}"

    study = optuna.create_study(direction="minimize", storage=storage, study_name="xgb_aqi_study", 
                                load_if_exists=True)

    # prepare data once (to speed up trials and ensure consistent validation)
    df_train, df_val, _ = split()
    obj = partial(objective, df_train=df_train, df_val=df_val,features_exclude=features_exclude)

    study.optimize(obj, n_trials=60, callbacks=[opt_tracker])

    # save best params to json
    best_params = study.best_params
    return best_params

def run_all():
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    

def train():
    best_params = tune()
def test():
    pass


def main():
    #set mlfluw tracking using relative path
    mlflow.set_tracking_uri(f"http://localhost:5000")
    
    tune()
    mlflow.set_experiment("Hourly_AQI_Experiment")

if __name__ == "__main__":
    main()

    