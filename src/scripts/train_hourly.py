from sklearn.calibration import signature
import xgboost as xgb
import pandas as pd
import optuna
import mlflow
from optuna.integration import MLflowCallback
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from modules.data_hourly_preprocessing import DataCleaner as clean
import os
import numpy as np
from functools import partial
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient


def clean_and_target(horizon: int= 24,train: pd.DataFrame=None,val: pd.DataFrame = None, 
                     test: pd.DataFrame = None, target_cols: list = None ):
    # cleaning the data with feature engineering after splitting the data
    train_df_cleaned = clean().run_from_cleaned(train)
    val_df_cleaned = clean().run_from_cleaned(val)
    test_df_cleaned = clean().run_from_cleaned(test)

    # Default to pm25 and o3 if not specified
    if target_cols is None:
        target_cols = ["pm25", "o3"]
    
    # creating targets horizons for specified targets using targets from df
    for targets in range(1,horizon+1):
        for target in target_cols:
            train_df_cleaned[f"{target}_plus_{targets}h"] = (train_df_cleaned.groupby("segment_id")[f"{target}_target"].shift(-targets))
            val_df_cleaned[f"{target}_plus_{targets}h"] = (val_df_cleaned.groupby("segment_id")[f"{target}_target"].shift(-targets))
            test_df_cleaned[f"{target}_plus_{targets}h"] = (test_df_cleaned.groupby("segment_id")[f"{target}_target"].shift(-targets))
    
    return train_df_cleaned, val_df_cleaned, test_df_cleaned



def split(cleaned:pd.DataFrame,train_size:float = 0.7, val_size:float = 0.15, test_size:float = 0.15, df:pd.DataFrame=None):

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
                    ["segment_id",],h:int=1, target_col:str="pm25", value_range:tuple=(0,500)):
    #Preparing data for 1 hour ahead prediction
    #Tuning hyperparamaters

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

    target_key = f"{target_col}_plus_{h}h"
    min_val, max_val = value_range
    
    #using mask to filter null data and extremes
    train_mask = df_train[target_key].notnull() & (df_train[target_key] >= min_val) \
    & (df_train[target_key] <= max_val)

    eval_mask = df_val[target_key].notnull() & (df_val[target_key] >= min_val) \
    & (df_val[target_key] <= max_val)

    # setting targets and features, by using train mask and dropping index
    y = df_train.loc[train_mask, target_key].reset_index(drop=True).astype(float)
    X = df_train.loc[train_mask].drop(columns=features_exclude).reset_index(drop=True)

    #same with val set
    y_val = df_val.loc[eval_mask, target_key].reset_index(drop=True).astype(float)
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
    

def tune(df_train:pd.DataFrame=None, df_val:pd.DataFrame=None,trials:int=60,
         optuna_path:str="sqlite:///db/optuna.db",callback:MLflowCallback=None,
         target_col:str="pm25", value_range:tuple=(0,500), horizons:int=24):
    #only tuning 1 or 2 model with 24 horizon  or 1h, using validation set maE as metric

    features_exclude =[f"{target_col}_plus_{i}h" for i in range(1,horizons+1)] + \
                     [f"o3_plus_{i}h" for i in range(1,horizons+1)] + \
                     [f"pm25_plus_{i}h" for i in range(1,horizons+1)] + \
                     ["segment_id",]


    ''' need to configure for airflow container path '''
    storage = f"{optuna_path}"

    # Allow nested runs so the callback can start a run per trial while an outer run is active
   
    study = optuna.create_study(direction="minimize", storage=storage, study_name=f"xgb_{target_col}_study", 
                                load_if_exists=True)

    # prepare data once (to speed up trials and ensure consistent validation)
   

    obj = partial(objective, df_train=df_train, df_val=df_val,features_exclude=features_exclude,
                 target_col=target_col, value_range=value_range)

    study.optimize(obj, n_trials=trials, callbacks=[callback])

    # save best params to json
    best_params = study.best_params

    return best_params

    

def train(horizons:int = 24,best_params:dict=None,
         df_val:pd.DataFrame=None,df_train:pd.DataFrame=None,features_exclude
              =[f"pm25_plus_{i}h" for i in range(1,25)] + \
              [f"o3_plus_{i}h" for i in range(1,25)] + \
                    ["segment_id","imputation_confidence"],
         target_col:str="pm25", value_range:tuple=(0,500)):
    #training all models for 24 horizons using best params
 
    if best_params is None:
        best_params = tune(df_train=df_train, df_val=df_val, target_col=target_col, 
                          value_range=value_range, horizons=horizons)

    val_metrics_mae = {}
    val_metrics_rmse = {}
    models = {}

    
    if "imputation_confidence" in df_train.columns and df_train["imputation_confidence"].dtype == object:
        df_train["imputation_confidence"] = df_train["imputation_confidence"].astype("category")

    if "imputation_confidence" in df_val.columns and df_val["imputation_confidence"].dtype == object:
        df_val["imputation_confidence"] = df_val["imputation_confidence"].astype("category")

    min_val, max_val = value_range

    for h in range(1, horizons+1):
        target_key = f"{target_col}_plus_{h}h"
        
        train_mask = df_train[target_key].notnull() & (df_train[target_key] >= min_val) \
        & (df_train[target_key] <= max_val)

        eval_mask = df_val[target_key].notnull() & (df_val[target_key] >= min_val) \
        & (df_val[target_key] <= max_val)
        #using only valid data points
        y = df_train.loc[train_mask, target_key].reset_index(drop=True).astype(float)
        X = df_train.loc[train_mask].drop(columns=features_exclude).reset_index(drop=True)


        y_val = df_val.loc[eval_mask, target_key].reset_index(drop=True).astype(float)
        X_val = df_val.loc[eval_mask].drop(columns=features_exclude).reset_index(drop=True)

        # Skip horizons with insufficient data
        if y.empty or y_val.empty:
            print(f"Skipping horizon {h} for {target_col} due to insufficient data (train={len(y)}, val={len(y_val)})")
            continue

        reg = xgb.XGBRegressor(**best_params)
        reg.fit(
            X,
            y,
            eval_set=[(X_val, y_val)],
                    verbose=False,
        )

        models[target_key] = reg
        val_metrics_mae[target_key] = mean_absolute_error(y_val, reg.predict(X_val))
        val_metrics_rmse[target_key] = root_mean_squared_error(y_val, reg.predict(X_val), squared=False)
        sign = infer_signature(X, reg.predict(X))

            
    return models, val_metrics_mae, val_metrics_rmse, sign
    
def test(horizons:int = 24, models:dict=None,features_exclude
              =[f"pm25_plus_{i}h" for i in range(1,25)] + \
              [f"o3_plus_{i}h" for i in range(1,25)] + \
                    ["segment_id","imputation_confidence"], df_test:pd.DataFrame=None, 
         target_col:str="pm25", value_range:tuple=(0,500)):
    test_metrics = {}
    min_val, max_val = value_range

    for h in range(1, horizons+1):
        target_key = f"{target_col}_plus_{h}h"
        model = models.get(target_key)
        if model is None:
            continue
            
        test_mask = (df_test[target_key] >= min_val) & (df_test[target_key] <= max_val)

        y_test = df_test.loc[test_mask, target_key].reset_index(drop=True).astype(float)
        X_test = df_test.loc[test_mask].drop(columns=features_exclude).reset_index(drop=True)
        
        if y_test.empty:
            continue
            
        test_metrics[target_key] = mean_absolute_error(y_test, model.predict(X_test))

    return test_metrics


def main():#entry point for cli
    #set mlfluw tracking using relative path on a local server
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))

    main_df = pd.read_csv(r"D:\pypipeline\data\raw\static\hourly\aqi\limited\us_diplomatic_post_hourly.csv") #load raw data
    main_df_cleaned = clean().run_clean(main_df) #initial cleaning on main data, feature extraction
    df_train, df_val, df_test = split(main_df_cleaned) #splliting after cleaning

    df_train_cleaned, df_val_cleaned, df_test_cleaned = clean_and_target(train=df_train, val=df_val, test=df_test) #clean and create targets
   

    mlflow.set_experiment("Hourly_AQI_Tuning_Experiment")
    best_params = tune(df_train=df_train_cleaned, df_val=df_val_cleaned, trials=5)

    mlflow.set_experiment("Hourly_AQI_Experiment") #setting differnt experiment for actual training
    models, val_metrics = train(horizons=24, best_params=best_params, df_train=df_train_cleaned, df_val=df_val_cleaned)

    print("Validation Metrics:", val_metrics)
    test_metrics = test(horizons=24, models=models, df_test=df_test_cleaned)
    print("Test Metrics:", test_metrics)


if __name__ == "__main__":
    main()

    