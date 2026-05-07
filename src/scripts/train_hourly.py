"""Training script for hourly AQI prediction using XGBoost, with hyperparameter tuning via Optuna and MLflow tracking."""


import xgboost as xgb
import pandas as pd
import optuna
import mlflow
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import os
from functools import partial
from mlflow.models.signature import infer_signature

try:
    from optuna.integration import MLflowCallback
except ModuleNotFoundError:
    try:
        from optuna_integration.mlflow import MLflowCallback
    except ModuleNotFoundError:
        MLflowCallback = None

try:
    from src.modules.data_hourly_preprocessing import DataCleaner as clean
    from src.modules.logging_utils import setup_logging
except ImportError:
    from modules.data_hourly_preprocessing import DataCleaner as clean
    from modules.logging_utils import setup_logging




logger = setup_logging(__name__)



def clean_and_target(horizon: int= 24,
                     train: pd.DataFrame=None,
                     val: pd.DataFrame = None, 
                     test: pd.DataFrame = None, 
                     target_cols: list = None ):
    
    if target_cols is None:
        target_cols = ["pm25", "o3"]

    # cleaning the data with feature engineering after splitting the data
    cleaner = clean(target_features=target_cols)
    train_df_cleaned = cleaner.run_feature_engineering(train, target_features=target_cols)
    val_df_cleaned = cleaner.run_feature_engineering(val, target_features=target_cols)
    test_df_cleaned = cleaner.run_feature_engineering(test, target_features=target_cols)
    
    # creating targets horizons for specified targets using targets from df
    for targets in range(1,horizon+1):
        for target in target_cols:
            train_df_cleaned[f"{target}_plus_{targets}h"] = (train_df_cleaned.groupby("segment_id")[f"{target}_target"].shift(-targets))
            val_df_cleaned[f"{target}_plus_{targets}h"] = (val_df_cleaned.groupby("segment_id")[f"{target}_target"].shift(-targets))
            test_df_cleaned[f"{target}_plus_{targets}h"] = (test_df_cleaned.groupby("segment_id")[f"{target}_target"].shift(-targets))
            logger.info("Created target column: %s", f"{target}_plus_{targets}h")
    return train_df_cleaned, val_df_cleaned, test_df_cleaned



def split(cleaned:pd.DataFrame,
          train_size:float = 0.7, 
          val_size:float = 0.15, 
          test_size:float = 0.15, 
          df:pd.DataFrame=None):
    """Splitting the data into train, validation, and test sets based on time series order, 
    ensuring no data leakage across segments."""

    #calculating the end of index based on number of samples and ratios
    train_end = int(len(cleaned) * train_size)
    val_end = int(len(cleaned) * (train_size + val_size))

    #splitting the data based on the ratios
    df_train = cleaned.iloc[:train_end]
    df_val = cleaned.iloc[train_end:val_end]
    df_test = cleaned.iloc[val_end:]

    #printing the data ranges and shapes
    logger.info("=================================")
    logger.info("Train start: %s end: %s", df_train.index.min(), df_train.index.max())
    logger.info("Validation start: %s end: %s", df_val.index.min(), df_val.index.max())
    logger.info("Test start: %s end: %s", df_test.index.min(), df_test.index.max())
    logger.info(
        "Test shape: %s Val shape: %s Test shape: %s",
        df_test.shape,
        df_val.shape,
        df_test.shape,
    )
    logger.info("===================================")

    return df_train, df_val, df_test

@staticmethod
def objective(trial, 
            df_train=None, 
            df_val=None,
            features_exclude =[f"pm25_plus_{i}h" for i in range(1,25)] + \
              [f"o3_plus_{i}h" for i in range(1,25)] + ["segment_id",],
            h:int=1, 
            target_col:str="pm25", 
            value_range:tuple=(0,500)):
    #Preparing data for 1 hour ahead prediction
    #Tuning hyperparamaters
    """Objective function for optuna tuning"""

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

    dtrain = xgb.DMatrix(data=X, label=y, enable_categorical=True)
    dval = xgb.DMatrix(data=X_val, label=y_val, enable_categorical=True)

    num_boost_round = params.pop("n_estimators")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "validation")],
        verbose_eval=False,
    )

    preds = booster.predict(dval)
    mae = mean_absolute_error(y_val, preds)

    trial.set_user_attr("best_iteration", getattr(booster, "best_iteration", None))
    return mae
    

def tune(df_train:pd.DataFrame=None, 
         df_val:pd.DataFrame=None,
         trials:int=60,
         optuna_path:str="sqlite:///db/optuna.db",
         callback=None,
         target_col:str="pm25", 
         value_range:tuple=(0,500), 
         horizons:int=24,
         features_exclude=[f"pm25_plus_{i}h" for i in range(1,25)] + \
                        [f"o3_plus_{i}h" for i in range(1,25)] + \
                        ["segment_id","imputation_confidence"])->dict:
    """Hyperparameter tuning using Optuna, with MLflow tracking and support for callbacks (e.g., pruning)."""

    #only tuning 1 or 2 model with 24 horizon  or 1h, using validation set maE as metric

    logger.info("====================Starting Hyperparameter Tuning=================")
    storage = f"{optuna_path}"
    logger.info("Optuna storage set to %s", storage)

    # Allow nested runs so the callback can start a run per trial while an outer run is active
   
    study = optuna.create_study(direction="minimize", storage=storage, study_name=f"xgb_{target_col}_study", 
                                load_if_exists=True)

    # prepare data once (to speed up trials and ensure consistent validation)
   

    obj = partial(objective, df_train=df_train, df_val=df_val,features_exclude=features_exclude,
                 target_col=target_col, value_range=value_range)
    
    logger.info("Starting hyperparameter tuning...")
    callbacks = [callback] if callback is not None else None
    study.optimize(obj, n_trials=trials, callbacks=callbacks)

    # save best params to json
    best_params = study.best_params
    logger.info("Tuning complete")
    logger.info("Best params from tuning: %s", best_params)
    logger.info("Best MAE from tuning: %s", study.best_value)
    

    return best_params

    

def train(horizons:int = 24,
          best_params:dict=None,
         df_val:pd.DataFrame=None,
         df_train:pd.DataFrame=None,
         features_exclude =[f"pm25_plus_{i}h" for i in range(1,25)] + \
              [f"o3_plus_{i}h" for i in range(1,25)] + \
                    ["segment_id","imputation_confidence"],
         target_col:str="pm25", 
         value_range:tuple=(0,500)):
    """Main training function with xgboost, training separate models for each horizon and tracking validation metrics."""

    #training all models for 24 horizons using best params
    logger.info("====================Starting Model Training=================")
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
        y = df_train.loc[train_mask, target_key].reset_index(drop=True).astype(float) #target 
        X = df_train.loc[train_mask].drop(columns=features_exclude).reset_index(drop=True) #features


        y_val = df_val.loc[eval_mask, target_key].reset_index(drop=True).astype(float)
        X_val = df_val.loc[eval_mask].drop(columns=features_exclude).reset_index(drop=True)

        # Skip horizons with insufficient data
        if y.empty or y_val.empty:
            logger.warning(
                "Skipping horizon %s for %s due to insufficient data (train=%s, val=%s)",
                h,
                target_col,
                len(y),
                len(y_val),
            )
            continue

        dtrain = xgb.DMatrix(data=X, label=y, enable_categorical=True)
        dval = xgb.DMatrix(data=X_val, label=y_val, enable_categorical=True)

        training_params = best_params.copy()
        num_boost_round = training_params.pop("n_estimators", 500)

        reg = xgb.train(
            training_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "validation")],
            verbose_eval=False,
        )

        models[target_key] = reg
        val_metrics_mae[target_key] = mean_absolute_error(y_val, reg.predict(dval))
        val_metrics_rmse[target_key] = root_mean_squared_error(y_val, reg.predict(dval))
        sign = infer_signature(X, reg.predict(dtrain))
        logger.info(
            "Trained model for horizon %sh with MAE: %.4f, RMSE: %.4f",
            h,
            val_metrics_mae[target_key],
            val_metrics_rmse[target_key],
        )
            
    return models, val_metrics_mae, val_metrics_rmse, sign
    
def test(horizons:int = 24, 
         models:dict=None,features_exclude
              =[f"pm25_plus_{i}h" for i in range(1,25)] + \
              [f"o3_plus_{i}h" for i in range(1,25)] + \
            ["segment_id","imputation_confidence"], df_test:pd.DataFrame=None, 
         target_col:str="pm25", value_range:tuple=(0,500)):
    """Testing the trained models on the test set, calculating MAE for each horizon, and returning a dictionary of test metrics."""
    

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

        dtest = xgb.DMatrix(data=X_test, label=y_test, enable_categorical=True)
        test_metrics[target_key] = mean_absolute_error(y_test, model.predict(dtest))

    return test_metrics



def main():#entry point for cli
    #set mlfluw tracking using relative path on a local server
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))

    main_df = pd.read_csv(r"D:\pypipeline\data\raw\static\hourly\aqi\limited\us_diplomatic_post_hourly.csv") #load raw data
    target_cols = ["pm25", "o3"]
    main_df_cleaned = clean(target_features=target_cols).run_clean(main_df, target_features=target_cols) #initial cleaning on main data, feature extraction
    df_train, df_val, df_test = split(main_df_cleaned) #splliting after cleaning

    df_train_cleaned, df_val_cleaned, df_test_cleaned = clean_and_target(train=df_train, val=df_val, test=df_test, target_cols=target_cols) #clean and create targets
   

    mlflow.set_experiment("Hourly_AQI_Tuning_Experiment")
    best_params = tune(df_train=df_train_cleaned, df_val=df_val_cleaned, trials=5)

    mlflow.set_experiment("Hourly_AQI_Experiment") #setting differnt experiment for actual training
    models, val_metrics = train(horizons=24, best_params=best_params, df_train=df_train_cleaned, df_val=df_val_cleaned)

    logger.info("Validation Metrics: %s", val_metrics)
    test_metrics = test(horizons=24, models=models, df_test=df_test_cleaned)
    logger.info("Test Metrics: %s", test_metrics)


if __name__ == "__main__":
    main()

    