import shap
import matplotlib.pyplot as plt
import numpy as np
import os


DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly/")
VAL_PROCESSED_FILE = os.getenv("VAL_PROCESSED_FILE", f"{DATA_PROCESSED_DIR}/val_processed.parquet")
TRAIN_PROCESSED_FILE = os.getenv("TRAIN_PROCESSED_FILE", f"{DATA_PROCESSED_DIR}/train_processed.parquet")
TARGET_COL = os.getenv("TARGET_COL", "pm25")  # Default to pm25, can be "o3" or others
HORIZON = int(os.getenv("HORIZON", 24))
 
def shap_model():
      
      features_exclude = [f"{TARGET_COL}_plus_{i}h" for i in range(1, HORIZON + 1)] + [
            f"o3_plus_{i}h" for i in range(1, HORIZON + 1)
        ] + [f"pm25_plus_{i}h" for i in range(1, HORIZON + 1)] + [
            "segment_id",
            "imputation_confidence",
            "pm25_target",
            "o3_target",
        ]
      
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

