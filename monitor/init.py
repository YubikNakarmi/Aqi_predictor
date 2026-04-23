import nannyml
import os
import pandas as pd

DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly")


def estimator(reference_df, analysis_df,cols):

    estmiate = nannyml.DLE(y_true="pm25_plus_1h_true", 
                        y_pred="pm25_plus_1h_pred",
                        feature_column_names=cols, 
                        timestamp_column_name="date", 
                        chunk_period="2w",
                        metrics=["rmse","mae"])
    
    reference_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "val_processed.parquet"))

    estimate = nannyml.DLE() 


def main():
