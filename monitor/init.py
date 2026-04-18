import nannyml
import os
import pandas as pd

DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly")


def estimator(reference_df, analysis_df):
    
    reference_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "val_processed.parquet"))

    estimate = nannyml.DLE()


def main():
