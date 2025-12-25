import pandas as pd
import numpy as np
import click



def dedup_and_agg(df, value_cols=None):
    if df['datetime'].duplicated().any():
        # convert to numeric and duplicate aggregation
        if value_cols is None:
            value_cols = df.select_dtypes('number').columns.tolist()
        return df.groupby('datetime', as_index=False)[value_cols].mean()
    return df

def feature_engineering(df:pd.DataFrame)->pd.DataFrame:
    df_fe = df.copy()
    df_fe["date"]





def main():
    df = pd.read_csv("D:\pypipeline\data\raw\static\hourly\aqi\baisipati_hourly.csv")


if main


