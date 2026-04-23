import nannyml
import os
import pandas as pd

DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly")

#estimation of possible performance (not actual performance) using the reference data and the analysis data
def estimator(reference_df:pd.DataFrame,analysis_df:pd.DataFrame,
              cols,horizons:int=24,
              period:str="2w")->dict:

    estimators= {}
    for h in range(1,horizons+1):
        estimiate = nannyml.DLE(y_true=f"pm25_plus_{h}h_true", 
                            y_pred=f"pm25_plus_{h}h_pred",
                            feature_column_names=cols, 
                            timestamp_column_name="date", 
                            chunk_period=period,
                            metrics=["rmse","mae"])
        estimiate.fit(reference_df)
        estimiate.estimate(analysis_df)
        estimators[h] = estimiate
    return estimators

#actual performance


def plots(estimators:dict, h):
    for h,est in estimators.items():
        est.plot().show()



def main():
    reference_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "test_processed.parquet"))
