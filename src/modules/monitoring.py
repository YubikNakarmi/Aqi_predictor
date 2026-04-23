import nannyml
import pandas as pd
import os


class Monitoring:
    def __init__(self,reference_df:pd.DataFrame,analysis_df:pd.DataFrame,
              cols,horizons:int=24,period:str="2w",metrics=["rmse","mae"]):
        self.reference_df = reference_df
        self.analysis_df = analysis_df
        self.cols = cols
        self.horizons = horizons
        self.period = period
        self.metrics = metrics

  
    def estimator(self,thresholds:dict)->dict:
        estimators= {}
        for h in range(1,self.horizons+1):
            estimiate = nannyml.DLE(y_true=f"pm25_plus_{h}h_true", 
                                y_pred=f"pm25_plus_{h}h_pred",
                                feature_column_names=self.cols, 
                                timestamp_column_name="date", 
                                chunk_period=self.period,
                                metrics=self.metrics)

            estimiate.fit(self.reference_df)
            estimiate.estimate(self.analysis_df)
            estimators[h] = estimiate
        return estimators
    
    def performance(self):
        pass

  

class plots(Monitoring):
    def __init__(self,reference_df:pd.DataFrame,analysis_df:pd.DataFrame,
              cols,horizons:int=24,period:str="2w"):
        super().__init__(reference_df, analysis_df, cols, horizons, period)

    def plots(self, estimators:dict):
        for h,est in estimators.items():
            est.plot().show() 