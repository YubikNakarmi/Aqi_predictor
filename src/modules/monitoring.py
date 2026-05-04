import nannyml
import pandas as pd
import os


class Monitoring:
    def __init__(self, reference_df: pd.DataFrame, analysis_df: pd.DataFrame,
                 cols, y_true_cols: list[str], y_pred_cols: list[str], period: str = "2w", metrics=["rmse", "mae"]):

        self.reference_df = reference_df
        self.analysis_df = analysis_df
        self.cols = cols
        self.y_true_cols = y_true_cols
        self.y_pred_cols = y_pred_cols
        self.period = period
        self.metrics = metrics


    @staticmethod
    def CheckfeatureDimensions(l1,l2)->bool:
        if len(l1) != len(l2):
            raise ValueError("The number of true and predicted columns must be the same.")
    
    def estimator(self, thresholds: dict = None) -> dict:
        estimators= {}

        self.CheckfeatureDimensions(self.y_true_cols, self.y_pred_cols)

        for Y_true, Y_pred in zip(self.y_true_cols, self.y_pred_cols):# Need to be ordered propelry
            estimiate = nannyml.DLE(y_true=Y_true, 
                                y_pred=Y_pred,
                                feature_column_names=self.cols, 
                                timestamp_column_name="date", 
                                chunk_period=self.period,
                                metrics=self.metrics,
                                thresholds= thresholds if not thresholds else None
                                )

            estimiate.fit(self.reference_df)
            estimiate.estimate(self.analysis_df)
            estimators[Y_true] = estimiate
        return estimators
    
    ''' performance calculator at availbility of ground truth in production'''
    def performance(self,thresholds)-> dict:
        self.CheckfeatureDimensions(self.y_true_cols, self.y_pred_cols)
        performances = {}

        for Y_true, Y_pred in zip(self.y_true_cols, self.y_pred_cols):# Need to be ordered propibly
            pefrormance = nannyml.PerformanceCalculator(y_true=Y_true, 
                                                        y_pred=Y_pred,
                                                        timestamp_column_name="date",
                                                        chunk_period=self.period,
                                                        metrics=self.metrics
                                                    )
            pefrormance.fit(self.reference_df)
            pefrormance.calculate(self.analysis_df)
            performances[Y_true] = pefrormance

        return performances
    

    def multivariate_drift(self, thresholds: dict = None):
        
        md = nannyml.DataReconstructionDriftCalculator(column_names=self.cols, timestamp_column_name="date", 
                                                       chunk_period=self.period, 
                                                       thresholds=thresholds if not thresholds else None)
        
        md.fit(self.reference_df)
        md.estimate(self.analysis_df)
        return md

    def univariate_drift(self, thresholds: dict = None,cont_methods:list[str]=["wasserstein"],
                         cat_methods:list[str]=["l_infinity"]):
        
        ud = nannyml.UnivariateDriftCalculator(column_names=self.cols, timestamp_column_name="date", 
                                              chunk_period=self.period, 
                                              continuous_methods=cont_methods,
                                              categorical_methods=cat_methods,)
        
        

class plots(Monitoring):
    def __init__(self,reference_df:pd.DataFrame,analysis_df:pd.DataFrame,
              cols,horizons:int=24,period:str="2w"):
        super().__init__(reference_df, analysis_df, cols, horizons, period)

    def plots(self, estimators:dict):
        for h,est in estimators.items():
            est.plot().show() 