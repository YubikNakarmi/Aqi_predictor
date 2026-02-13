import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import requests
import json
import click
from modules.logging_utils import setup_logging

logger = setup_logging(__name__)

class DataCleaner:
    def __init__(self, input_csv=None, 
                 output_csv=None, 
                 input_df=None, 
                 small_gap=6, 
                 medium_gap=12, 
                 very_large_gap=24):
        
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.input_df = input_df
        self.small_gap = small_gap
        self.medium_gap = medium_gap
        self.very_large_gap = very_large_gap
        self.df_raw = None
        self.df_merged = None
        self.df_clean = None
        self.df_imputed = None
        self.df_segmented = None
        self.df_features = None
        self.native_data = True 

    @staticmethod
    def trend_3(x):
        ''' ploy fit function for imputation(derivative estimation) '''
        arr = np.asarray(x)
        arr = arr[~np.isnan(arr)]
        if arr.size < 2:
            return np.nan
        t = np.arange(len(arr))
        return np.polyfit(t, arr, 1)[0]

    def load_raw(self):
        logger.info("Loading raw data")
        if self.input_df is not None:
            self.df_raw = self.input_df.copy()
        elif self.input_csv is not None:
            self.df_raw = pd.read_csv(self.input_csv)
        else:
            raise ValueError("Either input_df or input_csv must be provided.")
        logger.info("Loaded raw data with shape %s", self.df_raw.shape)
        return self.df_raw
    
    def split(self,df:pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15,train_size: float = 0.7):
        ''' split data into train, validation and test sets based on time '''
        n = len(df)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]
        return df_train, df_val, df_test
     

    @staticmethod
    def df_load_from_api(input_url: str) -> pd.DataFrame:
        ''' work in progress, have not implemented full functionality yet '''
        response = requests.get(input_url)
        data = response.json()
        records = data['result']['records']
        df = pd.DataFrame.from_records(records)
        return df

    @staticmethod
    def extract_pm_o3(df: pd.DataFrame) -> pd.DataFrame:
        df_pm25 = (
            df[df["parameter"] == "pm25"]
            .drop(columns=["locationId", "location", "parameter", 
                           "unit", "utc", "country", "latitude", "longitude", "city"])
            .rename(columns={"value": "pm25", "local": "date"})
            .reset_index(drop=True)
        )
        df_o3 = (
            df[df["parameter"] == "o3"]
            .drop(columns=["locationId", "location", "parameter", 
                           "unit", "utc", "country", "latitude", "longitude", "city"])
            .rename(columns={"value": "o3", "local": "date"})
            .reset_index(drop=True)
        )
        df_merged = df_pm25.merge(df_o3, on="date", how="outer").sort_values("date").reset_index(drop=True)
        df_merged["pm25_target"] = df_merged["pm25"]
        df_merged["o3_target"] = df_merged["o3"]
        return df_merged

    def add_target(self, df: pd.DataFrame,target:str = "pm25",horizon: int = 24) -> pd.DataFrame:
        logger.info("Adding target columns for horizon %s", horizon)
        df = df.copy()
        for h in range(1, horizon + 1):
            df[f"{target}_plus_{h}h"] = df[target].shift(-h)
        return df

    def clean_and_index(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning and indexing data")
        df = df.copy()
        df.loc[df["pm25"] < 0, "pm25"] = pd.NA
        df.loc[df["o3"] < 0, "o3"] = pd.NA
        df.loc[df["pm25"] > 500, "pm25"] = pd.NA
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Data must have a 'date' column or a DatetimeIndex.")
        deltas = df.index.sort_values().diff().value_counts()
        logger.info("Top time deltas:\n%s", deltas.head())

        df = df.asfreq("H")
        logger.info("Cleaned and indexed data with shape %s", df.shape)
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding time-based features")
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DatetimeIndex required for time features.")
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        return df

    def add_missing_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding missing-value flags")
        df = df.copy()
        df["pm25_missing"] = df["pm25"].isna().astype(int)
        df["o3_missing"] = df["o3"].isna().astype(int)
        return df

    def add_gap_length(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing gap length features")
        df = df.copy()
        if "pm25_missing" not in df.columns or "o3_missing" not in df.columns:
            df = self.add_missing_flags(df)
        pm25_run_id = (df["pm25_missing"] != df["pm25_missing"].shift()).cumsum()
        o3_run_id = (df["o3_missing"] != df["o3_missing"].shift()).cumsum()
        df["pm25_gap_length"] = df["pm25_missing"].groupby(pm25_run_id).transform("sum")
        df["o3_gap_length"] = df["o3_missing"].groupby(o3_run_id).transform("sum")
        logger.info("Gap length features added")
        return df
    
    def impute_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Imputing missing values")
        logger.info("NA counts before imputation:\n%s", df.isna().sum())
        df_imputation = df.copy()
        if not isinstance(df_imputation.index, pd.DatetimeIndex):
            raise ValueError("DatetimeIndex required for time-based imputation.")
        if "pm25_gap_length" not in df_imputation.columns or "o3_gap_length" not in df_imputation.columns:
            df_imputation = self.add_gap_length(df_imputation)
        if "hour" not in df_imputation.columns:
            df_imputation = self.add_time_features(df_imputation)
        if "was_imputed" not in df_imputation.columns:
            df_imputation["was_imputed"] = 0
        if "imputation_confidence" not in df_imputation.columns:
            df_imputation["imputation_confidence"] = "none"
        small_gap_mask_pm25 = df_imputation["pm25_gap_length"] <= self.small_gap
        small_gap_mask_o3 = df_imputation["o3_gap_length"] <= self.small_gap
        logger.info("Applying small-gap interpolation")
        df_imputation.loc[small_gap_mask_pm25, "pm25"] = (
            df_imputation["pm25"].interpolate(method="time", limit=self.small_gap)
        )
        df_imputation.loc[small_gap_mask_pm25, "was_imputed"] = int(1)
        df_imputation.loc[small_gap_mask_pm25, "imputation_confidence"] = "high"
        df_imputation.loc[small_gap_mask_o3, "o3"] = (
            df_imputation["o3"].interpolate(method="time", limit=self.small_gap)
        )
        df_imputation.loc[small_gap_mask_o3, "was_imputed"] = int(1)
        df_imputation.loc[small_gap_mask_o3, "imputation_confidence"] = "high"
        medium_gap_mask_pm25 = (df_imputation["pm25_gap_length"] > self.small_gap) & \
        (df_imputation["pm25_gap_length"] <= self.medium_gap)
        medium_gap_mask_o3 = (df_imputation["o3_gap_length"] > self.small_gap) & \
        (df_imputation["o3_gap_length"] <= self.medium_gap)
        logger.info("Applying medium-gap KNN imputation")
        cols = ['pm25', 'o3', 'hour_sin', 'hour_cos']
        imp = KNNImputer(n_neighbors=6)

        imputed_knn = imp.fit_transform(df_imputation[cols])
        df_all_imputed = pd.DataFrame(imputed_knn, columns=cols, index=df_imputation.index)
        df_imputation.loc[medium_gap_mask_pm25, "pm25"] = df_all_imputed.loc[medium_gap_mask_pm25, "pm25"]
        df_imputation.loc[medium_gap_mask_pm25, "was_imputed"] = int(1)
        df_imputation.loc[medium_gap_mask_pm25, "imputation_confidence"] = "medium"
        df_imputation.loc[medium_gap_mask_o3, "o3"] = df_all_imputed.loc[medium_gap_mask_o3, "o3"]
        df_imputation.loc[medium_gap_mask_o3, "was_imputed"] = int(1)
        df_imputation.loc[medium_gap_mask_o3, "imputation_confidence"] = "medium"

        large_gap_mask_pm25 = (df_imputation["pm25_gap_length"] > self.medium_gap) & (df_imputation["pm25_gap_length"] <= self.very_large_gap)
        large_gap_mask_o3 = (df_imputation["o3_gap_length"] > self.medium_gap) & (df_imputation["o3_gap_length"] <= self.very_large_gap)
        logger.info("Tagging large-gap confidence")
        df_imputation.loc[large_gap_mask_pm25, "imputation_confidence"] = "low"
        df_imputation.loc[large_gap_mask_o3, "imputation_confidence"] = "low"
        logger.info("Imputation complete")
        return df_imputation

    def add_segmentation(self, df_imputation: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding segment identifiers")
        df_imputation = df_imputation.copy()
        if "pm25_gap_length" not in df_imputation.columns or "o3_gap_length" not in df_imputation.columns:
            df_imputation = self.add_gap_length(df_imputation)
        segment = (
            (df_imputation['pm25_missing'].shift(fill_value=0) == 1) & (df_imputation['pm25_gap_length'].shift(fill_value=0) > self.very_large_gap)
        ) | (
            (df_imputation['o3_missing'].shift(fill_value=0) == 1) & (df_imputation['o3_gap_length'].shift(fill_value=0) > self.very_large_gap)
        )
        df_imputation["segment_id"] = segment.cumsum()
        logger.info("Segmentation complete")
        return df_imputation

    def engineer_features(self, df: pd.DataFrame, segment_col: str = "segment_id") -> pd.DataFrame:
        logger.info("Engineering features")
        df_engineering = df.copy()
        if segment_col not in df_engineering.columns:
            df_engineering[segment_col] = 0
        lags = [i for i in range(1, 25)]
        logger.info("Adding lag features")
        for lag in lags:
            df_engineering[f"pm25_lag_{lag}"] = df_engineering.groupby(segment_col)["pm25"].shift(lag)
            df_engineering[f"o3_lag_{lag}"] = df_engineering.groupby(segment_col)["o3"].shift(lag)
        windows = [3, 6, 12, 24]
        logger.info("Adding rolling window features")
        for w in windows:
            df_engineering[f'pm25_roll_{w}'] = (
                df_engineering.groupby(segment_col)["pm25"]
                .apply(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
            df_engineering[f'o3_roll_{w}'] = (
                df_engineering.groupby(segment_col)["o3"]
                .apply(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
        logger.info("Adding slope and ratio features")
        df_engineering["pm25_slope_3h"] = (
            df_engineering.groupby(segment_col)["pm25"].rolling(window=3, min_periods=3).apply(self.trend_3, raw=True).reset_index(level=0, drop=True)
        )
        df_engineering["o3_slope_3h"] = (
            df_engineering.groupby(segment_col)["o3"].rolling(window=3, min_periods=3).apply(self.trend_3, raw=True).reset_index(level=0, drop=True)
        )
        df_engineering["pm25_slope_12h"] = (
            df_engineering.groupby(segment_col)["pm25"].rolling(window=12, min_periods=3).apply(self.trend_3, raw=True).reset_index(level=0, drop=True)
        )
        df_engineering["o3_slope_12h"] = (
            df_engineering.groupby(segment_col)["o3"].rolling(window=12, min_periods=3).apply(self.trend_3, raw=True).reset_index(level=0, drop=True)
        )
        ratio = df_engineering["pm25"] / df_engineering["o3"]
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        df_engineering["pm25_o3_ratio"] = ratio
        logger.info("Feature engineering complete with shape %s", df_engineering.shape)
        return df_engineering

    def save_processed(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df.to_csv(self.output_csv)
        logger.info("Saved processed file to: %s", self.output_csv)

    def run(self):
        logger.info("Starting full cleaning pipeline")
        self.load_raw()
        neg_count = self.df_raw[self.df_raw["value"] < 0]["value"].count()
        logger.info("Negative value count: %s", neg_count)
        self.df_merged = self.extract_pm_o3(self.df_raw)
        self.df_clean = self.clean_and_index(self.df_merged)
        self.df_clean = self.add_time_features(self.df_clean)
        self.df_clean = self.add_missing_flags(self.df_clean)
        self.df_clean = self.add_gap_length(self.df_clean)

        self.df_imputed = self.impute_values(self.df_clean)
        self.df_segmented = self.add_segmentation(self.df_imputed)
        self.df_features = self.engineer_features(self.df_segmented)
        if self.output_csv:
            self.save_processed(self.df_features)
        logger.info("Full cleaning pipeline complete")
            
    def run_clean(self,df_raw:pd.DataFrame)->pd.DataFrame:
        ''' Runs full cleaning pipeline from raw dataframe input '''
        logger.info("Starting clean-only pipeline")
        if df_raw is None:
            df_raw = self.input_df #defaults to attribute input_df if no argument provided

        self.df_raw = df_raw
        self.df_merged = self.extract_pm_o3(self.df_raw)
        self.df_clean = self.clean_and_index(self.df_merged)
        self.df_clean = self.add_time_features(self.df_clean)
        self.df_clean = self.add_missing_flags(self.df_clean)
        if self.output_csv:
            self.save_processed(self.df_clean)
        logger.info("Clean-only pipeline complete")
        return self.df_clean
        
    def run_feature_engineering(self, cleaned_df: pd.DataFrame)->pd.DataFrame:
        """Assumes cleaned_df already has pm25/o3 columns and datetime index."""
        logger.info("Starting feature engineering pipeline")
        self.df_clean = cleaned_df.copy()
        if "hour" not in self.df_clean.columns:
            self.df_clean = self.add_time_features(self.df_clean)
        if "pm25_missing" not in self.df_clean.columns or "o3_missing" not in self.df_clean.columns:
            self.df_clean = self.add_missing_flags(self.df_clean)
        self.df_clean = self.add_gap_length(self.df_clean)
        self.df_imputed = self.impute_values(self.df_clean)
        self.df_segmented = self.add_segmentation(self.df_imputed)
        self.df_features = self.engineer_features(self.df_segmented)
        if self.output_csv:
            self.save_processed(self.df_features)
        logger.info("Feature engineering pipeline complete")
        return self.df_features

