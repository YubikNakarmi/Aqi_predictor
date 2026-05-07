"""Data preprocessing module for hourly AQI forecasting, including loading, cleaning, imputation, feature engineering, and saving processed data."""

import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import requests

try:
    from .logging_utils import setup_logging
except ImportError:
    from modules.logging_utils import setup_logging

logger = setup_logging(__name__)

class DataCleaner:
    def __init__(self,
                 input_csv=None,
                 output_csv=None,
                 input_df=None,
                 small_gap=6,
                 medium_gap=12,
                 very_large_gap=24,
                 target_features: list[str] | None = None,
                 feature_upper_bounds: dict[str, float] | None = None):

        self.input_csv = input_csv
        self.output_csv = output_csv
        self.input_df = input_df
        self.small_gap = small_gap
        self.medium_gap = medium_gap
        self.very_large_gap = very_large_gap
        self.target_features = target_features or ["pm25", "o3"]
        self.feature_upper_bounds = feature_upper_bounds or {"pm25": 500}
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

    def split(self, df: pd.DataFrame, val_size: float = 0.15, train_size: float = 0.7):
        """Split data into train, validation and test sets based on time."""
        n = len(df)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]
        return df_train, df_val, df_test

    @staticmethod
    def df_load_from_api(input_url: str) -> pd.DataFrame:
        response = requests.get(input_url)
        data = response.json()
        records = data["result"]["records"]
        df = pd.DataFrame.from_records(records)
        return df

    @staticmethod
    def extract_pm_o3(df: pd.DataFrame,
                      extract_columns: list[str] | None = None,
                      exclude_columns: list[str] | None = None,
                      rename: dict[str, str] | None = None,
                      parameter: str = "parameter",
                      date: str = "date") -> pd.DataFrame:
        extract_columns = extract_columns or ["pm25", "o3"]
        exclude_columns = exclude_columns or [
            "locationId", "location", "parameter", "unit", "utc",
            "country", "latitude", "longitude", "city"
        ]
        rename = rename or {"value": "value", "local": "date"}

        if parameter not in df.columns:
            raise ValueError(f"Column '{parameter}' not found in DataFrame.")

        df_all = {}
        param_values = set(df[parameter].dropna().astype(str))

        for col in extract_columns:
            if col not in param_values:
                raise ValueError(f"Value '{col}' not found in column '{parameter}'.")
            df_extract = df[df[parameter] == col].copy()
            df_extract = df_extract.drop(columns=exclude_columns, errors="ignore").rename(columns=rename).reset_index(drop=True)
            value_col = rename.get("value", "value")
            if value_col not in df_extract.columns:
                raise ValueError(f"Column '{value_col}' not found after rename.")
            df_extract = df_extract.rename(columns={value_col: col})
            df_all[col] = df_extract

        df_merged = None
        for col, df_extract in df_all.items():
            logger.info("Extracted %s with shape %s", col, df_extract.shape)
            df_merged = df_extract if df_merged is None else df_merged.merge(df_extract, on=date, how="outer")

        for col in extract_columns:
            df_merged[f"{col}_target"] = df_merged[col]
        logger.info("Merged extracted data with shape %s", df_merged.shape)
        return df_merged

    def add_target(self, df: pd.DataFrame, target: str = "pm25", horizon: int = 24) -> pd.DataFrame:
        """Add target columns for forecasting horizon, e.g., pm25_plus_1h, pm25_plus_2h, ..., pm25_plus_24h."""

        logger.info("Adding target columns for horizon %s", horizon)
        df = df.copy()
        for h in range(1, horizon + 1):
            df[f"{target}_plus_{h}h"] = df[target].shift(-h)
        return df

    def clean_and_index(self,
                        df: pd.DataFrame,
                        target_features: list[str] | None = None,
                        feature_upper_bounds: dict[str, float] | None = None,
                        date_feature: str = "date") -> pd.DataFrame:
        """Clean the data by handling negative values, applying upper bounds, converting date column to datetime, and setting it as index. Also logs the distribution of time deltas between records."""


        logger.info("Cleaning and indexing data")
        df = df.copy()
        target_features = target_features or self.target_features
        feature_upper_bounds = feature_upper_bounds or self.feature_upper_bounds

        for feature in target_features:
            if feature not in df.columns:
                raise ValueError(f"Column '{feature}' not found in DataFrame.")
            df.loc[df[feature] < 0, feature] = pd.NA
            if feature in feature_upper_bounds:
                df.loc[df[feature] > feature_upper_bounds[feature], feature] = pd.NA

        if date_feature in df.columns:
            df[date_feature] = pd.to_datetime(df[date_feature])
            df.set_index(date_feature, inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Data must have a 'date' column or a DatetimeIndex.")

        deltas = df.index.sort_values().diff().value_counts()
        logger.info("Top time deltas:\n%s", deltas.head())

        df = df.asfreq("h")
        logger.info("Cleaned and indexed data with shape %s", df.shape)
        return df

    def add_time_features(self, df: pd.DataFrame, date_feature: str = "date") -> pd.DataFrame:
        """Add time-based features such as hour of day, day of week, is_weekend, is_night, and cyclic encoding of hour."""

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

    def add_missing_flags(self, df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
        """Add binary flags indicating missing values for specified features."""

        logger.info("Adding missing-value flags")
        df = df.copy()
        features = features or self.target_features

        for feature in features:
            if feature not in df.columns:
                raise ValueError(f"Column '{feature}' not found in DataFrame.")
            df[f"{feature}_missing"] = df[feature].isna().astype(int)
        return df

    def add_gap_length(self, df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
        """Add gap length features for specified features."""

        logger.info("Computing gap length features")
        df = df.copy()
        features = features or self.target_features

        if any(f"{feature}_missing" not in df.columns for feature in features):
            df = self.add_missing_flags(df, features=features)

        for feature in features:
            missing_col = f"{feature}_missing"
            run_id = (df[missing_col] != df[missing_col].shift()).cumsum()
            df[f"{feature}_gap_length"] = df[missing_col].groupby(run_id).transform("sum")

        logger.info("Gap length features added")
        return df

    def impute_values(self,
                      df: pd.DataFrame,
                      features: list[str] | None = None,
                      cyclic_features: list[str] | None = None) -> pd.DataFrame:
        """Impute missing values using a combination of time-based interpolation for small gaps and KNN imputation for medium gaps, while tagging imputation confidence based on gap length."""

        logger.info("Imputing missing values")
        logger.info("NA counts before imputation:\n%s", df.isna().sum())
        df_imputation = df.copy()
        features = features or self.target_features
        cyclic_features = cyclic_features or ["hour_sin", "hour_cos"]

        if not isinstance(df_imputation.index, pd.DatetimeIndex):
            raise ValueError("DatetimeIndex required for time-based imputation.")

        if any(f"{feature}_gap_length" not in df_imputation.columns for feature in features):
            df_imputation = self.add_gap_length(df_imputation, features=features)

        if "hour" not in df_imputation.columns:
            df_imputation = self.add_time_features(df_imputation)

        if "was_imputed" not in df_imputation.columns:
            df_imputation["was_imputed"] = 0
        if "imputation_confidence" not in df_imputation.columns:
            df_imputation["imputation_confidence"] = "none"

        logger.info("Applying small-gap interpolation")
        for feature in features:
            gap_col = f"{feature}_gap_length"
            small_gap_mask = df_imputation[gap_col] <= self.small_gap
            df_imputation.loc[small_gap_mask, feature] = (
                df_imputation[feature].interpolate(method="time", limit=self.small_gap)
            )
            df_imputation.loc[small_gap_mask, "was_imputed"] = 1
            df_imputation.loc[small_gap_mask, "imputation_confidence"] = "high"

        logger.info("Applying medium-gap KNN imputation")
        cols = [col for col in (features + cyclic_features) if col in df_imputation.columns]
        imp = KNNImputer(n_neighbors=6)
        imputed_knn = imp.fit_transform(df_imputation[cols])
        df_all_imputed = pd.DataFrame(imputed_knn, columns=cols, index=df_imputation.index)

        for feature in features:
            gap_col = f"{feature}_gap_length"
            medium_gap_mask = (df_imputation[gap_col] > self.small_gap) & (df_imputation[gap_col] <= self.medium_gap)
            df_imputation.loc[medium_gap_mask, feature] = df_all_imputed.loc[medium_gap_mask, feature]
            df_imputation.loc[medium_gap_mask, "was_imputed"] = 1
            df_imputation.loc[medium_gap_mask, "imputation_confidence"] = "medium"

        logger.info("Tagging large-gap confidence")
        for feature in features:
            gap_col = f"{feature}_gap_length"
            large_gap_mask = (df_imputation[gap_col] > self.medium_gap) & (df_imputation[gap_col] <= self.very_large_gap)
            df_imputation.loc[large_gap_mask, "imputation_confidence"] = "low"

        logger.info("Imputation complete")
        return df_imputation

    def add_segmentation(self, df_imputation: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
        """Add segment identifiers to the DataFrame based on large gaps in the data, which can be used to prevent data leakage during feature engineering and modeling."""

        logger.info("Adding segment identifiers")
        df_imputation = df_imputation.copy()
        features = features or self.target_features

        if any(f"{feature}_gap_length" not in df_imputation.columns for feature in features):
            df_imputation = self.add_gap_length(df_imputation, features=features)

        segment = pd.Series(False, index=df_imputation.index)
        for feature in features:
            missing_col = f"{feature}_missing"
            gap_col = f"{feature}_gap_length"
            segment = segment | (
                (df_imputation[missing_col].shift(fill_value=0) == 1)
                & (df_imputation[gap_col].shift(fill_value=0) > self.very_large_gap)
            )

        df_imputation["segment_id"] = segment.cumsum()
        logger.info("Segmentation complete")
        return df_imputation

    def engineer_features(self,
                          df: pd.DataFrame,
                          segment_col: str = "segment_id",
                          target_features: list[str] | None = None,
                          lag_horizon: int = 24,
                          windows: list[int] | None = None,
                          slope_windows: list[int] | None = None,
                          ratio_pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
        """Engineer features such as lag features, rolling window statistics, slope features, and ratio features, while ensuring that feature engineering is done separately within each segment to prevent data leakage."""

        logger.info("Engineering features")
        df_engineering = df.copy()
        target_features = target_features or self.target_features
        windows = windows or [3, 6, 12, 24]
        slope_windows = slope_windows or [3, 12]

        if segment_col not in df_engineering.columns:
            df_engineering[segment_col] = 0

        logger.info("Adding lag features")
        for lag in range(1, lag_horizon + 1):
            for feature in target_features:
                df_engineering[f"{feature}_lag_{lag}"] = df_engineering.groupby(segment_col)[feature].shift(lag)

        logger.info("Adding rolling window features")
        for w in windows:
            for feature in target_features:
                df_engineering[f"{feature}_roll_{w}"] = (
                    df_engineering.groupby(segment_col)[feature]
                    .apply(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                    .reset_index(level=0, drop=True)
                )

        logger.info("Adding slope and ratio features")
        for w in slope_windows:
            for feature in target_features:
                min_periods = min(3, w)
                df_engineering[f"{feature}_slope_{w}h"] = (
                    df_engineering.groupby(segment_col)[feature]
                    .rolling(window=w, min_periods=min_periods)
                    .apply(self.trend_3, raw=True)
                    .reset_index(level=0, drop=True)
                )

        if ratio_pairs is None and len(target_features) >= 2:
            ratio_pairs = [(target_features[0], target_features[1])]
        ratio_pairs = ratio_pairs or []

        for num_feature, den_feature in ratio_pairs:
            ratio = df_engineering[num_feature] / df_engineering[den_feature]
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            df_engineering[f"{num_feature}_{den_feature}_ratio"] = ratio

        logger.info("Feature engineering complete with shape %s", df_engineering.shape)
        return df_engineering

    def save_processed(self, df: pd.DataFrame):
        """Save the processed DataFrame to a CSV file if output_csv is set, with error handling and logging."""

        if not self.output_csv:
            raise ValueError("output_csv must be set before saving.")
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df.to_csv(self.output_csv)
        logger.info("Saved processed file to: %s", self.output_csv)

    def run_feature_engineering(self,
                                cleaned_df: pd.DataFrame,
                                target_features: list[str] | None = None,
                                lag_horizon: int = 24,
                                windows: list[int] | None = None,
                                slope_windows: list[int] | None = None,
                                ratio_pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
        """Runs the feature engineering pipeline starting from a cleaned DataFrame, allowing for optional specification of target features, lag horizon, rolling windows, slope windows, and ratio pairs."""

        logger.info("Starting feature engineering pipeline")
        target_features = target_features or self.target_features
        self.df_clean = cleaned_df.copy()

        if "hour" not in self.df_clean.columns:
            self.df_clean = self.add_time_features(self.df_clean)
        if any(f"{feature}_missing" not in self.df_clean.columns for feature in target_features):
            self.df_clean = self.add_missing_flags(self.df_clean, features=target_features)

        self.df_clean = self.add_gap_length(self.df_clean, features=target_features)
        self.df_imputed = self.impute_values(self.df_clean, features=target_features)
        self.df_segmented = self.add_segmentation(self.df_imputed, features=target_features)
        self.df_features = self.engineer_features(
            self.df_segmented,
            target_features=target_features,
            lag_horizon=lag_horizon,
            windows=windows,
            slope_windows=slope_windows,
            ratio_pairs=ratio_pairs,
        )
        if self.output_csv:
            self.save_processed(self.df_features)
        logger.info("Feature engineering pipeline complete")
        return self.df_features


    ''' run full pipeline '''
    def run(self):
       """Runs the full data preprocessing pipeline from loading raw data to saving engineered features, with logging at each step.""" 


        logger.info("Starting full cleaning pipeline")
        target_features = self.target_features
        self.load_raw()
        if "value" in self.df_raw.columns:
            neg_count = self.df_raw[self.df_raw["value"] < 0]["value"].count()
            logger.info("Negative value count: %s", neg_count)

    


        self.df_merged = self.extract_pm_o3(self.df_raw, extract_columns=target_features)
        self.df_clean = self.clean_and_index(self.df_merged, target_features=target_features)
        self.df_clean = self.add_time_features(self.df_clean)
        self.df_clean = self.add_missing_flags(self.df_clean, features=target_features)
        self.df_clean = self.add_gap_length(self.df_clean, features=target_features)
        self.df_imputed = self.impute_values(self.df_clean, features=target_features)
        self.df_segmented = self.add_segmentation(self.df_imputed, features=target_features)
        self.df_features = self.engineer_features(self.df_segmented, target_features=target_features)
        if self.output_csv:
            self.save_processed(self.df_features)
        logger.info("Full cleaning pipeline complete")

    def run_clean(self,
                  df_raw: pd.DataFrame,
                  target_features: list[str] | None = None,
                  feature_upper_bounds: dict[str, float] | None = None) -> pd.DataFrame:
        """Runs cleaning pipeline from raw dataframe input."""


        logger.info("Starting clean-only pipeline")
        target_features = target_features or self.target_features
        feature_upper_bounds = feature_upper_bounds or self.feature_upper_bounds
        if df_raw is None:
            df_raw = self.input_df

        self.df_raw = df_raw
        self.df_merged = self.extract_pm_o3(self.df_raw, extract_columns=target_features)
        self.df_clean = self.clean_and_index(
            self.df_merged,
            target_features=target_features,
            feature_upper_bounds=feature_upper_bounds,
        )
        self.df_clean = self.add_time_features(self.df_clean)
        self.df_clean = self.add_missing_flags(self.df_clean, features=target_features)
        if self.output_csv:
            self.save_processed(self.df_clean)
        logger.info("Clean-only pipeline complete")
        return self.df_clean



