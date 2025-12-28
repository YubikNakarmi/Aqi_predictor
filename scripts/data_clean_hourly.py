import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import requests
import json


def trend_3(x):
    arr = np.asarray(x)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return np.nan
    t = np.arange(len(arr))
    return np.polyfit(t, arr, 1)[0]


def load_raw_csv(input_csv: str) -> pd.DataFrame:
    return pd.read_csv(input_csv)

def df_load_from_api(input_url: str) -> pd.DataFrame:
    response = requests.get(input_url)
    data = response.json()
    records = data['result']['records']
    df = pd.DataFrame.from_records(records)
    return df



def extract_pm_o3(df: pd.DataFrame) -> pd.DataFrame:
    df_pm25 = (
        df[df["parameter"] == "pm25"]
        .drop(columns=["locationId", "location", "parameter", "unit", "utc", "country", "latitude", "longitude", "city"])
        .rename(columns={"value": "pm25", "local": "date"})
        .reset_index(drop=True)
    )

    df_o3 = (
        df[df["parameter"] == "o3"]
        .drop(columns=["locationId", "location", "parameter", "unit", "utc", "country", "latitude", "longitude", "city"])
        .rename(columns={"value": "o3", "local": "date"})
        .reset_index(drop=True)
    )

    df_merged = df_pm25.merge(df_o3, on="date", how="outer").sort_values("date").reset_index(drop=True)
    df_merged["pm25_target"] = df_merged["pm25"]
    df_merged["o3_target"] = df_merged["o3"]
    return df_merged


def clean_and_index(df: pd.DataFrame) -> pd.DataFrame:
    # Remove negative values
    df.loc[df["pm25"] < 0, "pm25"] = pd.NA
    df.loc[df["o3"] < 0, "o3"] = pd.NA
    df.loc[df["pm25"] > 500, "pm25"] = pd.NA

    # Indexing and frequency
    df["date"] = pd.to_datetime(df["date"]) 
    df.set_index("date", inplace=True)

    # Force to hourly frequency (preserves existing rows)
    deltas = df.index.sort_values().diff().value_counts()
    print("Top time deltas:\n", deltas.head())
    df = df.asfreq("H")

    # Time-based features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    # Missing flags
    df["pm25_missing"] = df["pm25"].isna().astype(int)
    df["o3_missing"] = df["o3"].isna().astype(int)

    # Gap lengths
    pm25_run_id = (df["pm25_missing"] != df["pm25_missing"].shift()).cumsum()
    o3_run_id = (df["o3_missing"] != df["o3_missing"].shift()).cumsum()

    df["pm25_gap_length"] = df["pm25_missing"].groupby(pm25_run_id).transform("sum")
    df["o3_gap_length"] = df["o3_missing"].groupby(o3_run_id).transform("sum")

    return df


def impute_values(df: pd.DataFrame, small_gap=6, medium_gap=12, very_large_gap=24) -> pd.DataFrame:
    """Apply small/medium/large gap imputation and annotate imputation confidence."""
    print("NA counts before imputation:\n", df.isna().sum())

    df_imputation = df.copy()

    # Small gaps: time interpolation
    small_gap_mask_pm25 = df_imputation["pm25_gap_length"] <= small_gap
    small_gap_mask_o3 = df_imputation["o3_gap_length"] <= small_gap

    df_imputation.loc[small_gap_mask_pm25, "pm25"] = (
        df_imputation["pm25"].interpolate(method="time", limit=small_gap)
    )
    df_imputation.loc[small_gap_mask_pm25, "was_imputed"] = int(1)
    df_imputation.loc[small_gap_mask_pm25, "imputation_confidence"] = "high"

    df_imputation.loc[small_gap_mask_o3, "o3"] = (
        df_imputation["o3"].interpolate(method="time", limit=small_gap)
    )
    df_imputation.loc[small_gap_mask_o3, "was_imputed"] = int(1)
    df_imputation.loc[small_gap_mask_o3, "imputation_confidence"] = "high"

    # Medium gaps: KNN imputation with cyclic features
    medium_gap_mask_pm25 = (df_imputation["pm25_gap_length"] > small_gap) & (df_imputation["pm25_gap_length"] <= medium_gap)
    medium_gap_mask_o3 = (df_imputation["o3_gap_length"] > small_gap) & (df_imputation["o3_gap_length"] <= medium_gap)

    df_imputation["hour_sin"] = np.sin(2 * np.pi * df_imputation['hour'] / 24)
    df_imputation["hour_cos"] = np.cos(2 * np.pi * df_imputation['hour'] / 24)
    cols = ['pm25', 'o3', 'hour_sin', 'hour_cos']

    imp = KNNImputer(n_neighbors=6)
    imputed_knn = imp.fit_transform(df_imputation[cols])
    df_all_imputed = pd.DataFrame(imputed_knn, columns=cols, index=df_imputation.index)

    # Apply medium-gap imputations
    df_imputation.loc[medium_gap_mask_pm25, "pm25"] = df_all_imputed.loc[medium_gap_mask_pm25, "pm25"]
    df_imputation.loc[medium_gap_mask_pm25, "was_imputed"] = int(1)
    df_imputation.loc[medium_gap_mask_pm25, "imputation_confidence"] = "medium"

    df_imputation.loc[medium_gap_mask_o3, "o3"] = df_all_imputed.loc[medium_gap_mask_o3, "o3"]
    df_imputation.loc[medium_gap_mask_o3, "was_imputed"] = int(1)
    df_imputation.loc[medium_gap_mask_o3, "imputation_confidence"] = "medium"

    # Large gaps: leave NaN but add low confidence
    large_gap_mask_pm25 = (df_imputation["pm25_gap_length"] > medium_gap) & (df_imputation["pm25_gap_length"] <= very_large_gap)
    large_gap_mask_o3 = (df_imputation["o3_gap_length"] > medium_gap) & (df_imputation["o3_gap_length"] <= very_large_gap)

    df_imputation.loc[large_gap_mask_pm25, "imputation_confidence"] = "low"
    df_imputation.loc[large_gap_mask_o3, "imputation_confidence"] = "low"

    return df_imputation


def add_segmentation(df_imputation: pd.DataFrame, very_large_gap=24) -> pd.DataFrame:
    """Add segment IDs to prevent leakage across very large gaps and reset missing flags."""
    segment = (
        (df_imputation['pm25_missing'].shift(fill_value=0) == 1) & (df_imputation['pm25_gap_length'].shift(fill_value=0) > very_large_gap)
    ) | (
        (df_imputation['o3_missing'].shift(fill_value=0) == 1) & (df_imputation['o3_gap_length'].shift(fill_value=0) > very_large_gap)
    )

    df_imputation["segment_id"] = segment.cumsum()

    # Reset missing flags based on imputed values
    df_imputation["pm25_missing"] = df_imputation["pm25"].isna().astype(int)
    df_imputation["o3_missing"] = df_imputation["o3"].isna().astype(int)

    return df_imputation


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, trend and interaction features."""
    df_engineering = df.copy()

    # Lag features
    lags = [i for i in range(1, 25)]
    for lag in lags:
        df_engineering[f"pm25_lag_{lag}"] = df_engineering.groupby("segment_id")["pm25"].shift(lag)
        df_engineering[f"o3_lag_{lag}"] = df_engineering.groupby("segment_id")["o3"].shift(lag)

    # Rolling means
    windows = [3, 6, 12, 24]
    for w in windows:
        df_engineering[f'pm25_roll_{w}'] = (
            df_engineering.groupby('segment_id')['pm25']
            .apply(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df_engineering[f'o3_roll_{w}'] = (
            df_engineering.groupby('segment_id')['o3']
            .apply(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    # Momentum / trend features
    df_engineering["pm25_slope_3h"] = (
        df_engineering.groupby("segment_id")["pm25"].rolling(window=3, min_periods=3).apply(trend_3, raw=True).reset_index(level=0, drop=True)
    )
    df_engineering["o3_slope_3h"] = (
        df_engineering.groupby("segment_id")["o3"].rolling(window=3, min_periods=3).apply(trend_3, raw=True).reset_index(level=0, drop=True)
    )
    df_engineering["pm25_slope_12h"] = (
        df_engineering.groupby("segment_id")["pm25"].rolling(window=12, min_periods=3).apply(trend_3, raw=True).reset_index(level=0, drop=True)
    )
    df_engineering["o3_slope_12h"] = (
        df_engineering.groupby("segment_id")["o3"].rolling(window=12, min_periods=3).apply(trend_3, raw=True).reset_index(level=0, drop=True)
    )

    # Interaction features
    df_engineering["pm25_o3_ratio"] = df_engineering.groupby("segment_id").apply(lambda g: g["pm25"] / g["o3"]).reset_index(level=0, drop=True)
    df_engineering["pm25_o3_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

    return df_engineering


def save_processed(df: pd.DataFrame, output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv)
    print(f"Saved processed file to: {output_csv}")


def main():

    api_paths=["https://admin.opendatanepal.com/api/action/datastore_search?resource_id=918b774b-2a51-404f-92a0-0be1fe780f90&sort=_id asc&limit=100000",
              "https://admin.opendatanepal.com/api/action/datastore_search?resource_id=ad8d1b4d-7667-455d-ab74-d5966e8ba4c3&sort=_id asc&limit=100000" ]
    


    
    # Paths
    input_csv = r"D:\pypipeline\data\raw\static\hourly\aqi\limited\us_paro_hourly.csv"
    output_csv = r"D:\pypipeline\data\processed\us_paro_hourly_processed.csv"

    df_raw = load_raw_csv(input_csv)

    # Quick check
    neg_count = df_raw[df_raw["value"] < 0]["value"].count()
    print(f"Negative value count: {neg_count}")

    df_merged = extract_pm_o3(df_raw)
    df_clean = clean_and_index(df_merged)

    df_imputed = impute_values(df_clean)
    df_segmented = add_segmentation(df_imputed)

    df_features = engineer_features(df_segmented)

    save_processed(df_features, output_csv)


if __name__ == "__main__":
    main()
