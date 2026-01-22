import pandas as pd
import os
from scripts.train_hourly import split,clean_and_target
from modules.data_hourly_preprocessing import DataCleaner as clean
import datetime
import yaml
import json

DATA_PROCESSED_PATH = os.environ.get("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly/")
DATA_RAW_PATH = os.environ.get("DATA_RAW_PATH", r"data/raw/static/hourly/us_diplomatic_post_hourly.csv")
HORIZON = int(os.environ.get("HORIZON", 24))
TRAIN_SPLIT= float(os.environ.get("TRAIN_SPLIT", 0.7))
VAL_SPLIT= float(os.environ.get("VAL_SPLIT", 0.15))
TEST_SPLIT= float(os.environ.get("TEST_SPLIT", 0.15))
# Target columns to create - comma separated, e.g., "pm25,o3" or "pm25" for single target
TARGET_COLS = os.environ.get("TARGET_COLS", "pm25,o3").split(",")


def metadata(dvc_file_path)->dict:

    with open(dvc_file_path, 'r') as f:
        dvc_data = yaml.safe_load(f)

    metadata = {
        "Created date": str(datetime.datetime.now()),
        "Data source": DATA_RAW_PATH,
        "Data processed path": DATA_PROCESSED_PATH,
        "Horizon": HORIZON,
        "processed_data_md5": dvc_data['outs'][0]['md5'],
        "split_ratios": {
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
            "test_split": TEST_SPLIT
        }
    }
    return metadata


def main():
    # Load raw data from env path
    df = pd.read_csv(DATA_RAW_PATH)
    print(f"Raw data loaded from {DATA_RAW_PATH} with shape {df.shape}")
    main_df_cleaned = clean().run_clean(df)

    # Split the data based on env ratios
    train_df, val_df, test_df = split(main_df_cleaned, train_size=TRAIN_SPLIT, val_size=VAL_SPLIT, test_size=TEST_SPLIT)

    # Clean and create target variables for specified targets
    train_cleaned, val_cleaned, test_cleaned = clean_and_target(
        horizon=HORIZON, 
        train=train_df, 
        val=val_df, 
        test=test_df,
        target_cols=TARGET_COLS
    )

    # Save processed data
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    train_cleaned.to_parquet(os.path.join(DATA_PROCESSED_PATH, 'train_processed.parquet'))
    val_cleaned.to_parquet(os.path.join(DATA_PROCESSED_PATH, 'val_processed.parquet'))
    test_cleaned.to_parquet(os.path.join(DATA_PROCESSED_PATH, 'test_processed.parquet'))
    print(f"Processed data saved to {DATA_PROCESSED_PATH}")

    metadata = metadata(dvc_file_path="data.dvc")
    with open(os.path.join(DATA_PROCESSED_PATH, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)


if __name__ == '__main__':    
    main()