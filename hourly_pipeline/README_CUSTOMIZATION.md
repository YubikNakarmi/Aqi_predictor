# Customizable Target Training - Quick Reference

## Summary of Changes

The library is now fully customizable to train models for any pollutant target (PM2.5, O3, PM10, NO2, etc.) with configurable value ranges.

## Key Parameters Added

### `clean_and_target()`
- **`target_cols`** (list): List of target columns to create (e.g., `["pm25"]`, `["o3"]`, `["pm25", "o3"]`)
  - Default: `["pm25", "o3"]`

### `objective()`, `tune()`, `train()`, `test()`
- **`target_col`** (str): Single target column to train/tune (e.g., `"pm25"`, `"o3"`)
  - Default: `"pm25"`
- **`value_range`** (tuple): Valid range for target values as `(min, max)`
  - Default: `(0, 500)` for PM2.5

## Environment Variables

Set these in your shell or `.env` file:

```bash
# For preprocessing
export TARGET_COLS="pm25,o3"      # Comma-separated targets to create
export HORIZON=24
export DATA_RAW_PATH="data/raw/static/hourly/..."
export DATA_PROCESSED_PATH="data/processed/hourly/..."

# For training
export TARGET_COL="pm25"          # Single target to train
export VALUE_MIN=0
export VALUE_MAX=500
export MLFLOW_TRACKING_URI="http://localhost:5000"
export ARTIFACTS_PATH="/data/artifacts/"
```

## Quick Usage Examples

### Train PM2.5 Models
```bash
export TARGET_COL="pm25"
export VALUE_MIN=0
export VALUE_MAX=500
python hourly_pipeline/train.py
```

### Train O3 Models
```bash
export TARGET_COL="o3"
export VALUE_MIN=0
export VALUE_MAX=300
python hourly_pipeline/train.py
```

### Preprocess for Multiple Targets
```bash
export TARGET_COLS="pm25,o3,pm10"
python hourly_pipeline/preprocess.py
```

## Python API

```python
from scripts.train_hourly import train, tune, test, clean_and_target

# 1. Prepare data for multiple targets
train_df, val_df, test_df = clean_and_target(
    horizon=24,
    train=train_raw,
    val=val_raw,
    test=test_raw,
    target_cols=["pm25", "o3"]
)

# 2. Tune for specific target
best_params = tune(
    df_train=train_df,
    df_val=val_df,
    trials=60,
    target_col="pm25",
    value_range=(0, 500),
    horizons=24
)

# 3. Train models
models, metrics, sig = train(
    horizons=24,
    best_params=best_params,
    df_train=train_df,
    df_val=val_df,
    target_col="pm25",
    value_range=(0, 500)
)

# 4. Test models
test_metrics = test(
    horizons=24,
    models=models,
    df_test=test_df,
    target_col="pm25",
    value_range=(0, 500)
)
```

## Common Value Ranges

| Pollutant | Range | Unit | Notes |
|-----------|-------|------|-------|
| PM2.5 | (0, 500) | AQI | Default |
| PM10 | (0, 600) | AQI | Coarser particles |
| O3 | (0, 300) | ppb | Ozone |
| NO2 | (0, 200) | ppb | Nitrogen dioxide |
| SO2 | (0, 200) | ppb | Sulfur dioxide |
| CO | (0, 50) | ppm | Carbon monoxide |

## Model Naming Convention

Models are now named with the target:
- `pm25_plus_1h_model`, `pm25_plus_2h_model`, ...
- `o3_plus_1h_model`, `o3_plus_2h_model`, ...
- `pm10_plus_1h_model`, ...

## MLflow Integration

Experiments are automatically named with the target:
- `xgb_pm25_hourly_training`
- `xgb_o3_hourly_training`

Runs are tagged with `{"target": "pm25"}` or `{"target": "o3"}` for easy filtering.

## Backward Compatibility

All functions maintain backward compatibility:
- Without specifying `target_col`, defaults to `"pm25"`
- Without specifying `value_range`, defaults to `(0, 500)`
- Without specifying `target_cols`, creates both `["pm25", "o3"]`
