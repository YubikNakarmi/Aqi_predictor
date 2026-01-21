# Usage Examples for Customizable Target Training

The library now supports training models for any pollutant target (pm25, o3, pm10, etc.) with configurable value ranges.

## Using Environment Variables

### Training PM2.5 Models (default)
```bash
export TARGET_COL="pm25"
export VALUE_MIN=0
export VALUE_MAX=500
export HORIZON=24
python hourly_pipeline/train.py
```

### Training O3 (Ozone) Models
```bash
export TARGET_COL="o3"
export VALUE_MIN=0
export VALUE_MAX=300  # O3 typically has different range
export HORIZON=24
python hourly_pipeline/train.py
```

### Training PM10 Models
```bash
export TARGET_COL="pm10"
export VALUE_MIN=0
export VALUE_MAX=600
export HORIZON=24
python hourly_pipeline/train.py
```

## Direct Function Calls

### Example 1: Train PM2.5 Models
```python
from scripts.train_hourly import train, clean_and_target
import pandas as pd

# Load and prepare data with pm25 target
train_df, val_df, test_df = clean_and_target(
    horizon=24,
    train=train_raw,
    val=val_raw,
    test=test_raw,
    target_cols=["pm25"]  # Only create pm25 targets
)

# Train pm25 models
models, metrics, signature = train(
    horizons=24,
    best_params=best_params,
    df_train=train_df,
    df_val=val_df,
    target_col="pm25",
    value_range=(0, 500)
)
```

### Example 2: Train O3 Models
```python
from scripts.train_hourly import train, clean_and_target
import pandas as pd

# Load and prepare data with o3 target
train_df, val_df, test_df = clean_and_target(
    horizon=24,
    train=train_raw,
    val=val_raw,
    test=test_raw,
    target_cols=["o3"]  # Only create o3 targets
)

# Train o3 models
models, metrics, signature = train(
    horizons=24,
    best_params=best_params,
    df_train=train_df,
    df_val=val_df,
    target_col="o3",
    value_range=(0, 300)  # O3 has different valid range
)
```

### Example 3: Prepare Multiple Targets, Train One
```python
from scripts.train_hourly import train, clean_and_target
import pandas as pd

# Prepare data for both pm25 and o3
train_df, val_df, test_df = clean_and_target(
    horizon=24,
    train=train_raw,
    val=val_raw,
    test=test_raw,
    target_cols=["pm25", "o3", "pm10"]  # Create all targets
)

# Train only pm25 models
pm25_models, pm25_metrics, _ = train(
    horizons=24,
    best_params=pm25_best_params,
    df_train=train_df,
    df_val=val_df,
    target_col="pm25",
    value_range=(0, 500)
)

# Train o3 models with different hyperparameters
o3_models, o3_metrics, _ = train(
    horizons=24,
    best_params=o3_best_params,
    df_train=train_df,
    df_val=val_df,
    target_col="o3",
    value_range=(0, 300)
)
```

## Tuning for Different Targets

```python
from scripts.train_hourly import tune

# Tune hyperparameters specifically for O3
o3_best_params = tune(
    df_train=train_df,
    df_val=val_df,
    trials=60,
    target_col="o3",
    value_range=(0, 300),
    horizons=24
)

# Tune for PM2.5
pm25_best_params = tune(
    df_train=train_df,
    df_val=val_df,
    trials=60,
    target_col="pm25",
    value_range=(0, 500),
    horizons=24
)
```

## Testing Models

```python
from scripts.train_hourly import test

# Test pm25 models
pm25_test_metrics = test(
    horizons=24,
    models=pm25_models,
    df_test=test_df,
    target_col="pm25",
    value_range=(0, 500)
)

# Test o3 models
o3_test_metrics = test(
    horizons=24,
    models=o3_models,
    df_test=test_df,
    target_col="o3",
    value_range=(0, 300)
)

print("PM2.5 Test Metrics:", pm25_test_metrics)
print("O3 Test Metrics:", o3_test_metrics)
```

## Value Ranges by Pollutant

Common value ranges for different pollutants:

- **PM2.5**: (0, 500) - Air Quality Index scale
- **PM10**: (0, 600) - Higher than PM2.5
- **O3**: (0, 300) - Ozone concentration
- **NO2**: (0, 200) - Nitrogen dioxide
- **SO2**: (0, 200) - Sulfur dioxide
- **CO**: (0, 50) - Carbon monoxide (ppm)

Adjust these based on your data's actual range and measurement units.
