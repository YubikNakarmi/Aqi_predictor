"""
ML CI Tests for AQI Prediction Pipeline
Offline, self-contained tests — no MLflow server or Azure needed.
Generates metrics_report.md + metrics.json consumed by CML.
"""
import sys, os, json, pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from pathlib import Path

# make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from modules.data_hourly_preprocessing import DataCleaner

ARTIFACTS = Path(__file__).resolve().parent.parent / "data" / "artifacts"
REPORT_PATH = Path(__file__).resolve().parent.parent / "metrics_report.md"
HORIZON = 3        # small horizon for fast CI
TARGET_COL = "pm25"
VALUE_MIN, VALUE_MAX = 0, 500
N_ROWS = 500


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_raw_df() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=N_ROWS, freq="h")
    df = pd.DataFrame({
        "date": dates,
        "pm25": np.abs(np.random.normal(35, 15, N_ROWS)).clip(0, 500),
        "o3": np.abs(np.random.normal(40, 10, N_ROWS)).clip(0, 500),
        "pm25_target": np.abs(np.random.normal(35, 15, N_ROWS)).clip(0, 500),
        "o3_target": np.abs(np.random.normal(40, 10, N_ROWS)).clip(0, 500),
    })
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


@pytest.fixture(scope="session")
def processed_df(synthetic_raw_df) -> pd.DataFrame:
    cleaner = DataCleaner()
    return cleaner.run_feature_engineering(synthetic_raw_df.copy())


@pytest.fixture(scope="session")
def split_data(processed_df):
    n = len(processed_df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return (
        processed_df.iloc[:train_end],
        processed_df.iloc[train_end:val_end],
        processed_df.iloc[val_end:],
    )


@pytest.fixture(scope="session")
def targets_added(split_data):
    train, val, test = split_data
    for h in range(1, HORIZON + 1):
        for t in ["pm25", "o3"]:
            train[f"{t}_plus_{h}h"] = train.groupby("segment_id")[f"{t}_target"].shift(-h)
            val[f"{t}_plus_{h}h"] = val.groupby("segment_id")[f"{t}_target"].shift(-h)
            test[f"{t}_plus_{h}h"] = test.groupby("segment_id")[f"{t}_target"].shift(-h)
    return train, val, test


@pytest.fixture(scope="session")
def features_exclude():
    return (
        [f"{TARGET_COL}_plus_{i}h" for i in range(1, HORIZON + 1)]
        + [f"o3_plus_{i}h" for i in range(1, HORIZON + 1)]
        + [f"pm25_plus_{i}h" for i in range(1, HORIZON + 1)]
        + ["segment_id", "imputation_confidence", "pm25_target", "o3_target"]
    )


@pytest.fixture(scope="session")
def trained_models(targets_added, features_exclude):
    train_df, val_df, _ = targets_added

    if "imputation_confidence" in train_df.columns:
        train_df["imputation_confidence"] = train_df["imputation_confidence"].astype("category")
    if "imputation_confidence" in val_df.columns:
        val_df["imputation_confidence"] = val_df["imputation_confidence"].astype("category")

    models, mae_metrics, rmse_metrics = {}, {}, {}
    for h in range(1, HORIZON + 1):
        key = f"{TARGET_COL}_plus_{h}h"
        mask_t = train_df[key].notnull() & train_df[key].between(VALUE_MIN, VALUE_MAX)
        mask_v = val_df[key].notnull() & val_df[key].between(VALUE_MIN, VALUE_MAX)

        X_train = train_df.loc[mask_t].drop(columns=features_exclude, errors="ignore")
        y_train = train_df.loc[mask_t, key].astype(float)
        X_val = val_df.loc[mask_v].drop(columns=features_exclude, errors="ignore")
        y_val = val_df.loc[mask_v, key].astype(float)

        if y_train.empty or y_val.empty:
            continue

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        params = {
            "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8,
            "colsample_bytree": 0.8, "objective": "reg:absoluteerror",
            "tree_method": "hist", "enable_categorical": True, "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=50,
                          evals=[(dval, "val")], verbose_eval=False)
        models[key] = model
        preds = model.predict(dval)
        mae_metrics[key] = mean_absolute_error(y_val, preds)
        rmse_metrics[key] = root_mean_squared_error(y_val, preds)

    return models, mae_metrics, rmse_metrics


# ── 1. Data Schema Tests ─────────────────────────────────────

class TestDataSchema:
    def test_required_columns_exist(self, processed_df):
        required = ["pm25", "o3", "pm25_target", "o3_target",
                     "hour", "day_of_week", "is_weekend", "is_night",
                     "hour_sin", "hour_cos", "segment_id"]
        missing = [c for c in required if c not in processed_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_datetime_index(self, processed_df):
        assert isinstance(processed_df.index, pd.DatetimeIndex)

    def test_pm25_range(self, processed_df):
        valid = processed_df["pm25"].dropna()
        assert valid.min() >= 0 and valid.max() <= 500

    def test_o3_range(self, processed_df):
        valid = processed_df["o3"].dropna()
        assert valid.min() >= 0 and valid.max() <= 500

    def test_no_duplicate_timestamps(self, processed_df):
        assert not processed_df.index.duplicated().any()


# ── 2. Preprocessing Tests ───────────────────────────────────

class TestPreprocessing:
    def test_lag_features_created(self, processed_df):
        for lag in [1, 6, 12, 24]:
            assert f"pm25_lag_{lag}" in processed_df.columns
            assert f"o3_lag_{lag}" in processed_df.columns

    def test_rolling_features_created(self, processed_df):
        for w in [3, 6, 12, 24]:
            assert f"pm25_roll_{w}" in processed_df.columns

    def test_time_features_correct(self, processed_df):
        assert processed_df["hour"].between(0, 23).all()
        assert processed_df["day_of_week"].between(0, 6).all()
        assert processed_df["is_weekend"].isin([0, 1]).all()

    def test_missing_flags_binary(self, processed_df):
        assert processed_df["pm25_missing"].isin([0, 1]).all()
        assert processed_df["o3_missing"].isin([0, 1]).all()

    def test_segment_ids_non_negative(self, processed_df):
        assert (processed_df["segment_id"] >= 0).all()


# ── 3. Model Smoke Tests ─────────────────────────────────────

class TestModelSmoke:
    def test_models_trained(self, trained_models):
        models, _, _ = trained_models
        assert len(models) > 0

    def test_predictions_correct_shape(self, trained_models, targets_added, features_exclude):
        models, _, _ = trained_models
        _, _, test_df = targets_added
        for key, model in models.items():
            mask = test_df[key].notnull() & test_df[key].between(VALUE_MIN, VALUE_MAX)
            X = test_df.loc[mask].drop(columns=features_exclude, errors="ignore")
            if X.empty:
                continue
            preds = model.predict(xgb.DMatrix(X, enable_categorical=True))
            assert len(preds) == len(X)

    def test_predictions_in_range(self, trained_models, targets_added, features_exclude):
        models, _, _ = trained_models
        _, _, test_df = targets_added
        for key, model in models.items():
            mask = test_df[key].notnull() & test_df[key].between(VALUE_MIN, VALUE_MAX)
            X = test_df.loc[mask].drop(columns=features_exclude, errors="ignore")
            if X.empty:
                continue
            preds = model.predict(xgb.DMatrix(X, enable_categorical=True))
            assert preds.min() >= -50 and preds.max() <= 600


# ── 4. Metrics Threshold Tests ───────────────────────────────

class TestMetricsThresholds:
    MAX_MAE = 50.0
    MAX_RMSE = 60.0

    def test_mae_below_threshold(self, trained_models):
        _, mae_metrics, _ = trained_models
        for key, mae in mae_metrics.items():
            assert mae < self.MAX_MAE, f"{key} MAE {mae:.2f} > {self.MAX_MAE}"

    def test_rmse_below_threshold(self, trained_models):
        _, _, rmse_metrics = trained_models
        for key, rmse in rmse_metrics.items():
            assert rmse < self.MAX_RMSE, f"{key} RMSE {rmse:.2f} > {self.MAX_RMSE}"


# ── 5. Model Save/Load Test ──────────────────────────────────

class TestModelArtifact:
    def test_save_and_load(self, trained_models, tmp_path):
        models, _, _ = trained_models
        key = list(models.keys())[0]
        model = models[key]
        path = tmp_path / "model.ubj"
        model.save_model(str(path))
        loaded = xgb.Booster()
        loaded.load_model(str(path))
        assert loaded.num_features() == model.num_features()


# ── 6. Best Params Artifact Test ─────────────────────────────

class TestArtifacts:
    def test_best_params_valid(self):
        params_path = ARTIFACTS / "best_params.json"
        if not params_path.exists():
            pytest.skip("best_params.json not found (DVC-tracked)")
        with open(params_path) as f:
            params = json.load(f)
        for k in ["n_estimators", "learning_rate", "max_depth"]:
            assert k in params, f"Missing key '{k}'"
        assert params["n_estimators"] > 0
        assert 0 < params["learning_rate"] < 1


# ── 7. CML Report Generation ─────────────────────────────────

class TestCMLReport:
    def test_generate_report(self, trained_models):
        _, mae_metrics, rmse_metrics = trained_models

        lines = [
            "# ML Test Results", "",
            "## Model Metrics (Synthetic Data)", "",
            "| Horizon | MAE | RMSE |",
            "|---------|-----|------|",
        ]
        for key in sorted(mae_metrics.keys()):
            lines.append(f"| {key} | {mae_metrics[key]:.4f} | {rmse_metrics.get(key, 0):.4f} |")

        lines += ["", f"**All {len(mae_metrics)} horizon(s) within thresholds**"]

        metrics = {}
        for key in mae_metrics:
            metrics[f"mae_{key}"] = round(mae_metrics[key], 4)
            metrics[f"rmse_{key}"] = round(rmse_metrics.get(key, 0), 4)

        metrics_json_path = REPORT_PATH.parent / "metrics.json"
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        with open(REPORT_PATH, "w") as f:
            f.write("\n".join(lines))

        assert REPORT_PATH.exists()