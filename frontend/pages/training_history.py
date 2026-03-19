import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import json
import os
import datetime
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

TRAINING_METADATA_FILE = os.getenv(
    "TRAINING_METADATA_FILE",
    os.path.join(PROJECT_ROOT, "data", "metadata", "train.json"),
)
PREDICTIONS_FILE = os.getenv(
    "PREDICTIONS_DIR",
    os.path.join(PROJECT_ROOT, "data", "predictions", "hourly", "us_paro_hourly", "test", "hourly_pm25_predictions.csv"),
)
EVAL_METADATA_FILE = os.getenv(
    "EVAL_METADATA_FILE",
    os.path.join(PROJECT_ROOT, "data", "metadata", "eval.json"),
)

TUNING_METADATA_FILE = os.getenv(
    "TUNING_METADATA_FILE",
    os.path.join(PROJECT_ROOT, "data", "metadata", "tuning.json"),
)


ARTIFACT_PATH = os.getenv("ARTIFACT_PATH", os.path.join(PROJECT_ROOT, "data", "artifacts","hourly","us_paro","shap"))


st.set_page_config(page_title="Training & Evaluation History", layout="wide")
tab1, tab2, tab3 = st.tabs(["Training History", "Evaluation History","Tuning History"])

@st.cache_data
def load_training_metadata():
    payload = json.load(open(TRAINING_METADATA_FILE, "r", encoding="utf-8"))
    last_training_run = payload["runtime"]["last_run"]
    run_date = last_training_run["timestamp_utc"]
    horizon = last_training_run["horizon"]
    metrics_summary = last_training_run["metrics_summary"]
    runs = last_training_run["runs"]
    return {
        "run_date": run_date,
        "horizon": horizon,
        "metrics_summary": metrics_summary,
        "runs": runs,
    }

@st.cache_data
def load_eval_metadata():
    payload = json.load(open(EVAL_METADATA_FILE, "r", encoding="utf-8"))
    last_eval_run = payload["runtime"]["last_run"]
    run_date = last_eval_run["timestamp_utc"]
    horizon = last_eval_run["horizon"]
    metrics_summary = last_eval_run["metrics_summary"]
    runs = last_eval_run["metrics_by_horizon"]
    return {
        "run_date": run_date,
        "metrics_summary": metrics_summary,
        "runs": runs,
    }

@st.cache_data
def load_eval_artifacts(h:int):
    def _latest_match(patterns: list[str]):#find the latest match for /shap directory
        matches = []
        for pattern in patterns:
            matches.extend(glob.glob(os.path.join(ARTIFACT_PATH, pattern)))
        if not matches:
            return None
        return max(matches, key=os.path.getmtime)

    bar = _latest_match([
        f"eval_*_shap_bar_{h}h.png",
        f"eval_*_shap_bar{h}.png",
    ])
    beeswarm = _latest_match([
        f"eval_*_shap_beeswarm_{h}h.png",
    ])
    waterfall = _latest_match([
        f"eval_*_shap_waterfall_{h}h.png",
    ])
    return bar, beeswarm, waterfall



@st.cache_data
def load_mlflow__EvalData():
    pass


def render_shap_image(slot, image_path: str | None, caption: str, missing_message: str):
    slot.empty()
    with slot.container():
        if image_path and os.path.exists(image_path):
            st.image(image_path, caption=caption, use_container_width=True)
        else:
            st.warning(missing_message)

with tab1:
    try: 
        metadata = load_training_metadata()
        if metadata:
            st.title("Training History")
            run_date = datetime.datetime.fromisoformat(metadata['run_date'])
            st.write(f"**Last Training Run Date (UTC):** {run_date}")
            st.write(f"**Horizon:** {metadata['horizon']} hours")
            st.write("### Metrics Summary:")
            st.write(metadata["metrics_summary"])
            h = [run["horizon"] for run in metadata["runs"]]
            mae = [run["val_mae"] for run in metadata["runs"]]
            rmse = [run["val_rmse"] for run in metadata["runs"]]
            st.write("### Metrics by Horizon:")
            df = pd.DataFrame({
                "Horizon (hours)": h,
                "Validation MAE": mae,
                "Validation RMSE": rmse
            })
            st.dataframe(df)
            st.bar_chart(df.set_index("Horizon (hours)")[["Validation MAE", "Validation RMSE"]],stack=False)

    except Exception as e:
        st.error(f"Error loading training metadata: {e}")

with tab2:
    try:
        metadata = load_eval_metadata()
        if metadata and metadata.get("runs"):
            st.title("Evaluation History")
            run_date = datetime.datetime.fromisoformat(metadata["run_date"])
            st.write(f"**Last Evaluation Run Date (UTC):** {run_date}")
            st.write("### Metrics Summary:")
            st.write(metadata["metrics_summary"])

            runs = sorted(metadata["runs"], key=lambda item: item["horizon"])
            horizons = [item["horizon"] for item in runs]

            selected_horizon = st.slider(
                "Select forecast horizon (hours)",
                min_value=min(horizons),
                max_value=max(horizons),
                value=min(horizons),
                key="eval_horizon_slider",
            )

            selected_run = next(
                (item for item in runs if item["horizon"] == selected_horizon),
                None,
            )#iterate over json metadata horizons

            if selected_run:
                st.write(f"**Selected Horizon:** {selected_horizon}h")
                st.write(f"**MAE:** {selected_run.get('mae')}")
                st.write(f"**RMSE:** {selected_run.get('rmse')}")

            bar, beeswarm, waterfall = load_eval_artifacts(selected_horizon)

            st.write("### SHAP Plots")
            col1, col2, col3 = st.columns(3)
            bar_slot = col1.empty()
            beeswarm_slot = col2.empty()
            waterfall_slot = col3.empty()

            render_shap_image(
                bar_slot,
                bar,
                f"SHAP Bar Plot ({selected_horizon}h)",
                f"No SHAP bar plot found for {selected_horizon}h",
            )
            render_shap_image(
                beeswarm_slot,
                beeswarm,
                f"SHAP Beeswarm Plot ({selected_horizon}h)",
                f"No SHAP beeswarm plot found for {selected_horizon}h",
            )
            render_shap_image(
                waterfall_slot,
                waterfall,
                f"SHAP Waterfall Plot ({selected_horizon}h)",
                f"No SHAP waterfall plot found for {selected_horizon}h",
            )
        else:
            st.info("No evaluation metadata found.")
    except Exception as e:
        st.error(f"Error loading evaluation metadata: {e}")

with tab3: