import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import json
import os

TRAINING_METADATA_FILE = os.getenv(
    "TRAINING_METADATA_FILE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "metadata", "train.json"),
)


st.set_page_config(page_title="Training History", layout="centered")

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

try: 
    metadata = load_training_metadata()
    if metadata:
        st.title("Training History")
        st.write(f"**Last Training Run Date (UTC):** {metadata['run_date']}")
        st.write(f"**Horizon:** {metadata['horizon']} hours")
        st.write("### Metrics Summary:")
        st.write(metadata["metrics_summary"])

        st.write("### Metrics by Horizon:")
        for run in metadata["runs"]:
            h = run["horizon"]
            mae = run["val_mae"]
            rmse = run["val_rmse"]
            st.write(
                f"**Horizon {h}h:** MAE = {mae:.4f}, RMSE = {rmse:.4f}")

except Exception as e:
    st.error(f"Error loading training metadata: {e}")
   