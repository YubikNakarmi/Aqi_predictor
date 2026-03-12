import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import json
import os
import datetime

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
        st.line_chart(df.set_index("Horizon (hours)")[["Validation MAE", "Validation RMSE"]])

except Exception as e:
    st.error(f"Error loading training metadata: {e}")
   