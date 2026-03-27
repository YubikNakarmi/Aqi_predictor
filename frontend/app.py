import streamlit as st
import pandas as pd
from modules import mysql_utils
import json
import os
import plotly.express as px

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


PREDICTION_METADATA_FILE = os.environ.get("PREDICTION_METADATA_FILE", 
                                          os.path.join(PROJECT_ROOT, "data","metadata","prediction.json"))

PREDICTIONS_FILE = os.environ.get("PREDICTIONS_FILE", os.path.join(PROJECT_ROOT, os.path.join(PROJECT_ROOT,"data/predictions/hourly/us_paro_hourly/prod")))



st.set_page_config(page_title="Air Quality Dashboard", layout="wide", page_icon=":cloud:")



@st.cache_data
def load_prediction_metadata():
    payload = json.load(open(PREDICTION_METADATA_FILE, "r", encoding="utf-8"))
    last_pred_run = payload["runtime"]["last_run"]
    if last_pred_run["stage"] != "predict" and last_pred_run["status"]:
        return {
            "run_date": None,
            "metrics_summary": None,
        }
    run_date = last_pred_run["timestamp_utc"]
    output_file = last_pred_run["output_file"]
    return {
        "run_date": run_date,
        "output_file": output_file,
    }

@st.cache_data
def load_predictions():
    metadata = load_prediction_metadata()
    path = os.path.join(PROJECT_ROOT, metadata["output_file"])
    pred_df = pd.read_csv(path)

    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce", format="mixed")

    latest = pred_df["timestamp"].max()

    pred_df = pred_df[pred_df["timestamp"] == latest]
    return pred_df



st.write("# Air Quality Dashboard :cloud:")
st.sidebar.success("Welcome to the Air Quality Dashboard! :wave:")
st.title("Daily Air Quality Index (AQI) Forecasts")

with st.spinner("Loading prediction metadata..."):
    try:
        prediction_metadata = load_prediction_metadata()
        if not prediction_metadata["run_date"]:
            st.warning("No valid prediction run found. Please check the pipeline status.")
        else:
            st.write(f"**Last Prediction Run:** {prediction_metadata['run_date']}")
    except Exception as e:
        st.error(f"Error loading prediction metadata: {e}")

st.subheader("Station: US Paro(PAO)")
df = load_predictions().iloc[0]
date = df["timestamp"]
date_range = pd.date_range(start=date, end=date + pd.Timedelta(hours=24), freq="H")

predictions = [f"pm25_plus_{h}h_pred" for h in range(1, 25)]
st.subheader("Latest AQI Predictions")
fig = px.line(x=[i for i in range(1, 25)], y=df[predictions].values.flatten(), labels={"x": "Timestamp", "y": "Predicted AQI"}, title="24-Hour AQI Forecast")



st.plotly_chart(fig)