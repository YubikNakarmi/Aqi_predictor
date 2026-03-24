import streamlit as st
import pandas as pd
from modules import mysql_utils
import json





st.set_page_config(page_title="Air Quality Dashboard", layout="wide", page_icon=":cloud:")

@st.cache_data
def load_predictions():
    pred_df = pd.read_csv(PREDICTIONS_FILE)
    return pred_df


st.write("# Air Quality Dashboard :cloud:")
st.sidebar.success("Welcome to the Air Quality Dashboard! :wave:")
st.title("Daily Air Quality Index (AQI) Forecasts")

