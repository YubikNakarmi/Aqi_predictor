import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Daily AQI Prediction", page_icon=":smiley:", layout="wide")
st.title("Daily AQI Prediction")

st.sidebar.header("Inputs")
city = st.sidebar.selectbox("City", ["Kathmandu", "Pokhara", "Lalitpur"])
hour = st.sidebar.slider("Forecast horizon (hours)", 1, 24, 6)

st.subheader("Prediction")
predicted_aqi = np.random.randint(40, 180)
st.metric("Predicted AQI", predicted_aqi)

st.subheader("Trend")
df = pd.DataFrame({"hour": range(hour), "aqi": np.random.randint(30, 200, hour)})
st.line_chart(df.set_index("hour"))