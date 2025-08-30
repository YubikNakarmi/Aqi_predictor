FROM python:3.9-slim-bullseye

WORKDIR /opt/airflow
COPY /scripts/ingestion.py ingestion.py
COPY /data /opt/airflow/data

RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y && apt-get autoremove -y && apt-get clean
RUN pip install --upgrade pip
RUN pip install pandas requests

ENV AQI_API_KEY="your_api_key_here" 
ENV WEATHER_API_KEY="your_weather_api_key_here"

