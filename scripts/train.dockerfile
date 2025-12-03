FROM python:3.9-slim-bullseye

WORKDIR /opt/airflow
COPY /scripts/train.py train.py
COPY /data /opt/airflow/data

RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y && apt-get autoremove -y && apt-get clean
RUN pip install --upgrade pip
RUN pip install sklearn pandas xgboost click



