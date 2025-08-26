FROM python:3.9-slim-bullseye

ARG wkdir=/opt/airflow/scripts

WORKDIR ${wkdir}

RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y && apt-get autoremove -y && apt-get clean
RUN pip install --upgrade pip
RUN pip install pandas requests

COPY ./scripts/ingestion.py ${wkdir}/ingestion.py
CMD ["python", "ingestion.py "]