import nannyml
import os
import pandas as pd
from src.modules.monitoring import Monitoring, plots
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", r"data/processed/hourly/us_paro_hourly")


def main():
    monitor = Monitoring(reference_df=pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "reference.csv")),
                        analysis_df=pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "analysis.csv")),)

    estimation = monitor.estimator()