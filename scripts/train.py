from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import optuna
import mlflow



def objective(trial, X_train, y_train, X_valid, y_valid):
    pass


def train():
    xgb_model = xgb.XGBRegressor()


def main():
    global