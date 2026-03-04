import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import shap

class eval_plots:
    def __init__(self, models:dict,  horizon:int, target_col:str="pm25",
                 value_min:float=0, value_max:float=500, pred_suffix:str="_pred",
                 df_test:pd.DataFrame=None,preds:pd.DataFrame=None, y_true:pd.DataFrame=None, 
                 default_mlflow_artifact_dir:str="plots", default_save_dir:str="plots"):
        
        self.models = models
        self.df_test = df_test
        self.target_col = target_col
        self.horizon = horizon
        self.value_min = value_min
        self.value_max = value_max
        self.pred_suffix = pred_suffix
        self.preds = preds
        self.y_true = y_true
        self.default_mlflow_artifact_dir = default_mlflow_artifact_dir
        self.default_save_dir = default_save_dir

    def plot_predictions(self, samples:int=100, save_dir:str=None):
        """Plot true vs predicted values for each horizon using pre-computed predictions.

        Expects prediction columns named like f"{target_col}_plus_{h}h{pred_suffix}".
        Falls back to showing only horizons present in the predictions dataframe.
        """

        if self.preds is None or self.preds.empty:
            print("No predictions provided; supply a dataframe with pre-computed predictions.")
            return

        for h in range(1, self.horizon + 1):
            target_key = f"{self.target_col}_plus_{h}h"
            pred_key = f"{target_key}{self.pred_suffix}"

            if (self.df_test is not None and target_key not in self.df_test.columns) or pred_key not in self.preds.columns:
                continue

            # Align by index; assume preds rows correspond to df_test rows
            df_pair = pd.DataFrame({
                "y_true": self.y_true[target_key] if self.y_true is not None else self.df_test[target_key],
                "y_pred": self.preds[pred_key]
            }).dropna()

            df_pair = df_pair[(df_pair["y_true"] >= self.value_min) & (df_pair["y_true"] <= self.value_max)]
            if df_pair.empty:
                continue

            if len(df_pair) > samples:
                df_pair = df_pair.sample(n=samples, random_state=42)

            plt.figure(figsize=(10, 6))
            plt.plot(df_pair["y_true"].values, label="True", marker="o")
            plt.plot(df_pair["y_pred"].values, label="Predicted", marker="x")
            plt.title(f"Predictions vs True Values for {target_key}")
            plt.xlabel("Sample Index")
            plt.ylabel("AQI Value")
            plt.legend()
            plt.tight_layout()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"pred_vs_true_{target_key}.png")
                plt.savefig(out_path)
                print(f"Prediction plot saved to {out_path}")
                plt.close()
            else:
                plt.show()
                plt.close()

    def plot_horizon_errors(self, features_exclude:list, metric="mae", save_path:str=None):
        errs = []
        for h in range(1, self.horizon+1):
            if self.y_true is not None and self.preds is not None:
                key = f"{self.target_col}_plus_{h}h"
                y_true = self.y_true[key]
                y_pred = self.preds[f"{key}{self.pred_suffix}"]
            else:
             key, y_true, y_pred = self._predict_horizon(h, features_exclude)


            if key is None:
                continue
            if metric == "rmse":
                val = root_mean_squared_error(y_true, y_pred)
            else:
                val = mean_absolute_error(y_true, y_pred)
            errs.append((key, val))
        df = pd.DataFrame(errs, columns=["horizon","error"])
        plt.figure(figsize=(10,4))
        sns.barplot(x="horizon", y="error", data=df, palette="crest")
        plt.title(f"{metric.upper()} by Horizon"); plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(save_path) if save_path else plt.show()


class shap_plots:
    def __init__(self, model:xgb.Booster, feature_names:list, save_dir:str="plots"):
        self.model = model
        self.feature_names = feature_names
        self.save_dir = save_dir

    def plot_shap_summary(self, X:pd.DataFrame, max_display:int=20):
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X)

        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, features=X, feature_names=self.feature_names,
                          max_display=max_display, show=False)
        plt.tight_layout()
        out_path = os.path.join(self.save_dir, "shap_summary_plot.png")
        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(out_path)
        print(f"SHAP summary plot saved to {out_path}")
        plt.close()
