import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class eval:
    def __init__(self, models:dict,  horizon:int, target_col:str="pm25",
                 value_min:float=0, value_max:float=500, pred_suffix:str="_pred",
                 df_test:pd.DataFrame=None,preds:pd.DataFrame=None, y_true:pd.DataFrame=None):
        self.models = models
        self.df_test = df_test
        self.target_col = target_col
        self.horizon = horizon
        self.value_min = value_min
        self.value_max = value_max
        self.pred_suffix = pred_suffix
        self.preds = preds
        self.y_true = y_true



    def _predict_horizon(self, h:int, features_exclude:list):
        key = f"{self.target_col}_plus_{h}h"
        model = self.models.get(key)
        if model is None:
            return None, None, None
        mask = self.df_test[key].notnull() & (self.df_test[key].between(self.value_min, self.value_max))
        X = self.df_test.loc[mask].drop(columns=features_exclude)
        y_true = self.df_test.loc[mask, key]
        y_pred = model.predict(xgb.DMatrix(X))
        return key, y_true, y_pred    

    def plot_feature_importance(self, model_key:str, top_n:int=20, save_path:str=None):
        model = self.models.get(model_key)
        if model is None:
            print(f"No model found for key: {model_key}")
            return

        importance = model.get_booster().get_score(importance_type='weight')
        importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title(f'Top {top_n} Feature Importances for {model_key}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
    

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
                "y_true": y_true[target_key] if y_true is not None else self.df_test[target_key],
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
            if self.y_true and self.preds is not None:
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

