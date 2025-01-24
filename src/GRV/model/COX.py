from ..pre_process import coxDataLoader
import logging
import os

from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv
import torchtuples as tt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class COX:
    reader = 'coxDataLoader'

    def __init__(self, config, corpus: coxDataLoader):
        self.corpus = corpus

        # Define model and prediction paths
        base_path = os.getcwd()
        self.model_path = os.path.join(base_path, config["model_path"])
        self.prediction_path = os.path.join(base_path, config["prediction_path"])

    def define_model(self, config):
        # Define input features dynamically based on dataset
        in_features = self.corpus.x_train.shape[1]
        print(f"Input features: {in_features}")  # Debugging feature size

        # Network definition
        num_nodes = [32, 32]
        batch_norm = True
        dropout = 0.1
        net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
        self.model = CoxTime(net, tt.optim.Adam, labtrans=self.corpus.labtrans)
        logging.info("Model defined successfully.")

    def load_model(self):
        self.model.load_net(self.model_path)
        logging.info(f"Loaded model from {self.model_path}")

    def train(self):
        batch_size = 256
        lrfinder = self.model.lr_finder(self.corpus.x_train, self.corpus.y_train, batch_size, tolerance=2)
        print(f"[train] Best learning rate: {lrfinder.get_best_lr()}")
        self.model.optimizer.set_lr(0.01)

        epochs = 512
        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = True

        log = self.model.fit(
            self.corpus.x_train,
            self.corpus.y_train,
            batch_size,
            epochs,
            callbacks,
            verbose,
            val_data=self.corpus.val.repeat(10).cat()
        )

        # Use the history attribute directly
        # train_loss = log.monitors.train_loss
        #val_loss = log.history["val_loss"]

        #plt.plot(train_loss, label="Train")
        #plt.plot(val_loss, label="Validation")
        #plt.title("Training Log")
        #plt.xlabel("Epochs")
        #plt.ylabel("Partial Log-Likelihood")
        #plt.legend(loc="best")
        #plt.savefig(self.prediction_path + "_training_log.png")
        logging.info(f"[train] Model partial log-likelihood: {self.model.partial_log_likelihood(*self.corpus.val).mean()}")
        _ = self.model.compute_baseline_hazards()

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_net(self.model_path)
        logging.info(f"Model saved at {self.model_path}")

    def predict(self):
        self.surv = self.model.predict_surv_df(self.corpus.x_test)
        self.prediction_results = self.surv.T

        if hasattr(self.corpus, "df_test") and self.corpus.df_test is not None:
            self.corpus.df_test.reset_index(inplace=True)
            self.prediction_results["photo_id"] = self.corpus.df_test["photo_id"]
        else:
            raise ValueError("df_test is not set in the coxDataLoader. Ensure it is initialized during preprocessing.")

        os.makedirs(os.path.dirname(self.prediction_path), exist_ok=True)
        self.prediction_results.to_csv(self.prediction_path + ".csv", index=False)
        logging.info(f"Prediction results saved at {self.prediction_path}.csv")

    def evaluate(self):
        self.surv.iloc[:, 10:20].plot()
        plt.ylabel("S(t | x)")
        plt.xlabel("Time")
        plt.title("Survival Function for Selected Time Points")
        plt.legend(loc="best")
        plt.savefig(self.prediction_path + "_v0.png")
        plt.close()

        ev = EvalSurv(self.surv, self.corpus.durations_test, self.corpus.events_test, censor_surv="km")
        logging.info(f"Concordance index: {ev.concordance_td()}")

        time_grid = np.linspace(self.corpus.durations_test.min(), self.corpus.durations_test.max(), 100)
        logging.info(f"Integrated Brier Score: {ev.integrated_brier_score(time_grid)}")
        logging.info(f"Integrated Negative Log-Likelihood: {ev.integrated_nbll(time_grid)}")

        brier_score = ev.brier_score(time_grid)
        plt.plot(time_grid, brier_score, label="Brier Score", color="blue")
        plt.ylim(0, 0.25)
        plt.xlim(0, 300)
        plt.title("Brier Score Over Time")
        plt.xlabel("Time")
        plt.ylabel("Brier Score")
        plt.legend(loc="best")
        plt.savefig(self.prediction_path + "_v1.png")
        plt.close()

    def analysis(self, label, config):
        logging.info("Running analysis...")
        df = self.prediction_results
        df = df.sample(frac=1)
        time_list = df.columns.tolist()
        df["SA_score"] = 0

        start_time = int(config["start_time"])
        end_time = start_time + 12
        GROUP = 10

        self.corpus.filtered_data()
        col_list = self.corpus.coxData.columns.tolist()
        baseline = self.corpus.coxData[["photo_id"]]
        baseline["base_score"] = 0
        for col in col_list:
            if col != "photo_id":
                baseline["base_score"] += self.corpus.coxData[col]

        for i in range(start_time, end_time):
            if str(i) in time_list:
                df["SA_score"] += df[str(i)]

        df = pd.merge(df, baseline, on="photo_id")
        df = df.sample(frac=1)
        df["base_rank"] = df["base_score"].rank(method="first", pct=True)
        df["base_group"] = (df["base_rank"] * GROUP).astype(int)

        df["SA_rank"] = df["SA_score"].rank(method="first", pct=True)
        df["SA_group"] = (df["SA_rank"] * GROUP).astype(int)

        hour_data = label.read_all(config)
        hour_data["click"] = hour_data["click_rate"] * hour_data["counter"]

        item_info = hour_data[
            (hour_data["timelevel"] >= start_time) & (hour_data["timelevel"] < end_time)
        ].groupby("photo_id").agg({
            "click": "sum",
            "counter": "sum",
            "play_rate": "mean",
            "click_rate": "mean"
        })
        item_info["ctr"] = item_info["click"] / item_info["counter"]

        def get_rank(group):
            return pd.DataFrame((1 + np.lexsort((group["play_rate"].rank(), \
                                                 group["ctr"].rank()))) / len(group), \
                                index=group.index)

        item_info["per_rank"] = get_rank(item_info)
        item_info["per_rank"] -= item_info["per_rank"].min()

        item_info = pd.merge(item_info, df, on="photo_id", suffixes=["_itemInfo", ""])
        item_info.fillna(0, inplace=True)

        tag_list = item_info["SA_group"].unique()
        for tag in sorted(tag_list):
            tmp = item_info[item_info["SA_group"] == tag]
            logging.info(f"SA Group {tag} - Per Rank Description:\n{tmp['per_rank'].describe()}")

        sa_corr = item_info['per_rank'].corr(item_info['SA_rank'], method='pearson')
        logging.info(f"SA Correlation: {sa_corr}")

        base_corr = item_info['per_rank'].corr(item_info['base_rank'], method='pearson')
        logging.info(f"Base Correlation: {base_corr}")
