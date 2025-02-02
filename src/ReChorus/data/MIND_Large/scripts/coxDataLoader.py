import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from pycox.models import CoxTime


class coxDataLoader:
    def __init__(self, config):
        self.config = config
        self.item_hour_log_file = os.path.join(config["path"], config["item_hour_log_file"])
        self.cox_file = os.path.join(config["path"], "cox.csv")

        # Load ItemHourLog.csv
        if not os.path.exists(self.item_hour_log_file):
            raise ValueError(f"`ItemHourLog.csv` not found at {self.item_hour_log_file}")

        self.item_hour_log = pd.read_csv(self.item_hour_log_file)

        # Split the dataset into train, dev, and test
        self.split_dataset()

        # Preprocess the data
        self.preprocess()

    def split_dataset(self):
        """
        Splits the data into train (70%), validation (15%), and test (15%) sets.
        """
        print("Splitting dataset into train (70%), dev (15%), and test (15%)...")
        np.random.seed(int(self.config["random_seed"]))  # Set random seed for reproducibility

        # Shuffle the dataset
        shuffled = self.item_hour_log.sample(frac=1).reset_index(drop=True)

        # Calculate split indices
        n = len(shuffled)
        train_end = int(0.7 * n)
        val_end = train_end + int(0.15 * n)

        # Assign dataset labels
        shuffled.loc[:train_end - 1, 'dataset'] = 'train'
        shuffled.loc[train_end:val_end - 1, 'dataset'] = 'dev'
        shuffled.loc[val_end:, 'dataset'] = 'test'

        # Update the item_hour_log with split labels
        self.item_hour_log = shuffled

        # Split data into train, dev, and test
        self.train_data = self.item_hour_log[self.item_hour_log['dataset'] == 'train']
        self.val_data = self.item_hour_log[self.item_hour_log['dataset'] == 'dev']
        self.test_data = self.item_hour_log[self.item_hour_log['dataset'] == 'test']

    def preprocess(self):
        """
        Preprocesses the loaded ItemHourLog.csv for use in the Cox model.
        """
        print(f"Preprocessing data: {len(self.train_data)} train, "
              f"{len(self.val_data)} val, {len(self.test_data)} test")

        # Convert `timelevel` to numerical format
        def convert_timelevel(df):
            df = df.copy()  # Avoid SettingWithCopyWarning
            if not pd.api.types.is_numeric_dtype(df["timelevel"]):
                df["timelevel"] = pd.to_datetime(df["timelevel"])
                df["timelevel"] = (df["timelevel"] - df["timelevel"].min()).dt.total_seconds() / 3600
            return df

        self.train_data = convert_timelevel(self.train_data)
        self.val_data = convert_timelevel(self.val_data)
        self.test_data = convert_timelevel(self.test_data)

        ignore_columns = ["item_id", "dataset"]
        feature_columns = [col for col in self.train_data.columns if col not in ignore_columns]

        # Standardization pipeline
        standardize = [([col], StandardScaler()) for col in feature_columns if col not in ["timelevel", "died"]]
        x_mapper = DataFrameMapper(standardize)

        # Transform features
        self.x_train = x_mapper.fit_transform(self.train_data).astype("float32")
        self.x_val = x_mapper.transform(self.val_data).astype("float32")
        self.x_test = x_mapper.transform(self.test_data).astype("float32")

        # Define labels for survival analysis
        self.labtrans = CoxTime.label_transform()
        self.y_train = self.labtrans.fit_transform(
            self.train_data["timelevel"].values, self.train_data["died"].values
        )
        self.y_val = self.labtrans.transform(
            self.val_data["timelevel"].values, self.val_data["died"].values
        )
        self.durations_test = self.test_data["timelevel"].values
        self.events_test = self.test_data["died"].values

        # Prepare validation set for model evaluation
        self.val = tt.tuplefy(self.x_val, self.y_val)

        # Set df_test for COX model
        self.df_test = self.test_data.copy()  # Ensure test_data is accessible