import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from pycox.models import CoxTime


class coxDataLoader:
    def __init__(self, config):
        self.config = config
        self.train_file = os.path.join(config["path"], config["train_file"])
        self.val_file = os.path.join(config["path"], config["val_file"])
        self.test_file = os.path.join(config["path"], config["test_file"])
        self.cox_file = os.path.join(config["path"], "cox.csv")

        # Generate cox.csv if it doesn't exist
        if not os.path.exists(self.cox_file):
            self.generate_cox_csv()

        # Load pre-split datasets
        self.train_data = pd.read_csv(self.train_file)
        self.val_data = pd.read_csv(self.val_file)
        self.test_data = pd.read_csv(self.test_file)

        # Load the generated cox.csv
        self.cox_data = pd.read_csv(self.cox_file)

        # Ensure necessary columns are present in datasets
        expected_columns = [
            "photo_id", "timelevel", "exposure", "clicks",
            "click_rate", "photo_time", "play_time", "play_rate", "new_pctr", "died"
        ]
        for col in expected_columns:
            for dataset in [self.train_data, self.val_data, self.test_data]:
                if col not in dataset.columns:
                    raise ValueError(f"Missing required column '{col}' in one of the datasets")

        # Convert timelevel to numerical values if needed
        for dataset in [self.train_data, self.val_data, self.test_data]:
            if not pd.api.types.is_numeric_dtype(dataset["timelevel"]):
                dataset["timelevel"] = pd.to_datetime(dataset["timelevel"])
                dataset["timelevel"] = (
                    dataset["timelevel"] - dataset["timelevel"].min()
                ).dt.total_seconds() / 3600  # Convert to hours

    def generate_cox_csv(self):
        """
        Combines train, validation, and test datasets into a single cox.csv
        with aggregated features.
        """
        print("Generating `cox.csv`...")

        # Load datasets
        train_data = pd.read_csv(self.train_file)
        val_data = pd.read_csv(self.val_file)
        test_data = pd.read_csv(self.test_file)

        # Combine datasets
        combined_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

        # Aggregate data by `photo_id` and `timelevel`
        cox_data = combined_data.groupby(["photo_id", "timelevel"]).agg(
            exposure=("exposure", "sum"),
            clicks=("clicks", "sum"),
            click_rate=("click_rate", "mean"),
            photo_time=("photo_time", "mean"),
            play_time=("play_time", "mean"),
            play_rate=("play_rate", "mean"),
            new_pctr=("new_pctr", "mean"),
            died=("died", "mean")  # Ensure the `died` column is aggregated correctly
        ).reset_index()

        # Save `cox.csv`
        cox_data.to_csv(self.cox_file, index=False)
        print(f"`cox.csv` has been successfully generated at {self.cox_file}")

    def preprocess(self, config):
        """
        Preprocesses the loaded data for use in the Cox model.
        """
        df_train = self.train_data
        df_val = self.val_data
        df_test = self.test_data

        print(len(df_train), len(df_val), len(df_test))
        ignore_length = 3
        cols_standardize = df_train.columns[ignore_length:]

        # Standardization pipeline
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        x_mapper = DataFrameMapper(standardize)

        # Preprocess training data
        self.x_train = x_mapper.fit_transform(df_train).astype("float32")
        self.labtrans = CoxTime.label_transform()  # Define the label transformation

        # Process labels for training data
        self.y_train = self.labtrans.fit_transform(*self.get_target(df_train))

        if len(df_val) > 0:
            self.x_val = x_mapper.transform(df_val).astype("float32")
            self.y_val = self.labtrans.transform(*self.get_target(df_val))
            self.val = tt.tuplefy(self.x_val, self.y_val)
        else:
            self.x_val, self.y_val, self.val = None, None, None

        # Preprocess test data
        self.x_test = x_mapper.transform(df_test).astype("float32")
        self.df_test = df_test  # <-- Assign df_test attribute
        self.durations_test, self.events_test = self.get_target(df_test)

    def get_target(self, df):
        # Extract durations (timelevel) and events (died)
        return df["timelevel"].values, df["died"].values
