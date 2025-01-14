# -*- coding: UTF-8 -*-
import logging
import argparse
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from pycox.models import CoxTime


class coxDataLoader:
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='',
                            help='Input data directory.')
        parser.add_argument('--dataset', type=str, default='',
                            help='Name of the dataset (folder containing itemHour_log.csv).')
        parser.add_argument('--play_rate', type=int, default=1,
                            help='Include play_rate feature (1 or 0).')
        parser.add_argument('--pctr', type=int, default=0,
                            help='Include pctr feature (1 or 0).')
        parser.add_argument('--start_time', type=int, default=24,
                            help='Start time for filtering data.')
        return parser

    def __init__(self, args):
        # Default path construction
        data_folder = os.path.join(args.path, args.dataset, "itemHour_log.csv")

        # Check if the file exists at the constructed path
        if not os.path.exists(data_folder):
            # Fall back to `src/GRV/pre_process/` if the file isn't found in the default location
            alt_path = os.path.join(os.path.dirname(__file__), "itemHour_log.csv")
            if os.path.exists(alt_path):
                data_folder = alt_path
            else:
                raise FileNotFoundError(
                    f"Data file not found. Checked paths:\n- {data_folder}\n- {alt_path}"
                )

        # Load the data
        self.coxData = pd.read_csv(data_folder)
        print(f"Loaded {len(self.coxData)} rows from {data_folder}")

        # Clean and standardize column names
        self.coxData.columns = self.coxData.columns.str.strip().str.lower()
        print(f"Columns in dataset: {self.coxData.columns.tolist()}")

        # Verify the required columns exist
        required_columns = ['photo_id', 'timelevel', 'click_rate', 'play_rate']
        for col in required_columns:
            if col not in self.coxData.columns:
                raise ValueError(f"Missing required column: {col} in itemHour_log.csv")

        # Convert 'timelevel' to datetime
        self.coxData['timelevel'] = pd.to_datetime(self.coxData['timelevel'], errors='coerce')
        self.coxData = self.coxData.dropna(subset=['timelevel'])

        # Convert 'timelevel' to numeric (hours since the first timestamp in the dataset)
        min_time = self.coxData['timelevel'].min()
        self.coxData['timelevel'] = (self.coxData['timelevel'] - min_time).dt.total_seconds() // 3600
        self.coxData['timelevel'] = self.coxData['timelevel'].astype(int)

        # Debug range of timelevel
        print(f"Min timelevel: {self.coxData['timelevel'].min()}, Max timelevel: {self.coxData['timelevel'].max()}")

        # Adjust start_time if it's outside the range of timelevel
        if self.coxData['timelevel'].max() < args.start_time:
            args.start_time = self.coxData['timelevel'].min()
            print(f"Adjusted start_time to {args.start_time} based on dataset.")

        # Filter rows based on args.start_time
        self.coxData = self.coxData[self.coxData['timelevel'] >= args.start_time]
        if self.coxData.empty:
            raise ValueError("Filtered dataset is empty. Check your input data and filtering conditions.")
        print(f"Filtered data: {len(self.coxData)} rows remaining.")

        # Apply preprocessing
        self.labtrans = None
        self.play_rate = args.play_rate
        self.pctr = args.pctr
        self.start_time = args.start_time

        self.preprocess(args)

    def load_data(self, args):
        df = self.coxData
        if df.empty:
            raise ValueError("Dataset is empty after filtering.")

        df['died'] = (df['timelevel'] >= df['timelevel'].max() - 24).astype(int)  # Example died logic
        df_train = df.sample(frac=0.7, random_state=42)
        df_test = df.drop(df_train.index)
        df_val = df_train.sample(frac=0.2, random_state=42)
        df_train = df_train.drop(df_val.index)

        # Check if splits are empty
        if df_train.empty or df_val.empty or df_test.empty:
            raise ValueError("One of the data splits (train/val/test) is empty after filtering.")

        caredList = ['died', 'timelevel', 'photo_id', 'click_rate', 'play_rate']
        if args.pctr:
            caredList.append('new_pctr')
        return df_train, df_val, df_test, caredList

    def preprocess(self, args):
        # Load the split data
        df_train, df_val, df_test, cared = self.load_data(args)
        print(f"Train size: {len(df_train)}, Validation size: {len(df_val)}, Test size: {len(df_test)}")

        # Features to standardize
        ignore_length = 3
        cols_standardize = cared[ignore_length:]

        # Ensure the mapper handles single-column features correctly
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        x_mapper = DataFrameMapper(standardize, df_out=False)  # Set `df_out=False` for array output

        # Transform training, validation, and test data
        self.x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_val = x_mapper.transform(df_val).astype('float32') if len(df_val) > 0 else None
        self.x_test = x_mapper.transform(df_test).astype('float32')
        self.df_test = df_test

        # Prepare labels for the Cox model
        self.labtrans = CoxTime.label_transform()
        get_target = lambda df: (df['timelevel'].values, df['died'].values)
        self.y_train = self.labtrans.fit_transform(*get_target(df_train))
        y_val = self.labtrans.transform(*get_target(df_val)) if len(df_val) > 0 else None
        self.val = tt.tuplefy(x_val, y_val) if y_val is not None else None
        self.durations_test, self.events_test = get_target(df_test)

        print("Preprocessing complete.")
