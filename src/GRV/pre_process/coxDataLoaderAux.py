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
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data directory.')
        parser.add_argument('--dataset', type=str, default='my_dataset',
                            help='Name of the dataset (folder containing itemHourLog.csv).')
        parser.add_argument('--play_rate', type=int, default=1,
                            help='Include play_rate feature (1 or 0).')
        parser.add_argument('--pctr', type=int, default=0,
                            help='Include pctr feature (1 or 0).')
        parser.add_argument('--start_time', type=int, default=24,
                            help='Start time for filtering data.')
        return parser

    def __init__(self, args):
        data_folder = os.path.join(args.path, args.dataset, "itemHourLog_preprocessed.csv")  # Use preprocessed file
        self.coxData = pd.read_csv(data_folder)
        print(f"Loaded {len(self.coxData)} rows from {data_folder}")

        required_columns = ['photo_id', 'timelevel', 'click_rate', 'play_rate']
        for col in required_columns:
            if col not in self.coxData.columns:
                raise ValueError(f"Missing required column: {col} in itemHourLog_preprocessed.csv")

        # Convert 'timelevel' to datetime
        self.coxData['timelevel'] = pd.to_datetime(self.coxData['timelevel'], errors='coerce')

        # Drop rows where 'timelevel' could not be converted
        self.coxData = self.coxData.dropna(subset=['timelevel'])

        # Convert 'timelevel' to numeric (hours since the first timestamp in the dataset)
        min_time = self.coxData['timelevel'].min()  # Reference point
        self.coxData['timelevel'] = (self.coxData['timelevel'] - min_time).dt.total_seconds() // 3600

        # Ensure 'timelevel' is integer
        self.coxData['timelevel'] = self.coxData['timelevel'].astype(int)

        # Filter rows based on args.start_time
        self.coxData = self.coxData[self.coxData['timelevel'] >= args.start_time]
        print(f"Filtered data: {len(self.coxData)} rows remaining.")

        self.labtrans = None
        self.play_rate = args.play_rate
        self.pctr = args.pctr
        self.start_time = args.start_time
    def load_data(self, args):
        df = self.coxData
        df['died'] = (df['timelevel'] >= df['timelevel'].max() - 24).astype(int)  # Example died logic
        df_train = df.sample(frac=0.7, random_state=42)
        df_test = df.drop(df_train.index)
        df_val = df_train.sample(frac=0.2, random_state=42)
        df_train = df_train.drop(df_val.index)
        caredList = ['died', 'timelevel', 'photo_id', 'click_rate', 'play_rate']
        if args.pctr:
            caredList.append('new_pctr')
        return df_train, df_val, df_test, caredList

    def preprocess(self, args):
        df_train, df_val, df_test, cared = self.load_data(args)
        print(len(df_train), len(df_val), len(df_test))

        ignore_length = 3
        cols_standardize = cared[ignore_length:]

        # Ensure the mapper handles single-column features correctly
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        x_mapper = DataFrameMapper(standardize, df_out=False)  # Set `df_out=False` for array output

        # Transform training, validation, and test data
        self.x_train = x_mapper.fit_transform(df_train).astype('float32')
        if len(df_val) > 0:
            x_val = x_mapper.transform(df_val).astype('float32')
        self.x_test = x_mapper.transform(df_test).astype('float32')
        self.df_test = df_test

        # Prepare labels for the Cox model
        self.labtrans = CoxTime.label_transform()
        get_target = lambda df: (df['timelevel'].values, df['died'].values)
        self.y_train = self.labtrans.fit_transform(*get_target(df_train))
        y_val = self.labtrans.transform(*get_target(df_val))
        self.val = tt.tuplefy(x_val, y_val) if len(df_val) > 0 else None
        self.durations_test, self.events_test = get_target(df_test)