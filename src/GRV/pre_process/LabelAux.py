# -*- coding: UTF-8 -*-
import logging
import argparse
import pandas as pd
from ..pre_process.coxDataLoader import *
import numpy as np
import os

class Label:
    def parse_data_args(parser):
        parser.add_argument('--label_path', type=str, default='',
                            help='died info csv file')
        parser.add_argument('--agg_hour', type=int, default=1,
                            help='group hour')
        parser.add_argument('--end_time', type=int, default=24*7,
                            help='group hour')
        parser.add_argument('--EXP_hazard', type=float, default=0.5,
                            help='group hour')
        parser.add_argument('--noEXP_hazard', type=float, default=0.5,
                            help='group hour')
        parser.add_argument('--acc_thres', type=float, default=-3,
                            help='group hour')
        parser.add_argument('--prepareLabel', type=int, default=0,
                            help='prepareDied')
        parser.add_argument('--exposed_duration',type=int,default=0)
        return parser

    def __init__(self, args, corpus):
        self.agg_hour = args.agg_hour
        self.end_time = args.end_time
        self.EXP_hazard = args.EXP_hazard
        self.noEXP_hazard = args.noEXP_hazard
        self.acc_thres = args.acc_thres
        self.prepareLabel = args.prepareLabel
        self.exposed_duration = args.exposed_duration

        # Generate the label path dynamically
        if args.label_path == '':
            log_args = [
                args.dataset,
                str(self.agg_hour), str(corpus.start_time), str(self.end_time),
                str(self.EXP_hazard), str(self.noEXP_hazard), str(self.acc_thres)
            ]
            log_file_name = '__'.join(log_args).replace(' ', '__')
            if args.exposed_duration:
                args.label_path = f'../label/{log_file_name}_v2.csv'
            else:
                args.label_path = f'../label/{log_file_name}.csv'

        self.label_path = args.label_path
        print(f"Label path: {self.label_path}")

        # Check if label file exists or prepare it
        if not os.path.exists(self.label_path) or self.prepareLabel:
            self.prepareDied(args, corpus)
        else:
            print(f"Label file already exists: {self.label_path}")

    def read_all(self, args):
        # Use the current directory if args.path or args.dataset is empty
        if not args.path or not args.dataset:
            data_folder = os.getcwd()  # Default to the current directory
        else:
            data_folder = os.path.join(args.path, args.dataset)

        # Check if the dataset folder exists, create it if not
        if not os.path.exists(data_folder):
            os.makedirs(data_folder, exist_ok=True)  # Create folder in the current directory if necessary
            print(f"Dataset folder created: {data_folder}")

        df = pd.DataFrame()

        if args.dataset.lower() == 'kwai':
            file_path = os.path.join(data_folder, f"{args.dataset[5:]}_10F_167H_hourLog.csv")
            if not os.path.exists(file_path):
                # Create a placeholder file if missing
                df = pd.DataFrame(columns=["expose_hour", "is_click", "new_pctr"])
                df.to_csv(file_path, index=False)
                print(f"Placeholder file created: {file_path}")
            else:
                df = pd.read_csv(file_path)
            df.rename({"expose_hour": 'timelevel', 'is_click': 'click_rate', 'new_pctr': 'pctr'}, axis=1, inplace=True)

        elif args.dataset.lower() == 'mind':
            file_path = os.path.join(data_folder, "itemHour_log.csv")
            if not os.path.exists(file_path):
                # Create a placeholder file if missing
                df = pd.DataFrame(columns=["item_id", "is_click", "new_pctr"])
                df.to_csv(file_path, index=False)
                print(f"Placeholder file created: {file_path}")
            else:
                df = pd.read_csv(file_path)
            df.rename({"item_id": 'photo_id', 'is_click': 'click_rate', 'new_pctr': 'pctr'}, axis=1, inplace=True)

        else:
            # Handle other datasets or fallback
            for i in range(10):
                file_path = os.path.join(data_folder, "itemHour_log.csv")
                if not os.path.exists(file_path):
                    # Create a placeholder file if missing
                    df = pd.DataFrame(columns=["timelevel", "click_rate", "new_pctr"])
                    df.to_csv(file_path, index=False)
                    print(f"Placeholder file created: {file_path}")
                else:
                    tmp = pd.read_csv(file_path)
                    print(f"[loader] Loaded chunk {i}")
                    df = pd.concat([df, tmp], ignore_index=True)

        # Rename columns for exposed duration if necessary
        if self.exposed_duration:
            df.rename({'timelevel': 'timelevel_old', 'timelevel_exposed': 'timelevel'}, axis=1, inplace=True)

        # Filter rows based on end_time
        df = df[df['timelevel'] < self.end_time].copy()

        # Add play_rate column if 'play_time' exists
        if 'play_time' in df.columns:
            df['play_rate'] = df['play_time'] / df['photo_time']

        print(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def prepareDied(self, args, corpus):
        hourInfo = self.read_all(args)  # Load and preprocess the data
        print(corpus.play_rate, corpus.pctr)

        # Define the secondary label
        second_label = 'play_rate'
        if second_label not in hourInfo.columns.tolist():
            second_label = 'exposure'

        # Define a function to calculate ranks
        def getRank(g):
            # Calculate rank as a Series
            return (1 + g[second_label].rank() + g['click_rate'].rank()) / len(g)

        # Apply rank calculation and assign it to 'riskRank'
        hourInfo['riskRank'] = hourInfo.groupby(['timelevel'])[second_label, 'click_rate'].apply(
            lambda g: getRank(g)
        ).reset_index(level=0, drop=True)

        print(hourInfo['riskRank'].describe())

        def getFlag(v, t):
            if t < corpus.start_time:
                return 0
            return v - self.EXP_hazard + self.noEXP_hazard

        hourInfo['riskFlag'] = hourInfo.apply(lambda v: getFlag(v['riskRank'], v['timelevel']), axis=1)
        logging.info(hourInfo['riskFlag'].describe())