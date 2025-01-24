# -*- coding: UTF-8 -*-

import os
import sys
import logging
import pandas as pd
import numpy as np
import torch

from src.GRV.model.COX import COX
from src.GRV.pre_process import Label
from src.GRV.pre_process.coxDataLoader import coxDataLoader
from utils import utils



def load_config_from_csv(file_path):
    try:
        config_df = pd.read_csv(file_path, dtype=str)
        config = {}
        for _, row in config_df.iterrows():
            key = row['key']
            value = row['value']
            config[key] = value if pd.notna(value) else ''
        return config
    except Exception as e:
        raise ValueError(f"Error reading configuration file: {e}")


def main(config_file):
    try:
        config = load_config_from_csv(config_file)
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Logging configuration
    log_args = [config['model_name'], config['dataset'], config['random_seed']]
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'verbose', 'metric', 'test_epoch', 'buffer',
               'model_name', 'dataset', 'random_seed', 'prediction_path', 'gpu',
               'label_path', 'load', 'train', 'analysis', 'prepareLabel', 'prediction_dataset']

    keys = [k for k in config.keys() if k not in exclude]
    for key in keys:
        log_args.append(key + '=' + str(config[key]))

    log_file_name = '__'.join(log_args).replace(' ', '__')
    if not config.get('log_file'):
        config['log_file'] = f'../log/{config["model_name"]}/{log_file_name}.txt'
    if not config.get('model_path'):
        if config['model_name'] == 'COX':
            config['model_path'] = f'model/{config["model_name"]}/{log_file_name}.pt'
        else:
            config['model_path'] = f'model/{config["model_name"]}/{log_file_name}.h5'
    if not config.get('prediction_path'):
        config['prediction_path'] = f'../prediction/{config["model_name"]}/{log_file_name}'
        if config.get('prediction_dataset'):
            config['prediction_path'] = config['prediction_path'].replace(
                config['dataset'], config['prediction_dataset'])

    utils.check_dir(config['log_file'])
    logging.basicConfig(filename=config['log_file'], level=int(config['verbose']))
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Loaded Configuration: {config}")

    # Random seed
    np.random.seed(int(config['random_seed']))
    torch.manual_seed(int(config['random_seed']))
    torch.cuda.manual_seed(int(config['random_seed']))
    torch.backends.cudnn.deterministic = True

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    logging.info('GPU available: {}'.format(torch.cuda.is_available()))

    # Initialize components
    corpus = coxDataLoader(config)
    model = COX(config, corpus)
    label = Label
    corpus.preprocess(config)

    logging.info(model)

    # Define and train the model
    model.define_model(config)
    if int(config['load']) > 0:
        model.load_model()
    if int(config['train']) > 0:
        model.train()
    model.predict()
    if int(config['train']) > 0:
        model.evaluate()
    if int(config['analysis']) == 1:
        model.analysis(label, config)

    logging.info('-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


if __name__ == '__main__':
    # Specify the path to the configuration file
    CONFIG_FILE = "config.csv"
    main(CONFIG_FILE)
