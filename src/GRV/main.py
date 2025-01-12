# -*- coding: UTF-8 -*-
import os
import sys
import pickle
import logging
import argparse
import numpy as np
import warnings

import torch


# from helpers import *
# from models.general import *
# from models.sequential import *
# from models.developing import *
from utils import utils
from model import *
from src.GRV.pre_process import *


def parse_global_args(parser):
    """
    Adds global arguments to the argument parser.
    These arguments control general execution settings like GPU usage, logging, and whether to train/load the model.

    Args:
        parser (argparse.ArgumentParser): Argument parser to which global arguments are added.
    Returns:
        argparse.ArgumentParser: Updated argument parser.
    """
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--random_seed', type=int, default=1234,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--load', type=int, default=0,
                        help='Load trained model or not')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--analysis', type=int, default=0,
                        help='Analyze prediction results')
    return parser


def main():
    """
    Main function that orchestrates the training, evaluation, and analysis of the model.
    It sets up the logging, initializes the data and model, and performs operations based on the parsed arguments.
    """
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)

    # Exclude certain arguments from logging
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'verbose', 'metric', 'test_epoch', 'buffer']
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info('GPU available: {}'.format(torch.cuda.is_available()))

    # Initialize data reader, labels, and preprocess the data
    corpus = reader_name(args)
    label = label_name(args, corpus)
    corpus.preprocess(args)

    # Initialize and define the model
    model = model_name(args, corpus)
    logging.info(model)
    model.define_model(args)

    # Load trained model if specified
    if args.load > 0:
        model.load_model()

    # Train the model if specified
    if args.train > 0:
        model.train()

    # Predict using the model
    model.predict()

    # Evaluate the model if it was trained
    if args.train > 0:
        model.evaluate()

    # Perform additional analysis if specified
    if args.analysis == 1:
        model.analysis(label, args)

    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
    return


if __name__ == '__main__':
    """
    Entry point of the script. Parses arguments, initializes components dynamically,
    and starts the main workflow.
    """
    # Parse initial arguments to determine the model
    # warnings.filterwarnings("ignore")
    # parser = argparse.ArgumentParser(description='')
    # parser = parse_global_args(parser)

    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='COX', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()

    # Dynamically determine the model, reader, and label components
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    reader_name = eval('{0}.{0}'.format(model_name.reader))
    label_name = eval('{0}.{0}'.format('Label'))
    # runner_name = eval('{0}.{0}'.format(model_name.runner))

    # Initialize argument parser and add global, reader, label, and model arguments
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = label_name.parse_data_args(parser)
    parser = model_name.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    # Generate log file names and paths dynamically
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'verbose', 'metric', 'test_epoch', 'buffer',
               'model_name', 'dataset', 'random_seed', 'prediction_path', 'gpu',
               'label_path', 'load', 'train', 'analysis', 'prepareLabel', 'prediction_dataset']
    log_args = [init_args.model_name, args.dataset, str(args.random_seed)]
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude]
    for key in keys:
        log_args.append(key + '=' + str(eval('args.' + key)))
    log_file_name = '__'.join(log_args).replace(' ', '__')

    # Set up log, model, and prediction paths
    if args.log_file == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    if args.model_path == '':
        if model_name == 'COX':
            args.model_path = 'model/{}/{}.pt'.format(init_args.model_name, log_file_name)
        else:
            args.model_path = 'model/{}/{}.h5'.format(init_args.model_name, log_file_name)
    if args.prediction_path == '':
        args.prediction_path = '../prediction/{}/{}'.format(init_args.model_name, log_file_name)
        if args.prediction_dataset != '':
            args.prediction_path = args.prediction_path.replace(args.dataset, args.prediction_dataset)

    # Set up logging configuration
    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    # Execute the main workflow
    main()
