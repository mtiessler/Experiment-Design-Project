from model.COX import COX
from pre_process.coxDataLoaderAux import coxDataLoader
import logging
import os
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)


def preprocess_itemHourLog(input_path, output_path):
    logging.info(f"Loading data from {input_path}")
    data = pd.read_csv(input_path)

    logging.info(f"Columns in the dataset: {data.columns}")
    logging.info(f"Sample data before preprocessing:\n{data.head()}")

    data['timelevel'] = pd.to_datetime(data['timelevel'], errors='coerce')

    invalid_rows = data[data['timelevel'].isna()]
    if not invalid_rows.empty:
        logging.warning(f"Dropping rows with invalid 'timelevel':\n{invalid_rows}")
    data = data.dropna(subset=['timelevel'])

    data['timelevel'] = (data['timelevel'] - data['timelevel'].min()).dt.total_seconds() // 3600
    data['timelevel'] = data['timelevel'].astype(int)

    logging.info(f"Saving preprocessed data to {output_path}")
    data.to_csv(output_path, index=False)


def main():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    dataset_path = os.path.join(base_path, 'data', 'MIND')
    dataset_name = 'train'
    raw_data_path = os.path.join(dataset_path, dataset_name, 'itemHour_log.csv')
    preprocessed_data_path = os.path.join(dataset_path, dataset_name,
                                          'itemHour_log.csv')
    model_path = os.path.join(base_path, 'models', 'cox_model.pt')
    prediction_path = os.path.join(base_path, 'predictions', 'cox.csv')
    start_time = 24
    play_rate = 1
    pctr = 0

    logging.info(f"Raw Dataset Path: {raw_data_path}")
    logging.info(f"Preprocessed Dataset Path: {preprocessed_data_path}")
    logging.info(f"Model Path: {model_path}")
    logging.info(f"Prediction Path: {prediction_path}")

    preprocess_itemHourLog(raw_data_path, preprocessed_data_path)

    class Args:
        pass

    args = Args()
    args.path = dataset_path
    args.dataset = dataset_name
    args.model_path = model_path
    args.prediction_path = prediction_path
    args.start_time = start_time
    args.play_rate = play_rate
    args.pctr = pctr

    corpus = coxDataLoader(args)
    corpus.coxData = pd.read_csv(preprocessed_data_path)
    corpus.preprocess(args)

    # Train Cox model
    cox_model = COX(args, corpus)
    cox_model.define_model(args)
    cox_model.train()

    # Predict and save survival probabilities
    cox_model.predict()
    logging.info(f"Predictions saved to {prediction_path}")


if __name__ == "__main__":
    main()
