import logging
import numpy as np
import torch

from src.GRV.model.COX import COX
from src.GRV.pre_process.coxDataLoader import coxDataLoader

def load_config_from_csv(file_path):
    """
    Load configuration from a CSV file.
    """
    import pandas as pd
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
    """
    Main function to execute the Cox model pipeline.
    """
    # Load configuration
    config = load_config_from_csv(config_file)

    # Ensure necessary paths in config
    config.setdefault("model_path", "model/cox_model.pt")
    config.setdefault("prediction_path", "predictions/cox")

    # Logging setup
    log_file = config.get("log_file", "cox_log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info("Configuration loaded successfully.")

    # Set random seeds for reproducibility
    np.random.seed(int(config.get("random_seed", 42)))
    torch.manual_seed(int(config.get("random_seed", 42)))
    torch.cuda.manual_seed(int(config.get("random_seed", 42)))

    # Load data and initialize the coxDataLoader
    logging.info("Initializing coxDataLoader...")
    corpus = coxDataLoader(config)

    # Initialize and train the Cox model
    logging.info("Initializing COX model...")
    model = COX(config, corpus)

    # Define the model structure
    logging.info("Defining the Cox model...")
    model.define_model(config)

    # Train the model
    logging.info("Training the Cox model...")
    model.train()

    # Predict survival functions
    logging.info("Predicting survival probabilities...")
    model.predict()

    # Evaluate the model
    logging.info("Evaluating the Cox model...")
    model.evaluate()

    logging.info("Cox model pipeline completed successfully.")

if __name__ == "__main__":
    CONFIG_FILE = "config.csv"  # Specify the configuration file path
    main(CONFIG_FILE)
