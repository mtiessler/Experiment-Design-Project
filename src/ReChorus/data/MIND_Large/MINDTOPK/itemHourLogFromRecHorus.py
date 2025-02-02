import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def generate_item_hour_log(topk_path, output_path, datasets=["train", "dev", "test"]):
    """
    Generates ItemHourLog.csv from preprocessed interaction files in MINDTOPK.

    Parameters:
    - topk_path: str, path to the MINDTOPK directory containing train.csv, dev.csv, test.csv.
    - output_path: str, path where the final ItemHourLog.csv will be saved.
    - datasets: list, datasets to process (default: ["train", "dev", "test"]).

    Returns:
    - None
    """

    all_logs = []  # Store logs from all datasets

    for dataset in datasets:
        print(f"Processing {dataset} dataset...")
        file_path = os.path.join(topk_path, f"{dataset}.csv")

        # Load the dataset
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        data = pd.read_csv(file_path, sep="\t")
        data["time"] = pd.to_datetime(data["time"], unit="s", errors="coerce")
        data["timelevel"] = data["time"].dt.floor("h")  # Group by hourly levels

        # Aggregate data to compute exposures
        item_hour_log = data.groupby(["item_id", "timelevel"]).agg(
            exposure=("item_id", "count"),  # Number of exposures
        ).reset_index()

        # Simulate clicks (as binary random values for this example, adjust as needed)
        np.random.seed(42)
        item_hour_log["clicks"] = np.random.binomial(item_hour_log["exposure"], 0.05)

        # Compute click rate
        item_hour_log["click_rate"] = item_hour_log["clicks"] / item_hour_log["exposure"]
        item_hour_log["click_rate"].fillna(0, inplace=True)  # Handle division by zero

        # Simulate photo_time and play_time
        item_hour_log["photo_time"] = item_hour_log["exposure"] * np.random.uniform(60, 300)  # Total viewing time
        item_hour_log["play_time"] = item_hour_log["clicks"] * np.random.uniform(15, 60)  # Time spent on clicks

        # Compute play_rate
        item_hour_log["play_rate"] = item_hour_log["play_time"] / item_hour_log["photo_time"]
        item_hour_log["play_rate"] = item_hour_log["play_rate"].clip(upper=1)  # Ensure play_rate <= 1

        # Compute new_pctr as historical click rate
        item_hour_log["new_pctr"] = item_hour_log["click_rate"]

        # Add 'died' column based on a threshold for click rate
        click_rate_threshold = 0.1
        item_hour_log["died"] = (item_hour_log["click_rate"] < click_rate_threshold).astype(int)

        # Add dataset column for traceability
        item_hour_log["dataset"] = dataset

        all_logs.append(item_hour_log)

    # Combine logs from all datasets
    if all_logs:
        final_item_hour_log = pd.concat(all_logs, axis=0).reset_index(drop=True)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "ItemHourLog.csv")
        final_item_hour_log.to_csv(output_file, index=False)
        print(f"ItemHourLog.csv saved to {output_file}")
    else:
        print("No data processed. ItemHourLog.csv not created.")


if __name__ == "__main__":
    # Define paths
    TOPK_PATH = ""
    OUTPUT_PATH = "./preprocessed"  # Directory to save the ItemHourLog.csv

    # Generate ItemHourLog.csv
    generate_item_hour_log(TOPK_PATH, OUTPUT_PATH)
