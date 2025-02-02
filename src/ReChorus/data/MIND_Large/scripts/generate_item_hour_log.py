import pandas as pd
import numpy as np
import os

def process_behaviors(behaviors_file, news_mapping):
    print(f"Processing behaviors file: {behaviors_file}")
    behaviors = pd.read_csv(
        behaviors_file,
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impression_list"]
    )

    # Convert time to datetime and extract hourly timelevel
    behaviors["time"] = pd.to_datetime(behaviors["time"])
    behaviors["timelevel"] = behaviors["time"].dt.floor("h")

    # Parse impression list to extract exposures and clicks
    def parse_impressions(impression_list):
        impressions = impression_list.split(" ")
        exposure = len(impressions)
        clicks = sum(1 for imp in impressions if "-1" in imp)  # Clicked items have '-1'
        return exposure, clicks

    behaviors[["exposures", "clicks"]] = behaviors["impression_list"].apply(
        lambda x: pd.Series(parse_impressions(x))
    )

    # Map news_id (history) to photo_id
    behaviors = behaviors.merge(news_mapping, left_on="history", right_on="news_id", how="left")

    return behaviors


def compute_play_rate_and_new_pctr(item_hour_log):
    # Simulate play_time and photo_time for play_rate calculation
    np.random.seed(42)
    item_hour_log["photo_time"] = np.random.uniform(30, 300, size=len(item_hour_log))  # Simulate content duration (30-300 seconds)
    item_hour_log["play_time"] = item_hour_log["clicks"] * np.random.uniform(10, 30, size=len(item_hour_log))  # Simulate play_time

    # Compute play_rate
    item_hour_log["play_rate"] = item_hour_log["play_time"] / item_hour_log["photo_time"]
    item_hour_log["play_rate"] = item_hour_log["play_rate"].clip(upper=1)  # Ensure play_rate <= 1

    # Compute new_pctr as historical click_rate
    item_hour_log["new_pctr"] = item_hour_log["click_rate"]

    return item_hour_log


def generate_item_hour_log(base_path, dataset_folder):
    folder_path = os.path.join(base_path, dataset_folder)

    # File paths
    behaviors_file = os.path.join(folder_path, "behaviors.tsv")
    news_file = os.path.join(folder_path, "news.tsv")

    # Ensure necessary files exist
    if not os.path.exists(behaviors_file) or not os.path.exists(news_file):
        print(f"Missing required files in {folder_path}. Skipping...")
        return None

    # Load news.tsv and create mapping for news_id to photo_id
    print(f"Processing news file: {news_file}")
    news = pd.read_csv(
        news_file,
        sep="\t",
        header=None,
        names=["news_id", "category", "subcategory", "title", "abstract", "url", "entity_list", "extra_info"]
    )
    news["photo_id"] = range(1, len(news) + 1)  # Assign sequential IDs for news
    news_mapping = news[["news_id", "photo_id"]]

    # Process behaviors.tsv
    behaviors = process_behaviors(behaviors_file, news_mapping)

    # Aggregate by photo_id and timelevel
    item_hour_log = behaviors.groupby(["photo_id", "timelevel"]).agg(
        exposure=("exposures", "sum"),
        clicks=("clicks", "sum"),
    ).reset_index()

    # Compute click_rate
    item_hour_log["click_rate"] = item_hour_log["clicks"] / item_hour_log["exposure"]
    item_hour_log.fillna(0, inplace=True)  # Handle division by zero

    # Compute play_rate and new_pctr
    item_hour_log = compute_play_rate_and_new_pctr(item_hour_log)

    return item_hour_log


def add_died_column_and_save(item_hour_log, dataset_folder, output_path):
    # Define 'died' column logic: If click_rate < threshold, item 'died'
    threshold = 0.1
    item_hour_log["died"] = (item_hour_log["click_rate"] < threshold).astype(int)

    # Save the dataset to preprocessed folder
    output_file = os.path.join(output_path, f"{dataset_folder}.csv")
    os.makedirs(output_path, exist_ok=True)
    item_hour_log.to_csv(output_file, index=False)
    print(f"Saved {dataset_folder}.csv to {output_path}")


if __name__ == "__main__":
    base_path = os.path.join("..", "..", "data", "MIND")
    output_path = os.path.join("..", "..", "data", "preprocessed")

    dataset_folders = ["train", "validation", "test"]

    for dataset_folder in dataset_folders:
        print(f"Processing dataset folder: {dataset_folder}")
        item_hour_log = generate_item_hour_log(base_path, dataset_folder)
        if item_hour_log is not None:
            add_died_column_and_save(item_hour_log, dataset_folder, output_path)
