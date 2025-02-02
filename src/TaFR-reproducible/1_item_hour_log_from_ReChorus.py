import pandas as pd
import numpy as np
import os

def generate_basic_item_hour_log(
    data_path,
    output_path,
    datasets=("train", "dev", "test")
):
    """
    Step A1:
      Generates a basic ItemHourLog.csv from MIND data files (train/dev/test).
      Each row => (item_id, hour_offset, exposure, dataset).
      We'll skip 'clicked' in the hour log. We'll simply count how many rows => exposure.
      If you do have a 'clicked' column, you can incorporate it.
    """
    os.makedirs(output_path, exist_ok=True)
    global_earliest_time = None
    all_logs = []

    # 1) find earliest timestamp across all splits
    for ds in datasets:
        csv_file = os.path.join(data_path, f"{ds}.csv")
        if not os.path.exists(csv_file):
            print(f"[WARN] {csv_file} not found. Skipping {ds}.")
            continue
        df = pd.read_csv(csv_file, sep="\t")
        df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
        tmin = df["time"].min()
        if global_earliest_time is None or tmin < global_earliest_time:
            global_earliest_time = tmin

    if global_earliest_time is None:
        print("[ERROR] No valid data found. Aborting.")
        return

    # 2) For each dataset, group by (item, hour_offset)
    for ds in datasets:
        csv_file = os.path.join(data_path, f"{ds}.csv")
        if not os.path.exists(csv_file):
            continue

        print(f"[INFO] Processing {ds} ...")
        df = pd.read_csv(csv_file, sep="\t")
        df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
        df["timelevel"] = df["time"].dt.floor("h")

        # hour_offset from earliest
        df["hour_offset"] = (
            (df["timelevel"] - global_earliest_time).dt.total_seconds() // 3600
        ).astype(int)

        # Count how many rows => "exposure"
        # if you want to incorporate 'clicked', do it separately
        item_hour_df = (
            df.groupby(["item_id", "hour_offset"], as_index=False)
            .agg(exposure=("user_id", "count"))
        )
        item_hour_df["dataset"] = ds
        all_logs.append(item_hour_df)

    if not all_logs:
        print("[WARN] No logs found.")
        return

    final = pd.concat(all_logs, ignore_index=True)
    final.sort_values(["item_id", "hour_offset"], inplace=True)

    outfile = os.path.join(output_path, "ItemHourLog_basic.csv")
    final.to_csv(outfile, index=False)
    print(f"[INFO] Wrote basic item-hour logs to {outfile}.")

def compute_vitality_for_item_hour_log(
    input_csv,
    output_csv,
    beta_E=0.5,
    beta_NE=0.5
):
    """
    Step A2:
      Reads the basic item-hour log from input_csv,
      for each hour_offset, we rank items by a user feedback metric (exposure or CTR).
      Then compute vitality = rank_percentile - beta_E if exposure>0, else -beta_NE.
      We'll store the hour-by-hour 'vitality' in a new column, plus a cumulative vitality.
      The paper references thresholding on the cumulative sum for "death," but we'll
      store it here so the next script can handle it.
    """
    df = pd.read_csv(input_csv)
    # We'll assume 'exposure' is the measure of user feedback.
    # If you have actual CTR or click_count, you'd rank by that.

    # rank items *within each hour_offset* by exposure
    # then compute percentile rank. For each hour_offset => sort items by exposure
    # We can do a groupby on hour_offset
    all_rows = []
    for hour_off, group in df.groupby("hour_offset", sort=True):
        # sort by exposure descending
        sorted_g = group.sort_values("exposure", ascending=False)
        n = len(sorted_g)
        # rank percentile => rank(i) / n
        # let's do zero-based rank
        sorted_g["rank"] = np.arange(n)
        # rank_percentile = 1 - (rank / (n-1)) if we want top exposure => percentile near 1
        # or we can do rank / (n-1)
        # We'll define: rank_percentile = 1 - (rank / (n-1)) so that item with highest exposure => percentile=1
        if n>1:
            sorted_g["rank_percentile"] = 1.0 - (sorted_g["rank"]/(n-1))
        else:
            sorted_g["rank_percentile"] = 1.0

        all_rows.append(sorted_g)

    newdf = pd.concat(all_rows, ignore_index=True)
    # define vitality
    def compute_vitality(row):
        if row["exposure"]>0:
            return row["rank_percentile"] - beta_E
        else:
            return -beta_NE

    newdf["vitality"] = newdf.apply(compute_vitality, axis=1)
    newdf.sort_values(["item_id","hour_offset"], inplace=True)

    newdf.to_csv(output_csv, index=False)
    print(f"[INFO] Wrote vitality-based itemHourLog to {output_csv}.")

if __name__=="__main__":
    DATA_PATH = "./ReChorus_MIND_dataset"
    OUTPUT_PATH = "./item_hour_log"

    # Step A1
    generate_basic_item_hour_log(DATA_PATH, OUTPUT_PATH, datasets=["train","val","test"])
    # Step A2
    compute_vitality_for_item_hour_log(
        input_csv=os.path.join(OUTPUT_PATH, "ItemHourLog_basic.csv"),
        output_csv=os.path.join(OUTPUT_PATH, "ItemHourLog.csv"),
        beta_E=0.5,
        beta_NE=0.5
    )
