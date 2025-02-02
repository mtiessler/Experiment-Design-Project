import pandas as pd
import numpy as np
import os
from lifelines import CoxPHFitter


def train_and_generate_grv_with_vitality(
        item_hour_log_csv,
        output_dir="./cox_output",
        T_obs=12,
        T_pred=168,
        min_obs_hours=1,
        beta_d=-3.0
):
    """
    Similar to the earlier cox training script, but we incorporate a
    'vitality' approach for labeling the event time.
    We'll do:
      1) For each item, sort by hour_offset.
      2) Build cumulative vitality.
      3) The first hour >= T_i0+T_obs in which cumulativeVitality < beta_d => "death" hour.
         If none, censored at T_i0+T_obs+T_pred.

    If an item doesn't appear at certain hours (meaning no row), we treat vitality=0 that hour.
    We'll store the item-level (duration, event) and train Cox. Then produce cox_survival.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(item_hour_log_csv)
    required = ["item_id", "hour_offset", "exposure", "vitality"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Need vitality from Script A2.")
    df.sort_values(["item_id", "hour_offset"], inplace=True)

    items = []
    item_groups = df.groupby("item_id", sort=False)

    for it_id, group in item_groups:
        group = group.sort_values("hour_offset")
        T_i0 = group["hour_offset"].min()
        obs_end = T_i0 + T_obs
        pred_end = obs_end + T_pred

        # observation window
        obs_df = group[(group["hour_offset"] >= T_i0) & (group["hour_offset"] < obs_end)]
        # if insufficient hours => skip
        if obs_df["hour_offset"].nunique() < min_obs_hours:
            continue

        # Summation of 'exposure' or 'vitality' in obs
        sum_exposure_obs = obs_df["exposure"].sum()  # example feature
        # for Cox, we keep "sum_exposure_obs" or do more advanced

        # We'll look hour by hour in [obs_end, pred_end) to see if cummulativeVital < beta_d
        # Actually we must compute cumulative vitality from T_i0 up to each hour offset.
        # array from T_i0 => pred_end-1
        full_offsets = np.arange(T_i0, pred_end)
        # We'll map them to vitality or 0 if no row
        group_indexed = group.set_index("hour_offset")
        cume_val = 0.0
        event_time = pred_end
        event_flag = 0

        for h in full_offsets:
            if h in group_indexed.index:
                vit = group_indexed.loc[h, "vitality"]
                # if there's multiple rows, sum them, but typically there's 1
                if isinstance(vit, pd.Series):
                    vit = vit.sum()
            else:
                vit = 0.0  # no row => vitality=0
            cume_val += vit
            # only check if h >= obs_end
            if h >= obs_end:
                if cume_val < beta_d:
                    event_time = h
                    event_flag = 1
                    break

        duration = event_time - T_i0
        items.append({
            "item_id": it_id,
            "sum_exposure_obs": sum_exposure_obs,
            "duration": duration,
            "event": event_flag,
            "T_i0": T_i0
        })

    cox_df = pd.DataFrame(items)
    if cox_df.empty:
        print("No items. Exiting.")
        return

    # Fit Cox
    cph = CoxPHFitter()
    try:
        cph.fit(
            cox_df[["duration", "event", "sum_exposure_obs", "item_id", "T_i0"]],
            duration_col="duration",
            event_col="event",
            show_progress=True
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    cox_df_out = os.path.join(output_dir, "cox_data.csv")
    cox_df.to_csv(cox_df_out, index=False)
    print(f"[INFO] Saved cox_data.csv => {cox_df_out}")

    # cph summary
    with open(os.path.join(output_dir, "cox_model_summary.txt"), "w") as f:
        f.write(str(cph.summary))

    # Build survival => cox_survival.csv
    hour_points = list(range(T_obs + 1, T_obs + T_pred + 1))
    item_features = {}
    for row in cox_df.itertuples(index=False):
        item_features[row.item_id] = {
            "sum_exposure_obs": row.sum_exposure_obs,
            "T_i0": row.T_i0
        }
    grv_rows = []

    for it_id, feats in item_features.items():
        row_dict = {
            "sum_exposure_obs": feats["sum_exposure_obs"],
            "duration": 0,
            "event": 0,
            "item_id": it_id,
            "T_i0": feats["T_i0"]
        }
        df_item = pd.DataFrame([row_dict])
        surv_fn = cph.predict_survival_function(df_item, times=hour_points)
        # shape (#times, #rows=1)
        arr = surv_fn.iloc[:, 0].tolist()
        out = {"item_id": it_id}
        for i, t_val in enumerate(hour_points):
            out[f"GRV_t{t_val}"] = arr[i]
        grv_rows.append(out)
    grv_df = pd.DataFrame(grv_rows)
    surv_out = os.path.join(output_dir, "cox_survival.csv")
    grv_df.to_csv(surv_out, index=False)
    print(f"[INFO] Wrote survival => {surv_out}")


if __name__ == "__main__":
    item_hour_log_csv = "./item_hour_log/ItemHourLog.csv"
    train_and_generate_grv_with_vitality(
        item_hour_log_csv=item_hour_log_csv,
        output_dir="./cox_output",
        T_obs=12,
        T_pred=168,
        min_obs_hours=1,
        beta_d=-3.0
    )
