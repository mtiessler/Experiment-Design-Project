import os
import pandas as pd
import logging
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv
import torchtuples as tt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    "model_path": "./models/cox_model.pth",
    "prediction_path": "./output/grv_predictions",
    "start_time": 1,
}
DATASET_PATH = "../MINDTOPK/"
OUTPUT_PATH = "./output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)


# Step 1: Load and Preprocess Datasets
def preprocess_data():
    print("Loading datasets...")
    # Load datasets
    train = pd.read_csv(os.path.join(DATASET_PATH, "train.csv"), sep="\t")
    val = pd.read_csv(os.path.join(DATASET_PATH, "dev.csv"), sep="\t")
    test = pd.read_csv(os.path.join(DATASET_PATH, "test.csv"), sep="\t")

    # Generate required columns
    for df in [train, val, test]:
        if "label" not in df.columns:
            df["label"] = 1  # Example: Set `label` to 1 for all interactions
        if "duration" not in df.columns:
            df["duration"] = df["time"] - df["time"].min()  # Placeholder logic
        if "event" not in df.columns:
            df["event"] = 1  # Mark all events as active

    # Drop unwanted columns like neg_items
    feature_columns = [col for col in train.columns if col not in ["item_id", "label", "duration", "event", "neg_items"]]

    # Extract features and labels
    x_train = train[feature_columns]
    y_train = train[["duration", "event"]]

    x_val = val[feature_columns]
    y_val = val[["duration", "event"]]

    x_test = test[feature_columns]
    y_test = test[["duration", "event"]]

    # Standardize features
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=feature_columns)
    x_val = pd.DataFrame(scaler.transform(x_val), columns=feature_columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=feature_columns)

    return x_train, y_train, x_val, y_val, x_test, y_test, train, val, test

# Step 2: Train Cox Model and Predict GRV
def train_and_predict_grv(x_train, y_train, x_test, train_df):
    print("Training Cox model and predicting GRV...")
    # Define Cox Model
    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    batch_norm = True
    dropout = 0.1
    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
    model = CoxTime(net, tt.optim.Adam)

    # Train Cox model
    batch_size = 256
    lrfinder = model.lr_finder(x_train.values, y_train.values, batch_size=batch_size, tolerance=2)
    print(f"[train] Best learning rate: {lrfinder.get_best_lr()}")
    model.optimizer.set_lr(lrfinder.get_best_lr())

    epochs = 512
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True
    log = model.fit(x_train.values, y_train.values, batch_size, epochs, callbacks, verbose)

    model.compute_baseline_hazards()
    model.save_net(CONFIG["model_path"])
    print(f"Model saved to {CONFIG['model_path']}")

    # Predict GRV
    surv = model.predict_surv_df(x_test.values)
    grv_predictions = surv.T
    grv_predictions["item_id"] = train_df["item_id"].values
    grv_predictions.to_csv(CONFIG["prediction_path"] + ".csv", index=False)
    print(f"Predictions saved to {CONFIG['prediction_path']}.csv")

    return grv_predictions


# Step 3: Integrate GRV into Datasets
def integrate_grv(interactions, grv_predictions):
    print("Integrating GRV into datasets...")
    interactions = interactions.merge(grv_predictions, on="item_id", how="left")
    return interactions


# Step 4: Train and Evaluate Models
def train_and_evaluate(train, val, test, with_grv=False):
    print(f"Training and evaluating model {'with GRV' if with_grv else 'without GRV'}...")

    # Add GRV or use item popularity for baseline
    if with_grv:
        train["score"] = train["grv"]
    else:
        train["score"] = train.groupby("item_id")["label"].transform("sum")

    # Map scores to validation and test sets
    val["score"] = val["item_id"].map(train.groupby("item_id")["score"].mean())
    test["score"] = test["item_id"].map(train.groupby("item_id")["score"].mean())

    # Compute NDCG scores
    val_ndcg = ndcg_score([val["label"]], [val["score"]])
    test_ndcg = ndcg_score([test["label"]], [test["score"]])
    return val_ndcg, test_ndcg


# Step 5: Plot Comparison Results
def plot_comparison(results):
    print("Plotting comparison...")
    metrics = ["Validation NDCG", "Test NDCG"]
    models = ["Baseline", "With GRV"]

    for metric in metrics:
        plt.figure(figsize=(8, 6))
        plt.bar(models, results[metric], color=["blue", "orange"])
        plt.title(f"{metric} Comparison")
        plt.ylabel(metric)
        plt.show()


# Main Workflow
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Preprocess datasets
    x_train, y_train, x_val, y_val, x_test, y_test, train, val, test = preprocess_data()

    # Train Cox model and predict GRV
    grv_predictions = train_and_predict_grv(x_train, y_train, x_test, train)

    # Integrate GRV into datasets
    train_with_grv = integrate_grv(train, grv_predictions)
    val_with_grv = integrate_grv(val, grv_predictions)
    test_with_grv = integrate_grv(test, grv_predictions)

    # Train and evaluate baseline model (without GRV)
    baseline_val_ndcg, baseline_test_ndcg = train_and_evaluate(train, val, test, with_grv=False)

    # Train and evaluate model with GRV
    grv_val_ndcg, grv_test_ndcg = train_and_evaluate(train_with_grv, val_with_grv, test_with_grv, with_grv=True)

    # Results comparison
    results = {
        "Validation NDCG": [baseline_val_ndcg, grv_val_ndcg],
        "Test NDCG": [baseline_test_ndcg, grv_test_ndcg],
    }

    # Plot comparison
    plot_comparison(results)

    print("Workflow complete!")
