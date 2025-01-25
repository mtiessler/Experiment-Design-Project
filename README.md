# Timeliness-aware Fair Recommendation (TaFR)

This is the Repo for the paper:

Jiayin Wang, Weizhi Ma, Chumeng Jiang, Min Zhang, Yuan Zhang, Biao Li and Peng Jiang, 2023. Measuring Item Global Residual Value for Fair Recommendation. In SIGIR'23.

Currently under constructions.
**Disclaimer**: The paper evaluates its methods on specific datasets (MIND, Kuai) and metrics like Hit Rate (HR@k), NDCG@k, and coverage metrics. The code doesn't include any specific evaluation metrics or datasets that replicate this analysis.

## Adds to improve reproducibility
- Added a `requirements.txt`
- generated .csv config files
- edited main.py file for preprocessing the config values in the .csv
- edited main.py to avoid using eval() for security reasons
- reestructured the filesystem to fix references (removed __init__.py files)
- fixed reference unknonw "pre" references to pre_process in main.py, label.py, coxDataLoader.py 
- removed unused imports
- coxDataLoader.py now uses to a .csv config file instead of args
- added itemHourLog generator script (uses MIND dataset for generating the itemhourlog.csv)
- fixed gitignore
- Now the itemHourLog is already splitted by training test val, so the coxDataLoader does not do anymore its task -> simplified
- defined model in COX.py has been uncommented and fixed to reintegrate with the main pipeline
- COX.py now uses the config file and has better readability
- cox data loader now generates cox.csv from the files 
- plots have been improved

## ReChorus
- Added path to data folder


# Step-by-Step Reproduction Procedure

## 1. Dataset Preparation

### Datasets:
- **MIND Dataset**: Download from the [MIND dataset site](https://msnews.github.io/).
- **Kuai Dataset**: The paper mentions this will be publicly available. If not accessible, use an alternative short-video dataset with similar characteristics.

### Preprocessing:
- Apply a **10-core filter** (only keep users and items with at least 10 interactions).
- Split data into:
  - **Training Set**: First 3 days of interactions.
  - **Validation & Test Sets**: Last 2 days of interactions.

---

## 2. Global Residual Value (GRV) Module

### Input Features:
- Collect user feedback for items during an **observation period (T_obs)**:
  - Metrics like **CTR**, watch ratio, or similar user feedback.
- Define a **prediction period (T_pred)** to estimate item timeliness (GRV).

### Modeling GRV:
- Use **survival analysis (Cox proportional hazards model)** to model item timeliness. Key steps:
  - Define **deactivation labels** based on item vitality scores.
  - Train the GRV module to predict item-level timeliness for the prediction period.
- **Output**: A **GRV vector** for each item over the prediction period.

---

## 3. Generate Recommendation Datasets

- Create datasets for **CTR prediction** and **Top-K recommendation** tasks.
- Include **GRV values** as features for items in interaction files (`train.csv`, `dev.csv`, `test.csv`).
- Ensure alignment between GRV predictions and item metadata.

---

## 4. Backbone Recommendation Models

### Train Baseline Models:
- Use standard recommendation algorithms as backbones:
  - **NeuMF** (Collaborative Filtering).
  - **GRU4Rec** (Sequential Recommendation).
  - **TiSASRec** (Time-sensitive Sequential Recommendation).
- Evaluate recommendation performance **without GRV** (baseline).

### Integrate GRV into Backbones:
- Use the formula:

  \[
  G_r(u,t)(I) = (1 - \gamma) \cdot BBM_r(u,t)(I) + \gamma \cdot GRV_i(t)
  \]

  Where:
  - \( BBM_r \): Backbone model prediction.
  - \( GRV_i \): GRV prediction for item \( i \).
  - \( \gamma \): Weight parameter for GRV integration.

- Train and evaluate each backbone model **with GRV**.

---

## 5. Experimental Settings

### Hyperparameters:
- \( T_{obs} \): Observation period (e.g., 12 hours for MIND, 24 hours for Kuai).
- \( T_{pred} \): Prediction period (e.g., 7 days).
- \( \gamma \): GRV weight (e.g., 0.1–0.3 based on grid search).

### Evaluation Metrics:
- **Accuracy**: HR@K, NDCG@K.
- **Fairness**: Coverage of new items (N_Cov@K), overall item coverage (Cov@K).

### Experiments:
1. Train baseline models **without GRV**.
2. Train models **with GRV integration**.
3. Compare results to measure improvements in **accuracy** and **fairness**.

---

## 6. Comparison and Analysis

### Accuracy:
- Ensure GRV does not degrade standard recommendation metrics (**HR@K**, **NDCG@K**).
- Check for consistent or improved performance.

### Fairness:
- Evaluate improvements in **new item exposure** (N_Cov@K) and **overall item coverage** (Cov@K).
- Analyze exposure distribution across items grouped by upload time.


## Reference
```
@inproceedings{DBLP:conf/SIGIR/WangMJZYLJ23,
  author    = {Jiayin Wang and
               Weizhi Ma and
               Chumeng Jiang and
               Min Zhang and
               Zhang Yuan and
               Biao Li and
               Peng Jiang},
  title     = {Measuring Item Global Residual Value for Fair Recommendation},
  booktitle = {{SIGIR} '23: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  doi       = {10.1145/3539618.3591724}
}
```

For inquiries, contact Jiayin Wang (JiayinWangTHU AT gmail.com).
