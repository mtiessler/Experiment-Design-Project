# Reproduction of the Timeliness-aware Fair Recommendation (TaFR)

This repository accompanies the paper:

**Jiayin Wang, Weizhi Ma, Chumeng Jiang, Min Zhang, Yuan Zhang, Biao Li, and Peng Jiang.** 2023. _Measuring Item Global Residual Value for Fair Recommendation_. In SIGIR '23.

---

## 1. Repository Overview

```
Experiment-Design-Project/
├── data/
│   ├── MIND_large/          # ReChorus-based MIND data (train/dev/test/preprocessed)
│   └── ...                  # Output logs or additional data
├── output/                  # Saved outputs from scripts (e.g., cox, model predictions)
├── src/
│   ├── GRV/                 # Refactored code (original author’s) for GRV generation & Cox modeling
│   │   ├── model/
│   │   ├── pre_process/
│   │   ├── predictions/
│   │   ├── utils/
│   │   ├── config.csv
│   │   └── main.py
│   ├── ReChorus/            # The ReChorus library code w/ MIND dataset generator & preprocessing
│   │   ├── data/
│   │   ├── docs/
│   │   ├── src/
│   │   ├── requirements.txt
│   │   └── ...
│   └── ...
└── TaFR-reproducible/
    ├── cox_output/
    ├── ReChorus_MIND_dataset/
    ├── 1_item_hour_log_from_ReChorus.py
    ├── 2_COX_GRV.py
    └── ...
```

### Directory Details:
- **GRV/**: Refactored code for generating the Cox model and GRV values.
- **ReChorus/**: The ReChorus framework, including the MIND dataset loader & preprocessing pipeline.
- **TaFR-reproducible/**: Integrated scripts for the full TaFR pipeline.
- **data/**: Contains MIND data (train/val/test) and other data necessary for GRV original code.

> **Note:** The final experiments used both the MIND and Kuai datasets, evaluated on HR@k, NDCG@k, and coverage metrics. 
---

## 2. Additions to Improve Reproducibility

- Added a `requirements.txt` to track Python dependencies.
- Generated `.csv` config files for flexible parameter specification.
- Edited `main.py` to parse config values from CSV (removed `eval()` calls for security).
- Restructured the file system and fixed import references.
- Fixed unknown module references in `main.py`, `label.py`, and `coxDataLoader.py`.
- Removed unused imports across multiple scripts.
- Updated `coxDataLoader.py` to read from config CSV instead of command-line args.
- Added an itemHourLog generator script to produce `itemHourLog.csv`.
- Split `itemHourLog.csv` into train/val/test sets.
- Re-enabled and improved the model definition in `COX.py`.
- Enhanced plotting for survival analysis and calibration curves.

---

## 3. ReChorus Modifications

- Added path references to data in ReChorus scripts.
- Implemented chunk-based file processing to handle large `behaviors.tsv` data.
- Early sampling of data during chunk processing to limit dataset size for faster development.
- Lowered interaction frequency threshold (e.g., `MIN_INTERACTIONS=2`) to reduce dataset size.
- Vectorized negative sampling for performance improvements.
- Added progress tracking with `tqdm`.
- Reduced the training timeline to fewer days for a smaller dataset.

---

## 4. Step-by-Step Reproduction Procedure

Execute the file `reproduction.ipynb` for a full end-to-end reproduction of the experiment.

### 1. **Dataset Preparation**

- **MIND Dataset**: ReChorus generated one. You can find the already splitted sets in the folder `ReChorus_MIND_dataset` 
- **Filtering**: Apply a 10-core filter (retain only users/items with ≥10 interactions).
- **Splits**:
  - Train: First 3 days
  - Validation + Test: Subsequent 2 days

### 2. **Global Residual Value (GRV) Module**

- Define observation (`T_obs`) and prediction (`T_pred`) windows.
- Collect user feedback (e.g., CTR, watch ratio) in the observation window.
- Train a Cox survival model to estimate item “time-to-death.”
- Output: GRV vector (survival probabilities) for each item across future hours.

### 3. **Generate Recommendation Datasets**

- Incorporate GRV predictions into item metadata.
- Produce training files for each backbone model, with and without GRV features.

### 4. **Backbone Recommendation Models**

- Models: NeuMF, GRU4Rec, TiSASRec (or other algorithms).

- Train both baseline (no GRV) and GRV-enhanced models:

\[
s\text{core}(u, i) = (1 - \gamma) \times \text{backbone\_score}(u, i) + \gamma \times \text{GRV}_i
\]

### 5. **Experimental Settings**

- **Hyperparameters**:
  - `T_obs = 12` hours (MIND)
  - `T_pred = 7` days
  - \(\gamma \in [0, 0.5]\) for weighting

- **Metrics**: HR@K, NDCG@K, Cov@K, N_Cov@K

### 6. **Comparison and Analysis**

- **Accuracy**: Check HR@K and NDCG@K stability or improvement with GRV.
- **Fairness**: Evaluate changes in item coverage, focusing on newer items.

---

## Reference

```bibtex
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

> **Contact:** For questions, please reach out to Jiayin Wang (JiayinWangTHU at gmail dot com).

> **Disclaimer:** This repository is a work in progress. It implements core concepts from the paper but may not exactly match every experimental detail from the SIGIR publication.
