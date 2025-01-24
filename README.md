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
