# Network Intrusion Detection ML Challenge

## Overview
This project uses CatBoost multiclass classification to predict the `Target` label for network-flow records. The observed labels are `Availability`, `Benign`, `Reconnaissance`, and `Spoofing`.

## Files
- `solution.py`: preprocessing, validation, training, and submission generation.
- `requirements.txt`: Python package requirements.
- `README.md`: methodology and reproduction instructions.

The dataset is not included. Place `train.csv`, `test.csv`, and (if provided) `sample_submission.csv` in the same folder as `solution.py`.

## Methodology and reasoning
1. Load training and test data using pandas.
2. Separate `Target` as the label and exclude `id` from model features; preserve test IDs for the submission.
3. Align test columns with the training feature columns.
4. Detect categorical columns (the observed data included `proto` and `service`), fill missing categorical values with `Missing`, and pass these columns directly to CatBoost.
5. Replace positive and negative infinity in numeric columns with missing values.
6. Use a stratified 80/20 train-validation split with random seed 42 to preserve class proportions.
7. Train CatBoost using multiclass loss, depth 8, learning rate 0.05, up to 1,000 iterations, and early stopping. `auto_class_weights="Balanced"` helps account for the highly imbalanced target classes.
8. Evaluate with Macro F1, Weighted F1, and per-class precision/recall/F1.
9. Refit on all labelled training rows using the best validation iteration count and predict the test labels.
10. Write `submission.csv` with `id` and `Target`, matching the sample submission's column order when that file is available.

## Results observed during development
- Validation Macro F1: **0.985668**
- Validation Weighted F1: **0.998949**
- Best validation iteration: **697**; final refit used 698 iterations.
- Kaggle leaderboard score: **0.84496**

The leaderboard score is substantially lower than the local validation score. The random split may not fully represent the hidden test distribution, or the data may contain distribution differences or a scoring/validation mismatch. The validation score should not be reported as the competition score.

## How to run
Use Python 3.10 or newer:

```bash
pip install -r requirements.txt
python solution.py
```

By default, the script reads the CSV files from the current directory. To use another data directory, set `DATA_DIR`.

Windows Command Prompt:
```bat
set DATA_DIR=C:\path\to\data
python solution.py
```

The output is saved as `submission.csv` in the data directory.

## Limitations and future work
- Investigate the official competition metric and data-splitting rules.
- Compare class-wise validation errors and test a more representative split if the dataset has time, source, or group structure.
- Compare alternative hyperparameters and models using the same validation method.
- Avoid tuning solely against the hidden leaderboard.

## Reproducibility
The random seed is fixed at 42. Preprocessing and model settings are defined in `solution.py`.
