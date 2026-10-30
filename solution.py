"""Network Intrusion Detection ML Challenge solution.
Put train.csv, test.csv and optional sample_submission.csv beside this script.
Run: python solution.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from catboost import CatBoostClassifier

DATA_DIR = os.environ.get("DATA_DIR", ".")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "submission.csv")
SEED = 42

def make_model(iterations=1000, verbose=100):
    return CatBoostClassifier(
        iterations=iterations, depth=8, learning_rate=0.05,
        loss_function="MultiClass", auto_class_weights="Balanced",
        random_seed=SEED, verbose=verbose, allow_writing_files=False
    )

def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X = train.drop(columns=["Target", "id"]).copy()
    y = train["Target"].astype(str)
    X_test = test.drop(columns=["id"]).reindex(columns=X.columns).copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].fillna("Missing").astype(str)
        X_test[col] = X_test[col].fillna("Missing").astype(str)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
    X_test[num_cols] = X_test[num_cols].replace([np.inf, -np.inf], np.nan)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    validation_model = make_model()
    validation_model.fit(
        X_tr, y_tr, cat_features=cat_cols,
        eval_set=(X_val, y_val), early_stopping_rounds=100
    )
    val_pred = validation_model.predict(X_val).ravel()
    print("Validation Macro F1:", f1_score(y_val, val_pred, average="macro"))
    print("Validation Weighted F1:", f1_score(y_val, val_pred, average="weighted"))
    print(classification_report(y_val, val_pred, zero_division=0))

    best_iter = validation_model.best_iteration_
    if best_iter is None or best_iter < 1:
        best_iter = 1000
    final_model = make_model(iterations=int(best_iter) + 1)
    final_model.fit(X, y, cat_features=cat_cols)

    submission = pd.DataFrame({"id": test["id"], "Target": final_model.predict(X_test).ravel()})
    if os.path.exists(SAMPLE_PATH):
        sample = pd.read_csv(SAMPLE_PATH)
        if set(sample.columns) != set(submission.columns):
            raise ValueError(f"Submission columns mismatch. Expected {list(sample.columns)}")
        submission = submission[sample.columns]
    submission.to_csv(OUTPUT_PATH, index=False)
    print(submission.head())
    print("Saved submission:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
