# python -m src.models.train_baseline

from config import FEATURE_CONFIG

import pandas as pd
from typing import Tuple

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

from src.features.feature_engineering  import add_application_features

from src.features.preprocessing import (
    split_X_y,
    identify_feature_types,
    train_val_split,
    build_preprocessor,
)

from src.models.baseline import build_baseline_model
from src.models.metrics import ks_statistic

def train_and_predict(
        data_path: str = "data/raw/application_train.csv"
) -> Tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train baseline pipeline and return what evaluation needs.

    Returns:
        - pipeline: fitted sklearn Pipeline (preprocessor + logistic regression)
        - X_val: validation features (raw, before preprocessing)
        - y_val: validation labels
        - y_val_pred: predicted probabilities for class 1 on validation set
    """

    df = pd.read_csv(data_path)
    df = add_application_features(df)

    # Downcast numeric columns to reduce memory pressure during imputation
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    int_cols = df.select_dtypes(include="int64").columns
    df[int_cols] = df[int_cols].astype("int32")

    X, y = split_X_y(df)

    # Drop redundant columns
    if FEATURE_CONFIG["drop_cols"]:
        X = X.drop(columns=FEATURE_CONFIG["drop_cols"])
    
    # Apply keep_cols (overrides everything else)
    if FEATURE_CONFIG["keep_cols"]:
        X = X[FEATURE_CONFIG["keep_cols"]]

    # Apply force_categorical
    for col in FEATURE_CONFIG["force_categorical"]:
        X = X[col].astype("object")

    # Data plumbing
    numeric_cols, categorical_cols = identify_feature_types(X)
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    # Preprocessing + model
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = build_baseline_model(preprocessor)

    # Train
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict_proba(X_val)[:, 1]

    return pipeline, X_val, y_val, y_val_pred


# Evaluate
def main() -> None:
    pipeline, X_val, y_val, y_val_pred = train_and_predict()
    
    roc_auc = roc_auc_score(y_val, y_val_pred)
    pr_auc = average_precision_score(y_val, y_val_pred)
    ks, ks_threshold = ks_statistic(y_val, y_val_pred)

    print("Validation ROC AUC:", roc_auc)
    print("Validation PR-AUC:", pr_auc)
    print("Validation KS:", ks)
    print("KS Threshold:", ks_threshold)
    print("Sanity check", y_val_pred.min(), y_val_pred.max(), y_val_pred.mean())

if __name__ == "__main__":
    main()