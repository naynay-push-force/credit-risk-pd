# python -m src.models.train_baseline

from config import FEATURE_CONFIG

import pandas as pd
from typing import Tuple

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

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

    Split discipline:
        - One stratified train/test split. The TEST set is touched exactly once,
          at the end, for the reported metrics.
        - Calibration (Platt) is fit on the TRAIN set only, via internal CV folds
          (CalibratedClassifierCV(cv=5)) -- so no row is both fit-on and reported-on.

    Returns:
        - model:       fitted estimator (pipeline, optionally calibration-wrapped)
        - X_test:      held-out test features (raw, before preprocessing)
        - y_test:      held-out test labels
        - y_test_pred: predicted PD (class-1 probabilities) on the test set
    """

    df = pd.read_csv(data_path)
    df = add_application_features(df)

    # Downcast numeric columns to reduce memory pressure during imputation
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    int_cols = df.select_dtypes(include="int64").columns
    df[int_cols] = df[int_cols].astype("int32")

    X, y = split_X_y(df)

    # Apply config (section B refactor candidate)
    if FEATURE_CONFIG["drop_cols"]:
        X = X.drop(columns=FEATURE_CONFIG["drop_cols"])
    if FEATURE_CONFIG["keep_cols"]:
        X = X[FEATURE_CONFIG["keep_cols"]]
    for col in FEATURE_CONFIG["force_categorical"]:
        X = X[col].astype("object")

    # Data plumbing
    numeric_cols, categorical_cols = identify_feature_types(X)
    X_train, X_test, y_train, y_test = train_val_split(X, y)

    # Preprocessing, model + train
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = build_baseline_model(preprocessor)
    model.fit(X_train, y_train)

    y_test_pred = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, y_test_pred


# Evaluate
def main() -> None:
    model, X_test, y_test, y_test_pred = train_and_predict()
    
    roc_auc = roc_auc_score(y_test, y_test_pred)
    pr_auc = average_precision_score(y_test, y_test_pred)
    ks, ks_threshold = ks_statistic(y_test, y_test_pred)

    print("Test ROC AUC:", roc_auc)
    print("Test PR-AUC:", pr_auc)
    print("Test KS:", ks)
    print("KS Threshold:", ks_threshold)
    print("Sanity check", y_test_pred.min(), y_test_pred.max(), y_test_pred.mean())

if __name__ == "__main__":
    main()