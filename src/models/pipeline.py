"""
Training-pipeline steps for the credit-risk PD model.

A single-responsibility decomposition of what used to be one
`train_and_predict` god function. `run_evaluation` orchestrates these in
order: load_data -> make_splits -> build_pipeline -> train -> persist.
"""

from typing import Tuple, List
from pathlib import Path

import pandas as pd
import joblib
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from config import RunConfig
from src.features.preprocessing import (
    split_X_y,
    train_val_split,
    build_preprocessor,
)
from src.models.baseline import build_baseline_model


def load_data(path: Path) -> pd.DataFrame:
    """Read the raw application CSV into a DataFrame."""
    return pd.read_csv(path)


def make_splits(
    df: pd.DataFrame,
    cfg: RunConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features/target, apply the config's column selection, and make one
    stratified train/test split. The test set is held out for a single, final
    evaluation -- it must not inform any modelling decision.
    """
    X, y = split_X_y(df)

    if cfg.drop_cols:
        X = X.drop(columns=cfg.drop_cols)
    if cfg.keep_cols:
        X = X[cfg.keep_cols]

    X_train, X_test, y_train, y_test = train_val_split(X, y)

    return X_train, X_test, y_train, y_test


def build_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    cfg: RunConfig,
) -> Pipeline:
    """
    Build the unfitted estimator: a ColumnTransformer preprocessor + logistic
    regression. Calibration is not applied here -- that happens in train(), so
    this bare pipeline can also be cross-validated cheaply.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = build_baseline_model(preprocessor, cfg)
    return model


def train(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: RunConfig,
) -> BaseEstimator:
    """
    Fit the estimator on the training data and return it fitted.

    If the config requests Platt calibration, the whole pipeline is wrapped in
    CalibratedClassifierCV(cv=5) first -- so the sigmoid is fit on held-out
    folds of the training set (never on the reported test set), and the
    returned object is a CalibratedClassifierCV rather than a bare Pipeline.
    """
    model = estimator

    if cfg.calibration == "platt":
        model = CalibratedClassifierCV(estimator, method='sigmoid', cv=5)

    model.fit(X_train, y_train)

    return model


def persist(
    model: BaseEstimator,
    path: Path,
) -> None:
    """Save the fitted model to disk with joblib, so it can be reloaded for
    scoring without retraining."""
    joblib.dump(model, path)
