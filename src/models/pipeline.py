from typing import Dict, Tuple, List

import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from src.features.preprocessing import (
    split_X_y, 
    train_val_split,
    build_preprocessor,
    )
from src.models.baseline import build_baseline_model

import joblib

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def make_splits(df: pd.DataFrame,
                cfg: Dict,
)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y = split_X_y(df)

    # Apply config (section B refactor candidate)
    if cfg["drop_cols"]:
        X = X.drop(columns=cfg["drop_cols"])
    if cfg["keep_cols"]:
        X = X[cfg["keep_cols"]]

    X_train, X_test, y_train, y_test = train_val_split(X, y)

    return X_train, X_test, y_train, y_test

def build_pipeline(numeric_cols: List[str],
                   categorical_cols: List[str],
) -> Pipeline:
    # Preprocessing, model + train
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = build_baseline_model(preprocessor)
    return model

def train(estimator: Pipeline,
          X_train: pd.DataFrame,
          y_train: pd.Series,
          cfg: Dict,
) -> BaseEstimator:
    model = estimator

    if cfg.get("calibration", "none") == "platt":
        model = CalibratedClassifierCV(estimator, method='sigmoid', cv=5)

    model.fit(X_train, y_train)

    return model

def persist(model: BaseEstimator,
            path: Path,
) -> None:
    joblib.dump(model, path)
    
