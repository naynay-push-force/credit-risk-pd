# python -m src.models.train_baseline

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.features.preprocessing import (
    split_X_y,
    identify_feature_types,
    train_val_split,
    build_preprocessor,
)

from src.models.baseline import build_baseline_model
from src.models.metrics import ks_statistic

df = pd.read_csv("data/raw/application_train.csv")

# Data plumbing
X, y = split_X_y(df)
numeric_cols, categorical_cols = identify_feature_types(X)
X_train, X_val, y_train, y_val = train_val_split(X, y)

# Preprocessing + model
preprocessor = build_preprocessor(numeric_cols, categorical_cols)
pipeline = build_baseline_model(preprocessor)

# Train
pipeline.fit(X_train, y_train)

y_val_pred = pipeline.predict_proba(X_val)[:, 1]

# Evaluate
roc_auc = roc_auc_score(y_val, y_val_pred)
pr_auc = average_precision_score(y_val, y_val_pred)
ks, ks_threshold = ks_statistic(y_val, y_val_pred)

print("Validation ROC AUC:", roc_auc)
print("Validation PR-AUC:", pr_auc)
print("Validation KS:", ks)
print("KS Threshold:", ks_threshold)
print("sanity check", y_val_pred.min(), y_val_pred.max(), y_val_pred.mean())