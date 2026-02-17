# python -m src.models.train_baseline

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features.preprocessing import (
    split_X_y,
    identify_feature_types,
    train_val_split,
    build_preprocessor,
)

from src.models.baseline import build_baseline_model

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

# Evaluate
y_val_pred = pipeline.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred)

print("Validation ROC AUC:", roc_auc)

