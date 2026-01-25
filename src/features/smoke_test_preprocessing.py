# python -m src.features.smoke_test_preprocessing

import pandas as pd
from src.features.preprocessing import split_X_y, identify_feature_types

df = pd.read_csv('data/raw/application_train.csv')

X, y = split_X_y(df)
numeric_cols, categorical_cols = identify_feature_types(X)

print("Total features:", X.shape[1])
print("Numerical features:", len(numeric_cols))
print("Categorical features:", len(categorical_cols))
print("Total", len(numeric_cols) + len(categorical_cols))

print("\nSample numerical features:", numeric_cols[:5])

print("\nSample categorical features:", categorical_cols[:5])

