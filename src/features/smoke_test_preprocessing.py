# python -m src.features.smoke_test_preprocessing
import pandas as pd
from src.features.preprocessing import (
    split_X_y, 
    identify_feature_types,
    impute_missing_simple
)

# Load data
df = pd.read_csv('data/raw/application_train.csv')
X, y = split_X_y(df)
numeric_cols, categorical_cols = identify_feature_types(X)

# Confirm variable types and their counts
print("Total features:", X.shape[1])
print("Numerical features:", len(numeric_cols))
print("Categorical features:", len(categorical_cols))
print("Total", len(numeric_cols) + len(categorical_cols))

print("\nSample numerical features:", numeric_cols[:5])

print("\nSample categorical features:", categorical_cols[:5])

# Analyse missingness
missing_frac = X.isna().mean().sort_values(ascending=False)

print("Top 15 features with highest missing fraction:")
print(missing_frac.head(15))

print("\nHow many columns have > 50% missing?")
print((missing_frac > 0.5).sum())

# Simple imputation test
before = X.isna().sum().sum()
X_imp = impute_missing_simple(X, numeric_cols, categorical_cols)
after = X_imp.isna().sum().sum()

print(f"\nTotal missing values before imputation: {before}")
print(f"Total missing values after imputation: {after}")