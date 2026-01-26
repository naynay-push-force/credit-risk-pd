# python -m src.features.smoke_test_preprocessing

import pandas as pd
from src.features.preprocessing import (
    split_X_y, 
    identify_feature_types,
    impute_missing_simple,
    train_val_split,
    fit_imputers,
    apply_imputers,
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

# Train-validation split test
X_train, X_val, y_train, y_val = train_val_split(X, y)
print(f"\nOverall default rate: {y.mean()}")
print(f"Train default rate {y_train.mean()}")
print(f"Validation default rate {y_val.mean()}")
print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")

# Fit and apply imputers test
num_imp, cat_imp = fit_imputers(X_train, numeric_cols, categorical_cols)

train_missing_before = X_train.isna().sum().sum()
val_missing_before = X_val.isna().sum().sum()

X_train_imp = apply_imputers(X_train, numeric_cols, categorical_cols, num_imp, cat_imp)
X_val_imp = apply_imputers(X_val, numeric_cols, categorical_cols, num_imp, cat_imp)

train_missing_after = pd.DataFrame(X_train_imp).isna().sum().sum()
val_missing_after = pd.DataFrame(X_val_imp).isna().sum().sum()

print("Train missing BEFORE:", train_missing_before)
print("Train missing AFTER:", train_missing_after)
print("Val missing BEFORE:", val_missing_before)
print("Val missing AFTER:", val_missing_after)

print("Default rates - overall/train/val:", y.mean(), y_train.mean(), y_val.mean())
