import pandas as pd
from typing import Tuple, List

def split_X_y(
        df: pd.DataFrame, 
        target_col: str = "TARGET"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates features and target.
    
    This function defines the boundary between:
    - data we can observe at inference time (X)
    - the outcome we are trying to predict (y)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    return X, y


def identify_feature_types(
        X: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical feature columns.
    
    Numeric features:
    - int, float
    
    Categorical features:
    - object, category
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    
    return numeric_cols, categorical_cols

def impute_missing_simple(
        X: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Simple baseline imputation using pandas:
    - numeric: fill with median
    - categorical: fill with mode
    
    This is NOT the final imputation strategy (sklearn will do it later),
    but it makes the idea concrete and testable."""
    X_imp = X.copy()

    # Impute numeric columns with median
    for col in numeric_cols:
        median = X_imp[col].median()
        X_imp[col] = X_imp[col].fillna(median)

    # Impute categorical columns with mode
    for col in categorical_cols:
        mode = X_imp[col].mode(dropna=True)
        if len(mode) > 0:
            fill_value = mode.iloc[0]
        else:
            fill_value = "Missing"
        X_imp[col] = X_imp[col].fillna(fill_value)

    return X_imp
