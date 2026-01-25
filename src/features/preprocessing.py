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