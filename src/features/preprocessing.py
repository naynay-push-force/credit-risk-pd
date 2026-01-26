import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# 1. Split features and target
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

# 2. Identify feature types
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

# 3. Simple imputation
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

# 4. Train-validation split
def train_val_split(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified split so the default rate is similar in train and validation sets.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_val, y_train, y_val

# 5. Fitting and applying imputers separately
def fit_imputers(
        X_train: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str]
) -> Tuple[SimpleImputer, SimpleImputer]:
    """
    Fit imputers on training data only.
    Returns fitted numeric and categorical imputers.
    """
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    num_imputer.fit(X_train[numeric_cols])
    cat_imputer.fit(X_train[categorical_cols])

    return num_imputer, cat_imputer

def apply_imputers(
    X: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    num_imputer: SimpleImputer,
    cat_imputer: SimpleImputer
) -> pd.DataFrame:
    """
    Apply fitted imputers to any dataframe (tran/val/test).
    Returns a DataFrame with the same columns as X.
    """
    X_out = X.copy()

    X_out[numeric_cols] = num_imputer.transform(X_out[numeric_cols])
    X_out[categorical_cols] = cat_imputer.transform(X_out[categorical_cols])

    return X_out