import pandas as pd
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Split features and target
def split_X_y(
        df: pd.DataFrame,
        target_col: str = "TARGET"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates features and target.

    This function isolates the following:
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
    Identifies categorical and numerical feature columns.

    Numeric features:
    - int, float

    Categorical features:
    - object, category
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    return numeric_cols, categorical_cols

# 3. Train-validation split
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

# 4. ColumnTransformer implementation
def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    Builds an sklearn ColumnTransformer that:
    - Imputes & scales numeric features
    - Imputes + one-hot encodes categorical features

    Note:
    - This function does not fit anything.
    - Fitting happens on training data only: preprocessor.fit(X_train)
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def main() -> None:
    """
    Smoke check for the live preprocessing path: split -> type -> fit/transform.

    Run directly (`python -m src.features.preprocessing`) to confirm the module
    still behaves; the asserts fail loudly if a step regresses.
    """
    df = pd.read_csv("data/raw/application_train.csv")

    X, y = split_X_y(df)
    numeric_cols, categorical_cols = identify_feature_types(X)
    assert len(numeric_cols) + len(categorical_cols) == X.shape[1], "feature typing dropped columns"
    print(f"Features: {X.shape[1]} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")

    X_train, X_val, y_train, y_val = train_val_split(X, y)
    assert X_train.shape[0] + X_val.shape[0] == X.shape[0], "split lost rows"
    print(f"Split:    train {X_train.shape[0]:,}  val {X_val.shape[0]:,}")

    # Fit on train only, then transform both -- the ColumnTransformer is the live path.
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    Xt_train = preprocessor.transform(X_train)
    Xt_val = preprocessor.transform(X_val)
    assert Xt_train.shape[0] == X_train.shape[0], "transform changed the row count"
    assert Xt_train.shape[1] == Xt_val.shape[1], "train/val feature widths differ"
    print(f"Encoded:  {Xt_train.shape[1]} features (train {Xt_train.shape[0]:,} / val {Xt_val.shape[0]:,} rows)")

    # A stratified split should keep the default rate stable across train/val.
    assert abs(y_train.mean() - y.mean()) < 0.01, "train default rate drifted"
    assert abs(y_val.mean() - y.mean()) < 0.01, "val default rate drifted"
    print(f"Default rate: overall {y.mean():.4f}  train {y_train.mean():.4f}  val {y_val.mean():.4f}")

    print("OK -- preprocessing smoke check passed.")


if __name__ == "__main__":
    main()
