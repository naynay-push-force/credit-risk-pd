import pandas as pd

def split_X_y(df: pd.DataFrame, target_col: str = "TARGET"):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y