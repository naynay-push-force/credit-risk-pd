# python -m src.features.feature_engineering

import numpy as np
import pandas as pd

DAYS_EMPLOYED_SENTINEL = 365243

def add_application_features(
        df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    # Age (days are negative in this data set)
    if "DAYS_BIRTH" in df.columns:
        df['YEARS_BIRTH'] = (-df['DAYS_BIRTH']) / 365.25

    # Employment (handle sentinel properly)
    if "DAYS_EMPLOYED" in df.columns:
        days_emp = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
        df["YEARS_EMPLOYED"] = (-days_emp) / 365.25
        df["DAYS_EMPLOYED_MISSING"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(int)

    # ratios (safe divide)
    if "AMT_INCOME_TOTAL" in df.columns:
        income_safe = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        
        if "AMT_CREDIT" in df.columns:
            df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income_safe
        if "AMT_ANNUITY" in df.columns:
            df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income_safe
    
    if "AMT_CREDIT" in df.columns:
        credit_safe = df["AMT_CREDIT"].replace(0, np.nan)

        if "AMT_GOODS_PRICE" in df.columns:
            df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / credit_safe

    for col in ["EXT_SOURCE_1", "EXT_SOURCE_3"]:
        if col in df.columns:
            df[f"{col}_MISSING"] = df[col].isna().astype(int)

    return df
