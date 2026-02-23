# python -m src.features.feature_engineering

"""
Feature engineering for Home Credit application-level data.

Design principle:
- Only add features we can justify with either:
  (a) EDA evidence (bad-rate by decile / missingness signal / monotonic trend), or
  (b) domain logic (i.e., leverage / affordability ratios).

Evidence references:
- notebooks/01_eda_application.ipynb
  - EXT_SOURCE_* show strong monotonic risk separation; missingness increases bad rate.
  - Ratio features (credit/income, annuity/income, goods/credit) show clearer monotonic patterns than raw amounts.
  - DAYS_BIRTH and DAYS_EMPLOYED become more interpretable as YEARS_* and show monotone trends with default risk.
"""

import numpy as np
import pandas as pd

# DAYS_EMPLOYED has sentinel 365243 indicating "unknown"
DAYS_EMPLOYED_SENTINEL = 365243

def add_application_features(
        df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    # -- Age / employment tenure --
    # EDA evidence:
    # - Both YEARS_BIRTH and YEARS_EMPLOYED show monotonic bad-rate pattern across deciles.
    if "DAYS_BIRTH" in df.columns:
        df['YEARS_BIRTH'] = (-df['DAYS_BIRTH']) / 365.25
    if "DAYS_EMPLOYED" in df.columns:
        days_emp = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
        df["YEARS_EMPLOYED"] = (-days_emp) / 365.25
        df["DAYS_EMPLOYED_MISSING"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(int)

    # -- Affordability / leverage ratios --
    # EDA evidence:
    # - CREDIT_INCOME_RATIO and ANNUITY_INCOME_RATIO show clearer monotone relationships with default than raw features.
    # Domain intuition: 
    # - Higher leverage / repayment burden  -> higher default risk.
    if "AMT_INCOME_TOTAL" in df.columns:
        income_safe = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        
        if "AMT_CREDIT" in df.columns:
            df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income_safe
        if "AMT_ANNUITY" in df.columns:
            df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income_safe
    
    # -- Equity -- 
    # EDA evidence:
    # - GOODS_CREDIT_RATIO shows clear monotone relationship with default more than raw features.
    # Domain intuition:
    # - Higher equity -> lower default risk.
    if "AMT_CREDIT" in df.columns:
        credit_safe = df["AMT_CREDIT"].replace(0, np.nan)

        if "AMT_GOODS_PRICE" in df.columns:
            df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / credit_safe

    # -- Missigness indicators -- 
    # EDA evidence:
    # - EXT_SOURCE_1 / EXT_SOURCE_3 missingness corresponds to higher observed bad rate.
    # Missingness is treated as a signal, so we crease explicit flags 
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_3"]:
        if col in df.columns:
            df[f"{col}_MISSING"] = df[col].isna().astype(int)

    return df
