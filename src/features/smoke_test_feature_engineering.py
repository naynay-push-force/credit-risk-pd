# python -m src.features.smoke_test_feature_engineering

import pandas as pd

from src.features.feature_engineering import add_application_features

df = pd.read_csv("data/raw/application_train.csv")

df = add_application_features(df)

# check columns
print(f"Columns: \n{df.columns}\n")

# check indicator columns are 0/1
cols = ["EXT_SOURCE_1_MISSING", "EXT_SOURCE_3_MISSING", "DAYS_EMPLOYED_MISSING"]
for col in cols:
    print(f"Check on {col}: \n {df[col].head()}\n")
