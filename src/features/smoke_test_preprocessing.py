import pandas as pd
from src.features.preprocessing import split_X_y

df = pd.read_csv('data/raw/application_train.csv')
X, y = split_X_y(df)

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print(y.value_counts(normalize=True))