import pandas as pd

df = pd.read_csv('Weather Data (US).csv', nrows=1000)
print(df['PRCP'].isna().sum())
print(df.head())

df['PRCP'] = df['PRCP'].interpolate()
print(df.head())
