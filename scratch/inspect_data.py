import pandas as pd
import numpy as np

df = pd.read_csv('../Muskets_data.csv', low_memory=False)

print(f"Dataset shape: {df.shape}")
print("\n--- Column Types and Missing Values ---")
info = pd.DataFrame({
    'Type': df.dtypes,
    'Nulls': df.isnull().sum(),
    'Null_%': (df.isnull().sum() / len(df)) * 100
})
print(info[info['Nulls'] > 0].sort_values(by='Nulls', ascending=False).to_markdown())

print("\n--- Sample of Columns that Might Need Cleaning ---")
cols_to_check = ['Joined', 'Contract', 'Height', 'Weight', 'Value', 'Wage', 'Release Clause', 'W/F', 'SM', 'IR', 'Hits', 'Club']
for col in cols_to_check:
    if col in df.columns:
        print(f"\n{col} samples:")
        print(df[col].dropna().head(5).tolist())
