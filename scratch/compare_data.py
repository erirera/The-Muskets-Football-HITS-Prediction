import pandas as pd

df_raw = pd.read_csv('../Muskets_data.csv', low_memory=False)
df_clean = pd.read_csv('../Muskets_data_cleaned.csv', low_memory=False)

cols = ['Name', 'Club', 'Contract', 'Height', 'Weight', 'Value', 'Wage', 'Release Clause', 'W/F', 'Hits']

print('=== RAW DATA ===')
raw_sample = df_raw[cols].head(2)
for idx, row in raw_sample.iterrows():
    for col in cols:
        val = str(row[col])
        if col == 'Club':
            val = repr(val)
        print(f"{col}: {val}")
    print('--')

clean_cols = ['Name', 'Club', 'Contract Start', 'Contract End', 'Contract Status', 'Height', 'Weight', 'Value', 'Wage', 'Release Clause', 'W/F', 'Hits']
print('\n=== CLEANED DATA ===')
clean_sample = df_clean[clean_cols].head(2)
for idx, row in clean_sample.iterrows():
    for col in clean_cols:
        val = str(row[col])
        if col == 'Club':
            val = repr(val)
        print(f"{col}: {val}")
    print('--')
