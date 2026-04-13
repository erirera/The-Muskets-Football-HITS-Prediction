import pandas as pd
import numpy as np
import re

def clean_dataset(input_path, output_path):
    # 1. Load Data
    df = pd.read_csv(input_path, low_memory=False)

    # 2. Rename Columns
    df = df.rename(columns={'↓OVA': 'OVA'})

    # 3. String formatting - Strip leading/trailing whitespaces and newlines
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # 4. Height - parse to int (cm)
    def parse_height(h):
        if pd.isna(h):
            return np.nan
        h = str(h)
        if 'cm' in h:
            return int(h.replace('cm', '').strip())
        elif "'" in h:  # Format like 5'9"
            parts = h.split("'")
            feet = int(parts[0])
            inches = int(parts[1].replace('"', '')) if len(parts) > 1 and parts[1].replace('"', '') != '' else 0
            return int(round(feet * 30.48 + inches * 2.54))
        return np.nan

    if 'Height' in df.columns:
        df['Height'] = df['Height'].apply(parse_height)

    # 5. Weight - parse to int (kg)
    def parse_weight(w):
        if pd.isna(w):
            return np.nan
        w = str(w)
        if 'kg' in w:
            return int(w.replace('kg', '').strip())
        elif 'lbs' in w:
            return int(round(int(w.replace('lbs', '').strip()) * 0.45359237))
        return np.nan

    if 'Weight' in df.columns:
        df['Weight'] = df['Weight'].apply(parse_weight)

    # 6. Currency - Value, Wage, Release Clause parse
    def parse_currency(val):
        if pd.isna(val) or val == 'nan':
            return 0
        val = str(val).replace('€', '').strip()
        if val.endswith('M'):
            return int(float(val[:-1]) * 1000000)
        elif val.endswith('K'):
            return int(float(val[:-1]) * 1000)
        else:
            try:
                return int(float(val))
            except:
                return 0

    for col in ['Value', 'Wage', 'Release Clause']:
        if col in df.columns:
            df[col] = df[col].apply(parse_currency)

    # 7. Ratings - W/F, SM, IR parse
    for col in ['W/F', 'SM', 'IR']:
        if col in df.columns:
            # Remove star and cast to int, ignoring nans
            df[col] = df[col].astype(str).str.replace('★', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # 8. Hits
    def parse_hits(h):
        if pd.isna(h) or str(h).strip() == 'nan':
            return np.nan
        h = str(h)
        if h.endswith('K'):
            return int(float(h[:-1]) * 1000)
        return int(h)

    if 'Hits' in df.columns:
        df['Hits'] = df['Hits'].apply(parse_hits)

    # 9. Contract Breakdown
    def parse_contract(c):
        if pd.isna(c) or c == 'nan':
            return pd.Series([np.nan, np.nan, 'Unknown'])
        c = str(c)
        if '~' in c:
            parts = c.split('~')
            return pd.Series([parts[0].strip(), parts[1].strip(), 'Active'])
        elif 'On Loan' in c:
            # Try extracting year from 'Jun 30, 2021 On Loan'
            match = re.search(r'\d{4}', c)
            end_year = match.group(0) if match else np.nan
            return pd.Series([np.nan, end_year, 'On Loan'])
        elif 'Free' in c:
            return pd.Series([np.nan, np.nan, 'Free'])
        else:
            return pd.Series([np.nan, np.nan, 'Unknown'])

    if 'Contract' in df.columns:
        df[['Contract Start', 'Contract End', 'Contract Status']] = df['Contract'].apply(parse_contract)

    # Replace 'nan' strings introduced by strip on object columns with actual NaN
    df.replace('nan', np.nan, inplace=True)

    # Save output
    df.to_csv(output_path, index=False)
    print(f"Data successfully cleaned and saved to {output_path}")
    print("\nUpdated Schema:")
    print(df.info())

if __name__ == '__main__':
    clean_dataset('c:/Users/delef/.gemini/antigravity/Musket Analysis/Muskets_data.csv', 'c:/Users/delef/.gemini/antigravity/Musket Analysis/Muskets_data_cleaned.csv')
