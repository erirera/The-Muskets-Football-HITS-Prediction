import csv
import sys

file_path = '../Muskets_data.csv'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        cols = ['Club', 'Contract', 'Height', 'Weight', 'Value', 'Wage', 'Release Clause', 'W/F', 'SM', 'IR', 'Hits', 'Joined', 'Loan Date End']
        indices = {h: header.index(h) for h in cols if h in header}
        
        print('\n--- Samples for specific columns ---')
        for i in range(10):
            row = next(reader)
            row_dict = {h: row[idx] for h, idx in indices.items()}
            print(str(row_dict).encode('utf-8').decode('ascii', 'ignore'))
except Exception as e:
    print('Error:', e)
