import csv
import os
from datetime import datetime

def parse_date(date_str):
    if not date_str:
        return None
    if len(date_str) >= 10:
        if date_str[4] == '-' and date_str[7] == '-':
            return date_str[:10]
    return date_str

def process_pricing():
    # Including OneYearMarketData.csv just in case, though it's a diff match to 1year-test
    input_files = ['20year_train.csv', '1year-test.csv', 'OneYearMarketData.csv']
    output_file = 'data/master_ohlcv_dataset.csv'
    
    seen = set() # To avoid duplicates (Ticker, Date)
    count = 0
    
    with open(output_file, 'w', newline='', encoding='utf-8') as fout:
        fieldnames = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        
        for file_name in input_files:
            if not os.path.exists(file_name):
                print(f"Warning: {file_name} not found.")
                continue
                
            print(f"Processing {file_name}...")
            with open(file_name, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_ticker = row.get('Ticker')
                    if not raw_ticker:
                        continue
                        
                    # Clean ticker: remove suffix and whitespace
                    ticker = raw_ticker.replace('.NS', '').strip().upper()
                    
                    raw_date = row.get('Date')
                    date = parse_date(raw_date)
                    
                    if not date:
                        continue
                        
                    key = (ticker, date)
                    if key in seen:
                        continue
                    
                    seen.add(key)
                    
                    try:
                        writer.writerow({
                            'Date': date,
                            'Ticker': ticker,
                            'Open': row.get('Open'),
                            'High': row.get('High'),
                            'Low': row.get('Low'),
                            'Close': row.get('Close'),
                            'Volume': row.get('Volume')
                        })
                        count += 1
                    except Exception as e:
                        print(f"Error writing row {key}: {e}")

    print(f"Finished! Total UNIQUE master pricing rows: {count}")

if __name__ == "__main__":
    process_pricing()
