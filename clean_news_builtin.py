import csv
from pathlib import Path
import sys

csv.field_size_limit(sys.maxsize)

def clean_scraped_file(input_path: str, output_path: str):
    print(f"Reading {input_path}...")
    valid_rows = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8', newline='') as fout:
        
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        
        try:
            header = next(reader)
        except StopIteration:
            return
            
        # Map 'timestamp' to 'datetime', and 'summary' to 'body'
        new_header = []
        body_idx = -1
        headline_idx = -1
        for i, col in enumerate(header):
            col_lower = col.strip().lower()
            if col_lower == 'timestamp':
                new_header.append('datetime')
            elif col_lower == 'summary':
                new_header.append('body')
                body_idx = i
            else:
                new_header.append(col_lower)
            if col_lower == 'headline':
                headline_idx = i
                
        writer.writerow(new_header)
        
        for row in reader:
            if len(row) != len(header):
                continue  # Skip broken rows just in case
            
            # Clean headline newlines
            if headline_idx != -1 and row[headline_idx]:
                row[headline_idx] = row[headline_idx].replace('\n', ' ').replace('\r', '')
            
            # Truncate massive body text and remove newlines to make CSV 1 row per line physically
            if body_idx != -1 and row[body_idx]:
                row[body_idx] = row[body_idx][:500].replace('\n', ' ').replace('\r', '')
                
            writer.writerow(row)
            valid_rows += 1
            
    print(f"Cleaned file saved to {output_path} (Valid articles: {valid_rows})")

if __name__ == '__main__':
    raw_dir = Path("data/raw/news")
    input_file = raw_dir / "nifty_context_part1.csv"
    output_file = raw_dir / "cleaned_nifty_part1.csv"
    
    if input_file.exists():
        clean_scraped_file(input_file, output_file)
        # Rename original to a backup so pipeline ignores it
        backup_file = raw_dir / "nifty_context_part1.csv.bak"
        input_file.rename(backup_file)
        print(f"Backed up original to {backup_file.name}")
