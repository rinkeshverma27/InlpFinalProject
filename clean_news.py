import pandas as pd
from pathlib import Path

def clean_scraped_file(input_path: str, output_path: str):
    print(f"Reading {input_path}...")
    # pandas handles multi-line quotes perfectly
    df = pd.read_csv(input_path, encoding='utf-8')
    
    print(f"Original shape: {df.shape}")
    
    # 1. Rename 'timestamp' to 'datetime' for news_loader.py
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'datetime'})
        
    # 2. Rename 'summary' to 'body' and truncate it to save space (since only headline is scored)
    if 'summary' in df.columns:
        df = df.rename(columns={'summary': 'body'})
        # Truncate the massive full article to just the first 500 characters
        df['body'] = df['body'].str.slice(0, 500)
        # Remove any lingering newlines so the output CSV looks clean/aligned in text editors
        df['body'] = df['body'].str.replace('\n', ' ', regex=False).str.replace('\r', '', regex=False)
        
    # Remove any newlines in headlines as well
    df['headline'] = df['headline'].str.replace('\n', ' ', regex=False).str.replace('\r', '', regex=False)

    df.to_csv(output_path, index=False)
    print(f"Cleaned file saved to {output_path} (Shape: {df.shape})")

if __name__ == '__main__':
    raw_dir = Path("data/raw/news")
    # Clean part 1
    input_file = raw_dir / "nifty_context_part1.csv"
    output_file = raw_dir / "cleaned_nifty_part1.csv"
    
    if input_file.exists():
        clean_scraped_file(input_file, output_file)
        # We can rename the original to a backup so the pipeline doesn't read the dirty one
        input_file.rename(raw_dir / "nifty_context_part1.csv.bak")
        print(f"Backed up original to {input_file.name}.bak")
