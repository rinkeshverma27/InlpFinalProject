"""
preprocess_hindi.py
Cleans and normalizes Hindi/Hinglish financial news headlines for MuRIL inference.

Usage:
    python src/preprocessing/preprocess_hindi.py \
        --input  data/hindi_news/hindi_news_sample.csv \
        --output data/hindi_news/hindi_news_cleaned.csv
"""

import re
import argparse
import pandas as pd
from pathlib import Path


def clean_hindi_text(text: str) -> str:
    """Clean a single Hindi/Hinglish headline for MuRIL tokenization."""
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # 4. Keep Devanagari (Unicode 0900-097F), Latin alphanumeric, digits,
    #    basic punctuation, and whitespace. Remove everything else.
    text = re.sub(r'[^\u0900-\u097F\u0020-\u007Ea-zA-Z0-9\s।,.!?%₹$\-\']', '', text)

    # 5. Normalize multiple spaces / newlines into a single space
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Remove leading/trailing punctuation-only residues
    text = text.strip('.,!?;: ')

    return text


def preprocess_hindi_news(input_path: str, output_path: str):
    """Read raw Hindi news CSV, clean headlines, and save."""
    input_csv = Path(input_path)
    output_csv = Path(output_path)

    if not input_csv.exists():
        print(f"ERROR: Input file not found: {input_csv}")
        return

    print(f"Reading raw Hindi news from {input_csv} ...")
    df = pd.read_csv(input_csv)

    required_cols = ['timestamp', 'ticker', 'source', 'language', 'headline']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return

    original_count = len(df)

    # Filter to Hindi only (safety check)
    df = df[df['language'] == 'hi'].copy()
    print(f"  Filtered to {len(df)} Hindi articles (from {original_count} total rows).")

    # Clean headlines
    df['headline_raw'] = df['headline']  # keep original for reference
    df['headline'] = df['headline'].apply(clean_hindi_text)

    # Drop rows where cleaning produced empty string
    empty_mask = df['headline'].str.strip() == ''
    if empty_mask.any():
        print(f"  Dropping {empty_mask.sum()} rows with empty headlines after cleaning.")
        df = df[~empty_mask]

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"Successfully cleaned {len(df)} Hindi headlines.")
    print(f"Saved to {output_csv}")

    # Preview
    print("\nSample cleaned data:")
    print(df[['timestamp', 'ticker', 'headline']].head(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Hindi financial news")
    parser.add_argument("--input", type=str,
                        default="data/hindi_news/hindi_news_sample.csv",
                        help="Path to raw Hindi news CSV")
    parser.add_argument("--output", type=str,
                        default="data/hindi_news/hindi_news_cleaned.csv",
                        help="Path to save cleaned CSV")
    args = parser.parse_args()

    preprocess_hindi_news(args.input, args.output)
