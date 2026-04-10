#!/usr/bin/env python3
"""
scripts/ingest_data.py — Master Data Ingestion, Auto-Tagging, & Merger

PURPOSE:
1. OHLCV: Splits '20year_train.csv' into individual stock files (e.g., RELIANCE.csv).
2. NORMALIZE: Fixes headers and formats for all 4 news sources.
3. AUTO-TAG: Scans the 153k context headlines for tickers.
4. SUMMARIZE: Truncates/Cleans long article bodies for the LSTM.
5. MERGE: Combines all news into data/raw/news/final_news_merged.csv.
"""

import pandas as pd
from pathlib import Path
import os
import re

# Ticker Keyword Map for Auto-Tagging
TICKER_MAP = {
    r'RELIANCE': 'RELIANCE',
    r'HDFC': 'HDFCBANK',
    r'INFOSYS|INFY': 'INFY',
    r'ICICI': 'ICICIBANK',
    r'SBI|STATE BANK': 'SBIN',
    r'TCS|TATA CONSULTANCY': 'TCS',
    r'KOTAK': 'KOTAKBANK',
    r'AXIS': 'AXISBANK',
    r'BHARTI|AIRTEL': 'BHARTIARTL',
    r'BAJAJ FIN': 'BAJFINANCE',
    r'ADANI': 'ADANIENT',
    r'WIPRO': 'WIPRO',
    r'TATA MOTOR': 'TATAMOTORS'
}

def clean_and_tag(df, source_name):
    """Normalize headers, tag tickers, and clean text."""
    # 1. Normalize Header Case
    df.columns = df.columns.str.strip().str.lower()
    
    # 2. Map common variants to standard 'ticker' and 'body'
    mapping = {
        'symbol': 'ticker',
        'datepublished': 'datetime',
        'timestamp': 'datetime',
        'articlebody': 'body',
        'summary': 'body',    # For context file
        'description': 'body'  # Fallback
    }
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
            
    # 3. Auto-Tag Tickers from Headline
    if 'ticker' not in df.columns: df['ticker'] = ''
    df['ticker'] = df['ticker'].fillna('')
    mask = (df['ticker'] == '')
    
    for pattern, ticker in TICKER_MAP.items():
        found = df['headline'].str.contains(pattern, case=False, na=False, regex=True)
        df.loc[mask & found, 'ticker'] = ticker
        
    # 4. Clean Body & Headline (Remove newlines, truncate to 500 chars)
    if 'body' in df.columns:
        df['body'] = df['body'].str.replace('\n', ' ', regex=False).str.slice(0, 500)
    if 'headline' in df.columns:
        df['headline'] = df['headline'].str.replace('\n', ' ', regex=False)
        
    # 5. Add Metadata
    df['source'] = source_name
    return df

def run_ingestion():
    raw_news_dir = Path("data/raw/news")
    raw_ohlcv_dir = Path("data/raw/ohlcv")
    
    # --- 1. Split Master OHLCV ---
    master_ohlcv = raw_ohlcv_dir / "20year_train.csv"
    if master_ohlcv.exists():
        print("Splitting 20-year OHLCV Master File...")
        p_df = pd.read_csv(master_ohlcv)
        if 'Ticker' in p_df.columns:
            p_df['Ticker'] = p_df['Ticker'].str.replace('.NS', '', regex=False)
            for t in p_df['Ticker'].unique():
                ticker_df = p_df[p_df['Ticker'] == t]
                output_path = raw_ohlcv_dir / f"{t.upper()}.csv"
                ticker_df.to_csv(output_path, index=False)
        print(f"  [OK] OHLCV splitting complete.")

    # --- 2. Process all News Sources ---
    sources = {
        "context": raw_news_dir / "nifty_context_part1.csv",
        "legacy":  raw_news_dir / "Nifty50_news_data(2020Jan_2024April).csv",
        "hindi":   raw_news_dir / "hindi_articles_all.csv"
    }
    
    merged_frames = []
    for name, path in sources.items():
        if path.exists():
            print(f"Processing source: {path.name}...")
            df = pd.read_csv(path, low_memory=False)
            df = clean_and_tag(df, name)
            
            # Keep only rows that successfully got a Ticker
            tagged_df = df[df['ticker'] != '']
            print(f"  -> {len(tagged_df)}/{len(df)} articles tagged with tickers.")
            
            # Subselect key columns for merger
            cols_to_keep = [c for c in ['datetime', 'ticker', 'headline', 'body', 'source'] if c in tagged_df.columns]
            merged_frames.append(tagged_df[cols_to_keep])

    # --- 3. Merge and Save ---
    if merged_frames:
        final_df = pd.concat(merged_frames, ignore_index=True)
        # Final formatting
        final_df['datetime'] = pd.to_datetime(final_df['datetime'], errors='coerce')
        final_df = final_df.dropna(subset=['datetime'])
        
        output_path = raw_news_dir / "final_news_merged.csv"
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ MERGE COMPLETE: {output_path}")
        print(f"Total News Records: {len(final_df):,}")
    else:
        print("\n⚠️ No news sources found to merge.")

if __name__ == "__main__":
    run_ingestion()
