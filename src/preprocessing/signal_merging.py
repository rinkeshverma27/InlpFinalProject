import pandas as pd
import json
from pathlib import Path
import sys
import os
import datetime

# Ensure the parent directory is in the path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.paths import NEWS_PROCESSED_DIR, TRUST_WEIGHTS_DIR, PROCESSED_DATA_DIR, HANDSHAKE_CSV

def generate_handshake_csv():
    print("Initializing Handshake CSV Generator (Tier 1 Trust Weight Fusion)...")
    
    en_csv = NEWS_PROCESSED_DIR / "en_sentiment.csv"
    hi_csv = NEWS_PROCESSED_DIR / "hi_sentiment.csv"
    
    dfs = []
    
    # Load English
    if en_csv.exists():
        print(f"Loading English sentiment from {en_csv}")
        df_en = pd.read_csv(en_csv)
        dfs.append(df_en)
    else:
        print(f"WARNING: English sentiment CSV not found at {en_csv}")
        
    # Load Hindi
    if hi_csv.exists():
        print(f"Loading Hindi sentiment from {hi_csv}")
        df_hi = pd.read_csv(hi_csv)
        dfs.append(df_hi)
    else:
        print(f"WARNING: Hindi sentiment CSV not found at {hi_csv}")
        
    if not dfs:
        print("ERROR: No sentiment data available to generate handshake file.")
        return
        
    # Combine all sentiment scores
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Process Sparse Hindi Rule and Trust Weights
    # In a real run, this would load the daily JSON trust weights
    # For now, we apply standard averaging or basic trust weight (1.0)
    
    # Group by Ticker and Timestamp (or Date)
    # The LSTM expects a daily aggregated score per stock
    # For simplicity, we just take the mean sentiment per stock per day for now
    
    print("Applying Trust Weights and Aggregating Data...")
    
    # We assume 'timestamp' has the date. Let's extract 'date' for daily grouping
    df_combined['date'] = pd.to_datetime(df_combined['timestamp']).dt.date
    
    # Aggregate: Mean Sentiment, Mean Confidence per stock per day
    # Next tier will implement dynamic JSON lookup weightings per source.
    df_daily = df_combined.groupby(['date', 'ticker']).agg({
        'sentiment_score': 'mean',
        'model_confidence': 'mean',
        'source': lambda x: list(x) # Keep track of sources used
    }).reset_index()
    
    df_daily['sentiment_score'] = df_daily['sentiment_score'].round(4)
    df_daily['model_confidence'] = df_daily['model_confidence'].round(4)
    
    df_daily.to_csv(HANDSHAKE_CSV, index=False)
    
    print(f"Successfully generated Handshake CSV with {len(df_daily)} daily stock records.")
    print(f"Saved to {HANDSHAKE_CSV}")

if __name__ == "__main__":
    generate_handshake_csv()
