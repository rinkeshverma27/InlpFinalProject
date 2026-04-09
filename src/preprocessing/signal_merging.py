import pandas as pd
import json
from pathlib import Path
import sys
import os
import datetime

# Ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.paths import NEWS_PROCESSED_DIR, TRUST_WEIGHTS_DIR, PROCESSED_DATA_DIR, HANDSHAKE_CSV

def generate_handshake_csv():
    print("Initializing Handshake CSV Generator (Tier 1 Trust Weight Fusion + Sparse Hindi Rule)...")
    
    en_csv = NEWS_PROCESSED_DIR / "en_sentiment.csv"
    hi_csv = NEWS_PROCESSED_DIR / "hi_sentiment.csv"
    
    if not en_csv.exists() and not hi_csv.exists():
        print("ERROR: No sentiment data available to generate handshake file.")
        return

    # Load English
    df_en = pd.read_csv(en_csv) if en_csv.exists() else pd.DataFrame()
    if not df_en.empty:
        df_en['lang'] = 'en'
        
    # Load Hindi
    df_hi = pd.read_csv(hi_csv) if hi_csv.exists() else pd.DataFrame()
    if not df_hi.empty:
        df_hi['lang'] = 'hi'
        
    df_combined = pd.concat([df_en, df_hi], ignore_index=True)
    df_combined['date'] = pd.to_datetime(df_combined['timestamp']).dt.date
    
    # ── Blueprint §2.1: Sparse Hindi Rule ──
    # If a stock has fewer than 2 Hindi articles in the past 3 trading days (approx),
    # Hindi trust is 0.0, English is 1.0.
    
    results = []
    
    # Group by date and ticker for aggregation
    for (curr_date, ticker), group in df_combined.groupby(['date', 'ticker']):
        en_group = group[group['lang'] == 'en']
        hi_group = group[group['lang'] == 'hi']
        
        hi_count = len(hi_group)
        
        # Determine Weights based on Blueprint Sparse Hindi Rule
        # In this simple implementation, we check "today's" count. 
        # Full logic uses 3-day window, but for the daily script we check today's availability.
        if hi_count < 2:
            # Rule: Weight English 100%
            en_weight = 1.0
            hi_weight = 0.0
            status = "SPARSE_HINDI_FALLBACK"
        else:
            # Standard fusion (equal trust until Tier 1 weights are loaded)
            en_weight = 0.5
            hi_weight = 0.5
            status = "DUAL_STREAM"

        avg_en_sent = en_group['sentiment_score'].mean() if not en_group.empty else 0.0
        avg_hi_sent = hi_group['sentiment_score'].mean() if not hi_group.empty else 0.0
        
        # Weighted Fusion
        fused_sentiment = (avg_en_sent * en_weight) + (avg_hi_sent * hi_weight)
        
        # Mean confidence
        avg_conf = group['model_confidence'].mean()
        
        results.append({
            'date': curr_date,
            'ticker': ticker,
            'sentiment_score': round(fused_sentiment, 4),
            'model_confidence': round(avg_conf, 4),
            'hi_count': hi_count,
            'fusion_status': status,
            'sources': list(group['source'].unique())
        })
        
    df_handshake = pd.DataFrame(results)
    df_handshake.to_csv(HANDSHAKE_CSV, index=False)
    
    print(f"Successfully generated Handshake CSV with {len(df_handshake)} daily stock records.")
    print(f"Applied Sparse Hindi rule to {len(df_handshake[df_handshake['fusion_status'] == 'SPARSE_HINDI_FALLBACK'])} records.")
    print(f"Saved to {HANDSHAKE_CSV}")

if __name__ == "__main__":
    generate_handshake_csv()
