import pandas as pd
import torch
from transformers import pipeline
import sys
import os

# Ensure the parent directory is in the path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.paths import NEWS_RAW_DIR, NEWS_PROCESSED_DIR

def run_english_sentiment():
    print("Initializing FinBERT Pipeline for English Sentiment...")
    
    # Use FP16 to stay under 3.5GB VRAM limit as per blueprint
    device = 0 if torch.cuda.is_available() else -1
    model_name = "yiyanghkust/finbert-tone"
    
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=model_name, 
            device=device,
            model_kwargs={"torch_dtype": torch.float16} if device == 0 else {}
        )
    except Exception as e:
        print(f"Failed to load FinBERT: {e}")
        return

    input_csv = NEWS_RAW_DIR / "news_articles_timestamped.csv"
    output_csv = NEWS_PROCESSED_DIR / "en_sentiment.csv"

    if not input_csv.exists():
        print(f"Input file {input_csv} not found. Operating on dummy data for testing.")
        data = [
            {"timestamp": "2026-03-15 08:00:00", "ticker": "RELIANCE", "source": "Economic Times", "language": "en", "headline": "Reliance reports massive jump in quarterly profits, beating all estimates."},
            {"timestamp": "2026-03-15 08:05:00", "ticker": "TCS", "source": "Moneycontrol", "language": "en", "headline": "TCS misses revenue projections due to low global IT spending."},
            {"timestamp": "2026-03-15 08:10:00", "ticker": "HDFCBANK", "source": "LiveMint", "language": "en", "headline": "HDFC Bank announces stable interest rates for the upcoming quarter."}
        ]
        df = pd.DataFrame(data)
    else:
        print(f"Reading from {input_csv}")
        df = pd.read_csv(input_csv)
        # Filter for English articles
        if 'language' in df.columns:
            df = df[df['language'] == 'en'].copy()
            
        if df.empty:
            print("No English articles found in the input CSV.")
            return

    print("Running batch inference (batch_size=16)...")
    results = []
    
    # Process in batches
    batch_size = 16
    headlines = df['headline'].tolist()
    
    with torch.cuda.amp.autocast() if device == 0 else open(os.devnull, 'w'):
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i + batch_size]
            batch_results = sentiment_pipeline(batch)
            results.extend(batch_results)

    # Convert FinBERT categorical labels to continuous float representation [-1.0 to 1.0]
    def label_to_score(res):
        lbl = res['label'].upper()
        confidence = res['score']
        
        if 'POSITIVE' in lbl:
            return float(confidence)  # [0.0 to 1.0]
        elif 'NEGATIVE' in lbl:
            return float(-confidence) # [-1.0 to 0.0]
        else:
            return 0.0 # Neutral

    output_df = pd.DataFrame({
        'timestamp': df['timestamp'],
        'ticker': df['ticker'],
        'source': df['source'],
        'sentiment_score': [round(label_to_score(res), 4) for res in results],
        'model_confidence': [round(res['score'], 4) for res in results]
    })

    output_df.to_csv(output_csv, index=False)
    print(f"Successfully processed {len(output_df)} English articles. Saved to {output_csv}")

if __name__ == "__main__":
    run_english_sentiment()
