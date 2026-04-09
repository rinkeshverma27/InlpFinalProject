import pandas as pd
import torch
from transformers import pipeline
import sys
import os

from pathlib import Path
# Define paths locally
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NEWS_RAW_DIR = PROJECT_ROOT / "data" / "news" / "raw"
NEWS_PROCESSED_DIR = PROJECT_ROOT / "data" / "news" / "processed"

# Ensure directories exist
NEWS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def run_hindi_sentiment():
    print("Initializing Evaluator for Hindi/Hinglish Sentiment (MuRIL/IndicBERT)...")
    
    device = 0 if torch.cuda.is_available() else -1
    # For Stage 0 evaluate MuRIL. Can switch to ai4bharat/indic-bert if needed.
    # Note: Using a generic sentiment model or a fine-tuned MuRIL if available.
    # We will use an available multilingual/Hindi model for demonstration of the pipeline.
    # Load the newly trained Custom MuRIL model
    model_path = str(PROJECT_ROOT / "models" / "muril_financial_sentiment_v1")
    print(f"Selected Custom Model: {model_path}")
    
    try:
        sentiment_pipeline = pipeline(
            "text-classification", 
            model=model_path, 
            device=device
        )
    except Exception as e:
        print(f"Error loading custom model: {e}")
        print("Falling back to lxyuan/distilbert-base-multilingual-cased-sentiments-student")
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
            device=device
        )

    input_csv = NEWS_RAW_DIR / "news_articles_timestamped.csv"
    output_csv = NEWS_PROCESSED_DIR / "hi_sentiment.csv"

    if not input_csv.exists():
        print(f"Input file {input_csv} not found. Operating on dummy data for testing.")
        data = [
            {"timestamp": "2026-03-15 08:00:00", "ticker": "INFY", "source": "Navbharat Times", "language": "hi", "headline": "इन्फोसिस ने एक प्रमुख यूरोपीय ग्राहक के साथ शानदार सौदा हासिल किया।"},
            {"timestamp": "2026-03-15 08:05:00", "ticker": "ICICIBANK", "source": "Amar Ujala", "language": "hi", "headline": "आंतरिक प्रबंधन विवाद की खबरों के बाद आईसीआईसीआई बैंक के शेयर लुढ़के।"},
            {"timestamp": "2026-03-15 08:10:00", "ticker": "BHARTIARTL", "source": "Zee Business Hindi", "language": "hi", "headline": "Airtel ne plans mehangi kar di, jiske baad stock me jabardast tezi aayi."}
        ]
        df = pd.DataFrame(data)
    else:
        print(f"Reading from {input_csv}")
        df = pd.read_csv(input_csv)
        # Filter for Hindi articles
        if 'language' in df.columns:
            df = df[df['language'] == 'hi'].copy()
        
        if df.empty:
            print("No Hindi articles found in the input CSV.")
            return

    print("Running inference...")
    results = []
    
    headlines = df['headline'].tolist()
    batch_results = sentiment_pipeline(headlines)
    results.extend(batch_results)

    # Convert model labels to a continuous float representation [-1.0 to 1.0]
    # This is required for the regression-based T-LSTM architecture
    def label_to_score(res):
        lbl = res['label'].upper()
        confidence = res['score']
        
        if 'LABEL_1' in lbl or 'POSITIVE' in lbl:
            return float(confidence)  # [0.0 to 1.0]
        elif 'LABEL_0' in lbl or 'NEGATIVE' in lbl:
            return float(-confidence) # [-1.0 to 0.0]
        else:
            return 0.0 # Neutral

    # The blueprint requires 'date' (which we take from timestamp), 'ticker', 'source', 'sentiment_score', 'model_confidence'
    # We rename 'timestamp' to 'date' or keep it as 'timestamp' as needed. The schema states 'date' or 'timestamp'
    
    output_df = pd.DataFrame({
        'timestamp': df['timestamp'],
        'ticker': df['ticker'],
        'source': df['source'],
        'sentiment_score': [round(label_to_score(res), 4) for res in results],
        'model_confidence': [round(res['score'], 4) for res in results]
    })

    output_df.to_csv(output_csv, index=False)
    print(f"Successfully processed {len(output_df)} Hindi/Hinglish articles. Saved to {output_csv}")

if __name__ == "__main__":
    run_hindi_sentiment()
