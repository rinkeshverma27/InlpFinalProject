"""
analyzer_hi.py
Hindi/Hinglish financial sentiment analysis using MuRIL.

Pipeline:
  1. Tries to load a fine-tuned MuRIL from models/muril_financial_sentiment_v1
  2. If not found, automatically triggers fine-tuning on synthetic data
  3. Falls back to multilingual distilBERT if all else fails
  4. Outputs continuous sentiment scores in [-1.0, +1.0]

Usage:
    python src/sentiment/analyzer_hi.py \
        --input  data/hindi_news/hindi_news_cleaned.csv \
        --output data/hindi_news/hi_sentiment.csv
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Paths
FINETUNED_MODEL_DIR = PROJECT_ROOT / "models" / "muril_financial_sentiment_v1"
SYNTHETIC_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_hindi_financial_train.csv"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "scripts" / "train_sentiment_model.py"
SYNTH_SCRIPT = PROJECT_ROOT / "src" / "sentiment" / "synthetic_data_gen.py"

# Defaults
DEFAULT_INPUT = "data/hindi_news/hindi_news_cleaned.csv"
DEFAULT_OUTPUT = "data/hindi_news/hi_sentiment.csv"


def ensure_finetuned_model_exists():
    """If the fine-tuned MuRIL model is missing, generate synthetic data and train it."""
    
    config_json = FINETUNED_MODEL_DIR / "config.json"
    if config_json.exists():
        print(f"Fine-tuned MuRIL model found at {FINETUNED_MODEL_DIR}")
        return True

    print("=" * 70)
    print("Fine-tuned MuRIL model NOT found. Auto-triggering training pipeline...")
    print("=" * 70)

    # Step A: Generate synthetic training data if it doesn't exist
    if not SYNTHETIC_DATA_PATH.exists():
        print("\n[Step A] Generating synthetic Hindi financial training data...")
        try:
            subprocess.run(
                [sys.executable, str(SYNTH_SCRIPT)],
                check=True,
                cwd=str(PROJECT_ROOT),
            )
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Synthetic data generation failed: {e}")
            return False
        except FileNotFoundError:
            print(f"WARNING: Synthetic data gen script not found at {SYNTH_SCRIPT}")
            return False
    else:
        print(f"\n[Step A] Synthetic training data already exists at {SYNTHETIC_DATA_PATH}")

    # Step B: Fine-tune MuRIL
    print("\n[Step B] Fine-tuning MuRIL on synthetic Hindi financial data...")
    print("         This may take 5-15 minutes on CPU...")
    try:
        subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            check=True,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.CalledProcessError as e:
        print(f"WARNING: MuRIL fine-tuning failed: {e}")
        return False
    except FileNotFoundError:
        print(f"WARNING: Training script not found at {TRAIN_SCRIPT}")
        return False

    # Verify
    if config_json.exists():
        print(f"\nFine-tuning complete! Model saved to {FINETUNED_MODEL_DIR}")
        return True
    else:
        print("\nWARNING: Training completed but model config not found.")
        return False


def load_sentiment_pipeline():
    """Load the best available sentiment pipeline, with cascading fallback."""
    
    device = 0 if torch.cuda.is_available() else -1

    # Tier 1: Try fine-tuned MuRIL
    finetuned_exists = ensure_finetuned_model_exists()
    
    if finetuned_exists:
        try:
            print(f"\nLoading fine-tuned MuRIL from {FINETUNED_MODEL_DIR}...")
            sent_pipeline = pipeline(
                "text-classification",
                model=str(FINETUNED_MODEL_DIR),
                tokenizer=str(FINETUNED_MODEL_DIR),
                device=device,
            )
            print("Successfully loaded fine-tuned MuRIL for Hindi sentiment.")
            return sent_pipeline, "muril-finetuned"
        except Exception as e:
            print(f"WARNING: Failed to load fine-tuned model: {e}")

    # Tier 2: Fallback to multilingual distilBERT
    print("\nFalling back to multilingual distilBERT sentiment model...")
    fallback_model = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    try:
        sent_pipeline = pipeline(
            "sentiment-analysis",
            model=fallback_model,
            device=device,
        )
        print(f"Loaded fallback model: {fallback_model}")
        return sent_pipeline, "distilbert-multilingual"
    except Exception as e:
        print(f"CRITICAL: Even fallback model failed to load: {e}")
        raise RuntimeError("No sentiment model could be loaded.")


def label_to_score(result: dict) -> float:
    """Convert model output to continuous [-1.0, +1.0] score."""
    label = result['label'].upper()
    confidence = result['score']

    if 'POSITIVE' in label or label == 'LABEL_1':
        return round(float(confidence), 4)       # [0.0 to +1.0]
    elif 'NEGATIVE' in label or label == 'LABEL_0':
        return round(float(-confidence), 4)      # [-1.0 to 0.0]
    else:
        return 0.0  # Neutral


def run_hindi_sentiment(input_path: str, output_path: str):
    """Main entry point: preprocess → infer → save."""

    input_csv = Path(input_path)
    output_csv = Path(output_path)

    # ── Load input data ──────────────────────────────────────────────
    if not input_csv.exists():
        print(f"Input file {input_csv} not found. Using built-in dummy data for testing.")
        data = [
            {"timestamp": "2025-01-15 08:00:00", "ticker": "INFY.NS",
             "source": "Navbharat Times", "language": "hi",
             "headline": "इन्फोसिस ने एक प्रमुख यूरोपीय ग्राहक के साथ शानदार सौदा हासिल किया।"},
            {"timestamp": "2025-01-15 08:05:00", "ticker": "ICICIBANK.NS",
             "source": "Amar Ujala", "language": "hi",
             "headline": "आईसीआईसीआई बैंक के शेयर लुढ़के आंतरिक प्रबंधन विवाद की खबर।"},
            {"timestamp": "2025-01-15 08:10:00", "ticker": "BHARTIARTL.NS",
             "source": "Zee Business Hindi", "language": "hi",
             "headline": "Airtel ne plans mehangi kar di jiske baad stock me jabardast tezi aayi."},
        ]
        df = pd.DataFrame(data)
    else:
        print(f"Reading from {input_csv}")
        df = pd.read_csv(input_csv)

        # Filter Hindi articles only
        if 'language' in df.columns:
            df = df[df['language'] == 'hi'].copy()

        if df.empty:
            print("No Hindi articles found in the input CSV.")
            return

    # ── Load model ───────────────────────────────────────────────────
    sent_pipeline, model_name = load_sentiment_pipeline()

    # ── Run inference ────────────────────────────────────────────────
    print(f"\nRunning Hindi sentiment inference ({model_name}) on {len(df)} headlines...")

    headlines = df['headline'].tolist()
    results = []
    batch_size = 16

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        # Truncate long headlines to 512 tokens (model max)
        batch_results = sent_pipeline(batch, truncation=True, max_length=512)
        results.extend(batch_results)

    # ── Build output DataFrame ───────────────────────────────────────
    output_df = pd.DataFrame({
        'timestamp': df['timestamp'].values,
        'ticker': df['ticker'].values,
        'source': df['source'].values,
        'sentiment_score': [label_to_score(r) for r in results],
        'model_confidence': [round(r['score'], 4) for r in results],
        'model_used': model_name,
    })

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    print(f"\nSuccessfully processed {len(output_df)} Hindi/Hinglish articles.")
    print(f"Saved to {output_csv}")

    # Print summary
    pos = (output_df['sentiment_score'] > 0).sum()
    neg = (output_df['sentiment_score'] < 0).sum()
    neu = (output_df['sentiment_score'] == 0).sum()
    print(f"\nSentiment Distribution: Positive={pos}, Negative={neg}, Neutral={neu}")
    print(f"Mean Score: {output_df['sentiment_score'].mean():.4f}")
    print(f"Score Range: [{output_df['sentiment_score'].min():.4f}, {output_df['sentiment_score'].max():.4f}]")

    print("\nSample output:")
    print(output_df.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hindi Sentiment Analysis (MuRIL)")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="Path to cleaned Hindi news CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Path to save sentiment results CSV")
    args = parser.parse_args()

    run_hindi_sentiment(args.input, args.output)
