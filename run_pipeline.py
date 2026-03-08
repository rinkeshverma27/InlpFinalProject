#!/usr/bin/env python3
"""
master_pipeline.py
Orchestrates the entire NIFTY-NLP workflow from Stage 0 (data cleaning) 
through Stage 3 (model execution & prediction).

Usage:
    conda activate nifty-nlp
    python master_pipeline.py --ticker RELIANCE
"""

import argparse
import subprocess
import sys
import pathlib
import datetime

def run_step(script_path: str, args: list[str], description: str):
    """Helper to run a step of the pipeline safely."""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING: {description}")
    print(f"📦 COMMAND: python {script_path} {' '.join(args)}")
    try:
        subprocess.run([sys.executable, script_path] + args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR in {script_path}. Pipeline halted.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="NIFTY-NLP Master Pipeline")
    parser.add_argument("--ticker", type=str, default="RELIANCE", help="Target ticker (e.g. RELIANCE)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for TLSTM")
    parser.add_argument("--train-years", type=int, default=17, help="Walk forward split training years")
    # Use real datasets by default, but allow overriding for tests
    parser.add_argument("--news-data", type=str, default="data/inputs/Nifty50_news_data(2020Jan_2024April).csv")
    parser.add_argument("--price-data", type=str, default="data/price/{ticker}/ohlcv.csv")
    parser.add_argument("--nifty-data", type=str, default="data/price/NIFTY50/ohlcv.csv")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip the long NLP processing if handshakes exist")
    args = parser.parse_args()

    active_price_data = args.price_data.replace("{ticker}", args.ticker)
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    print("\n" + "="*80)
    print(f"📊 NIFTY-NLP PIPELINE STARTING FOR TICKER: {args.ticker}")
    print("="*80)

    # ---------------------------------------------------------
    # STAGE 0 & 1: Data Preparation & NLP Scoring
    # ---------------------------------------------------------
    if not args.skip_nlp:
        # Step 1: Clean Raw News
        run_step("src/01_data_cleaning.py", 
                 ["--input", args.news_data], 
                 "Stage 1: Cleaning Raw News Data")
        
        # Step 2: Separate English & Hindi News
        run_step("src/02_news_separator.py", 
                 ["--input", "data/outputs/cleaned_news.csv"], 
                 "Stage 1: Language Language Separation")

        # Step 3: English FinBERT Scoring
        run_step("src/03_finbert_score.py", 
                 ["--input", "data/outputs/english_news.csv", "--ticker", args.ticker], 
                 "Stage 1: FinBERT Sentiment Scoring (English)")
                 
        # Step 4: Hindi MuRIL Scoring
        run_step("src/04_muril_score.py", 
                 ["--input", "data/outputs/hindi_news.csv", "--ticker", args.ticker], 
                 "Stage 1: MuRIL Sentiment Scoring (Hindi)")
                 
        # Step 5: Trust-weighting and Handshake matrix generation
        run_step("src/05_handshake.py", 
                 ["--ticker", args.ticker], 
                 "Stage 1: Generating Trust-Weighted Handshake Vectors")
    else:
        print("\n⏩ SKIPPING Stage 1 (NLP Processing). Assuming handshake matrices exist in data/handshake/")

    # ---------------------------------------------------------
    # STAGE 3: Model Dual-Stream Execution
    # ---------------------------------------------------------
    
    # Step 6: Train the T-LSTM Model
    run_step("src/main.py",
             [
                 "--mode", "train", 
                 "--ticker", args.ticker, 
                 "--ohlcv", active_price_data,
                 "--nifty", args.nifty_data,
                 "--train_years", str(args.train_years)
             ],
             f"Stage 3: Training T-LSTM Dual-Stream Model for {args.epochs} epochs")

    # Step 7: Prediction Sequence (Daily Run)
    ckpt_path = f"outputs/checkpoints/model_{today_str}.pt"
    run_step("src/main.py",
             [
                 "--mode", "predict", 
                 "--ticker", args.ticker,
                 "--ohlcv", active_price_data,
                 "--nifty", args.nifty_data,
                 "--checkpoint", ckpt_path
             ],
             f"Stage 3: Monte Carlo Dropout Prediction Sequence")

    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"👉 Check data/predictions/predictions_{today_str}.csv for the latest trading signals.")
    print("="*80)


if __name__ == "__main__":
    main()
