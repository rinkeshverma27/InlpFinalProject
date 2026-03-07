import subprocess
import pathlib
import sys
import os
import re

def run_batch(ticker_filter=None):
    base_dir = pathlib.Path("/home/rinkesh-verma/Desktop/INLP Final Project/InlpFinalProject")
    price_dir = base_dir / "data/price"
    nifty_path = price_dir / "NIFTY50" / "ohlcv.csv"
    results = []

    # Get tickers from directories
    if ticker_filter:
        tickers = [t for t in ticker_filter if (price_dir / t).is_dir()]
    else:
        tickers = [d.name for d in price_dir.iterdir() if d.is_dir() and d.name != "NIFTY50"]
    
    tickers.sort()

    print(f"🚀 Starting Batch Run for {len(tickers)} tickers...")
    print("-" * 60)

    for ticker in tickers:
        print(f"\n📦 Processing {ticker}...")
        ohlcv_path = price_dir / ticker / "ohlcv.csv"
        
        # 1. Train
        train_cmd = [
            "python", str(base_dir / "src/main.py"),
            "--mode", "train",
            "--ticker", ticker,
            "--ohlcv", str(ohlcv_path),
            "--nifty", str(nifty_path)
        ]
        print(f"  Training...")
        try:
            subprocess.run(train_cmd, check=True, capture_output=True, text=True, env=os.environ.copy())
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Training failed for {ticker}: {e.stderr}")
            continue

        # 2. Predict
        checkpoint_path = base_dir / "outputs/checkpoints/model_2026-03-07.pt" # Standard date-based naming
        predict_cmd = [
            "python", str(base_dir / "src/main.py"),
            "--mode", "predict",
            "--ticker", ticker,
            "--ohlcv", str(ohlcv_path),
            "--nifty", str(nifty_path),
            "--checkpoint", str(checkpoint_path)
        ]
        print(f"  Predicting...")
        try:
            res = subprocess.run(predict_cmd, check=True, capture_output=True, text=True, env=os.environ.copy())
            output = res.stdout
            
            # Extract Direction Acc
            match = re.search(r"Direction Acc: ([\d.]+)%", output)
            acc = match.group(1) if match else "N/A"
            
            # Extract Mean Error
            match_err = re.search(r"Mean Error:\s+([\d.]+%)", output)
            err = match_err.group(1) if match_err else "N/A"
            
            results.append({
                "ticker": ticker,
                "accuracy": acc,
                "error": err
            })
            print(f"  ✅ Complete! Acc: {acc}%, Err: {err}")
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Prediction failed for {ticker}: {e.stderr}")

    # 3. Report
    print("\n" + "=" * 60)
    print("📊 FINAL BATCH RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Ticker':<15} | {'Accuracy (%)':<15} | {'Mean Error':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['ticker']:<15} | {r['accuracy']:<15} | {r['error']:<15}")
    print("=" * 60)

if __name__ == "__main__":
    # Ensure PYTHONPATH is set
    os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"
    
    # User requested filter (Top 4)
    top_tickers = [
        "HDFCBANK", "BHARTIARTL", "KOTAKBANK", "RELIANCE"
    ]
    run_batch(ticker_filter=top_tickers)
