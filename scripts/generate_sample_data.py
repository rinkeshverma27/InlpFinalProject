import os
import pathlib
import numpy as np
import pandas as pd

def generate_ohlcv(ticker, output_dir, n_days=5000):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2006-01-01", periods=n_days, freq="B")
    
    close = 1000 + np.cumsum(rng.normal(0, 5, n_days))
    if ticker == "NIFTY50":
        close = 10000 + np.cumsum(rng.normal(0, 10, n_days))
        
    df = pd.DataFrame({
        "Date": dates,
        "ticker": ticker,
        "Open": close * rng.uniform(0.99, 1.0, n_days),
        "High": close * rng.uniform(1.00, 1.01, n_days),
        "Low": close * rng.uniform(0.98, 0.99, n_days),
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n_days),
        "price_10am": close * rng.uniform(0.995, 1.005, n_days)
    })
    
    out_path = output_dir / "ohlcv.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} ({len(df)} rows)")

def generate_handshake(output_dir, ticker="RELIANCE", n_days=5000):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2006-01-01", periods=n_days, freq="B")
    
    # We only need past 50 days of handshake data to speed up generation, 
    # but the training loop will look up the specific dates. 
    # Let's just generate the last 150 days.
    recent_dates = dates[-150:]
    
    for date in recent_dates:
        date_str = date.strftime("%Y-%m-%d")
        
        # 512 + 512 + 4 = 1028 columns approx
        row = {"date": date_str, "ticker": ticker, "en_sentiment": rng.uniform(-1, 1), "hi_sentiment": rng.uniform(-1, 1), 
               "trust_weight_en": rng.uniform(0.8, 1.5), "trust_weight_hi": rng.uniform(0.5, 1.2)}
               
        for i in range(512):
            row[f"en_emb_{i}"] = rng.normal(0, 0.1)
            row[f"hi_emb_{i}"] = rng.normal(0, 0.1)
            
        df = pd.DataFrame([row])
        out_path = output_dir / f"fused_{date_str}.csv"
        df.to_csv(out_path, index=False)

if __name__ == "__main__":
    base_dir = pathlib.Path("data")
    
    # Generate Nifty 50 and Reliance
    generate_ohlcv("NIFTY50", base_dir / "price" / "NIFTY50")
    generate_ohlcv("RELIANCE", base_dir / "price" / "RELIANCE")
    
    # Generate Handshake
    generate_handshake(base_dir / "handshake", "RELIANCE")
