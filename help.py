# Create this file and run it: python split_data.py
import pandas as pd
import pathlib

csv_path = "data/datasets/OneYearMarketData.csv"
out_dir  = pathlib.Path("data/raw/ohlcv")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

# Rename to the exact standardized schema our pipeline requires
df.rename(columns={
    "Date": "date",    "Open": "open", "High": "high",
    "Low": "low",      "Close": "close", "Volume": "volume"
}, inplace=True)

# Parse dates 
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Split and save
for ticker in df["Ticker"].unique():
    clean_ticker = str(ticker).replace(".NS", "").replace(".BO", "")
    ticker_df = df[df["Ticker"] == ticker].copy().sort_values("date")
    
    out_path = out_dir / f"{clean_ticker}.csv"
    ticker_df.to_csv(out_path, index=False)
    print(f"Saved {out_path.name} ({len(ticker_df)} rows)")
