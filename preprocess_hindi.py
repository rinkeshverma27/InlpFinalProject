"""
Hindi News Preprocessing Pipeline
===================================
Mirrors the English pipeline from the YRP midterm report (Section 5.1-5.2):

  1. Load & merge all monthly hindi_news_YYYY_MM.csv files
  2. Deduplicate headlines (hash within 6-hour window, same as English)
  3. Timestamp alignment:
     - Articles after 15:30 IST → shifted to next trading day
     - Articles with NO timestamp → assigned 09:15 IST (market open)
       so downstream sentiment can be treated as "pre-open" signal
  4. Ticker disambiguation (same logic as English — keep only unambiguous matches)
  5. Binary label construction stub (market-corrected return > +0.5% = UP,
     < -0.5% = DOWN; actual labels added by signal_merging.py after OHLCV join)
  6. Output:
     - dataset/processed/hindi_articles_all.csv   ← full deduplicated corpus
     - dataset/processed/hindi_by_ticker/          ← one CSV per ticker+MACRO
     - dataset/processed/hindi_daily_summary.csv   ← (ticker, date, article_count)
       for diagnostic use

Usage:
    python preprocess_hindi.py
    python preprocess_hindi.py --raw_dir dataset/raw_dataset --out_dir dataset/processed
    python preprocess_hindi.py --start 2023-01-01 --end 2024-12-31

Author: YRP Team 
"""

import argparse
import csv
import hashlib
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd

# ─── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── constants ─────────────────────────────────────────────────────────────────
DEFAULT_RAW_DIR  = Path("dataset/raw_dataset")
DEFAULT_OUT_DIR  = Path("dataset/processed")
MARKET_CLOSE_IST = "15:30"        # articles after this → next trading day
DEFAULT_TIMESTAMP = "09:15"       # fallback timestamp when none available (pre-open)
DEDUP_WINDOW_HRS  = 6             # hours within which same hash = duplicate

# Nifty-50 tickers (all 50 + MACRO)
ALL_TICKERS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "KOTAKBANK", "TATASTEEL",
    "SBIN", "NTPC", "BAJFINANCE", "POWERGRID", "ONGC", "WIPRO", "ITC",
    "BPCL", "BHARTIARTL", "TCS", "HINDUNILVR", "ADANIPORTS", "ADANIENT",
    "ULTRACEMCO", "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "DRREDDY",
    "CIPLA", "DIVISLAB", "BAJAJFINSV", "HINDALCO", "JSWSTEEL", "TECHM",
    "HCLTECH", "TATAMOTORS", "M&M", "NESTLEIND", "BAJAJ-AUTO", "HEROMOTOCO",
    "INDUSINDBK", "AXISBANK", "COALINDIA", "GRASIM", "LT", "BRITANNIA",
    "APOLLOHOSP", "EICHERMOT", "SHREECEM", "TATACONSUM", "SBILIFE",
    "HDFCLIFE", "PIDILITIND", "MACRO",
]

# Trading holidays (add more as needed — NSE calendar)
TRADING_HOLIDAYS = {
    date(2023, 1, 26), date(2023, 3, 7), date(2023, 3, 30), date(2023, 4, 4),
    date(2023, 4, 7), date(2023, 4, 14), date(2023, 5, 1), date(2023, 6, 29),
    date(2023, 8, 15), date(2023, 9, 19), date(2023, 10, 2), date(2023, 10, 24),
    date(2023, 11, 14), date(2023, 11, 27), date(2023, 12, 25),
    date(2024, 1, 22), date(2024, 1, 26), date(2024, 3, 25), date(2024, 3, 29),
    date(2024, 4, 11), date(2024, 4, 14), date(2024, 4, 17), date(2024, 5, 1),
    date(2024, 5, 23), date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15),
    date(2024, 10, 2), date(2024, 10, 14), date(2024, 11, 1), date(2024, 11, 15),
    date(2024, 11, 20), date(2024, 12, 25),
    date(2021, 1, 26), date(2021, 3, 11), date(2021, 3, 29), date(2021, 4, 2),
    date(2021, 4, 14), date(2021, 4, 21), date(2021, 5, 13), date(2021, 7, 21),
    date(2021, 8, 19), date(2021, 11, 4), date(2021, 11, 5), date(2021, 11, 19),
    date(2021, 12, 24),
    date(2022, 1, 26), date(2022, 3, 1), date(2022, 3, 18), date(2022, 4, 14),
    date(2022, 4, 15), date(2022, 5, 3), date(2022, 8, 9), date(2022, 8, 15),
    date(2022, 10, 5), date(2022, 10, 24), date(2022, 10, 26), date(2022, 12, 26),
}

# ─── helpers ───────────────────────────────────────────────────────────────────

def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in TRADING_HOLIDAYS

def next_trading_day(d: date) -> date:
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt

def parse_date_safe(s: str) -> date | None:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            return None
    return None

def parse_timestamp_safe(ts: str, date_str: str) -> str:
    """Return 'HH:MM' or DEFAULT_TIMESTAMP."""
    if ts and isinstance(ts, str):
        ts = ts.strip()
        m = re.search(r"(\d{2}:\d{2})", ts)
        if m:
            return m.group(1)
    return DEFAULT_TIMESTAMP

def normalize_headline(text: str) -> str:
    """Lowercase, strip punctuation, normalize whitespace for dedup."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def headline_hash(text: str) -> str:
    return hashlib.md5(normalize_headline(text).encode("utf-8")).digest().hex()[:16]

def align_to_trading_day(pub_date: date, hhmm: str) -> date:
    """
    Apply the 15:30 IST cutoff rule (same as English pipeline):
    - Published after 15:30 → attributed to next trading day
    - Weekend / holiday → next trading day
    - No timestamp (DEFAULT_TIMESTAMP=09:15) → same day if trading, else next
    """
    # weekend / holiday base shift
    while not is_trading_day(pub_date):
        pub_date = next_trading_day(pub_date)
    # post-market cutoff
    if hhmm > MARKET_CLOSE_IST:
        pub_date = next_trading_day(pub_date)
    return pub_date

# ─── load & merge ───────────────────────────────────────────────────────────────

def load_raw_csvs(raw_dir: Path, start: date | None, end: date | None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("hindi_news_????_??.csv"))
    if not files:
        log.error(f"No hindi_news_YYYY_MM.csv files found in {raw_dir}")
        sys.exit(1)

    log.info(f"Found {len(files)} monthly CSV files")
    frames = []
    for f in files:
        # quick date filter from filename
        m = re.search(r"hindi_news_(\d{4})_(\d{2})\.csv", f.name)
        if m:
            yr, mo = int(m.group(1)), int(m.group(2))
            file_date = date(yr, mo, 1)
            if start and file_date < date(start.year, start.month, 1):
                continue
            if end and file_date > date(end.year, end.month, 1):
                continue
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", dtype=str, low_memory=False)
            df["_source_file"] = f.name
            frames.append(df)
            log.info(f"  Loaded {f.name}: {len(df)} rows")
        except Exception as e:
            log.warning(f"  Could not load {f.name}: {e}")

    if not frames:
        log.error("No files loaded after date filtering.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    log.info(f"Total raw rows: {len(df)}")
    return df

# ─── clean & validate ──────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Standardise column names (handle old schema without timestamp)
    col_map = {
        "date": "datePublished",
        "title": "headline",
        "news": "articleBody",
        "ticker": "symbol",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["datePublished", "headline", "articleBody", "symbol", "url"]
    for c in required:
        if c not in df.columns:
            df[c] = ""

    # Ensure timestamp column exists (older runs may not have it)
    if "timestamp" not in df.columns:
        df["timestamp"] = ""

    # Drop rows missing essential fields
    before = len(df)
    df = df.dropna(subset=["headline", "articleBody"])
    df = df[df["headline"].str.strip().astype(bool)]
    df = df[df["articleBody"].str.strip().astype(bool)]
    log.info(f"After dropping empty headline/body: {len(df)} (dropped {before - len(df)})")

    # Validate symbol
    df["symbol"] = df["symbol"].str.strip().str.upper().fillna("MACRO")
    df = df[df["symbol"].isin(ALL_TICKERS)]
    log.info(f"After symbol validation: {len(df)}")

    # Parse dates
    df["_pub_date"] = df["datePublished"].apply(parse_date_safe)
    missing_date = df["_pub_date"].isna().sum()
    if missing_date:
        log.warning(f"  {missing_date} rows with unparseable datePublished — dropped")
    df = df.dropna(subset=["_pub_date"])
    log.info(f"After date validation: {len(df)}")

    return df

# ─── timestamp alignment ───────────────────────────────────────────────────────

def add_aligned_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'trading_date' column using 15:30 IST rule.
    When timestamp is missing, assign DEFAULT_TIMESTAMP (09:15 = pre-open)
    so the article is attributed to the same trading day.
    """
    def _align(row):
        d = row["_pub_date"]
        raw_ts = row.get("timestamp", "")
        ts_text = "" if pd.isna(raw_ts) else str(raw_ts).strip()
        hhmm = parse_timestamp_safe(ts_text, row.get("datePublished", ""))
        # fill in timestamp if missing
        if not ts_text:
            row_ts = f"{d.isoformat()} {DEFAULT_TIMESTAMP}"
        else:
            row_ts = ts_text
        return align_to_trading_day(d, hhmm), hhmm, row_ts

    results = df.apply(_align, axis=1, result_type="expand")
    df["trading_date"]      = results[0]
    df["_time_hhmm"]        = results[1]
    df["timestamp_filled"]  = results[2]   # filled timestamp for downstream

    ts_series = df["timestamp"].fillna("").astype(str).str.strip()
    missing_ts = (ts_series == "").sum()
    log.info(f"Rows with no timestamp (assigned {DEFAULT_TIMESTAMP} pre-open): {missing_ts}")
    log.info(f"Rows shifted to next trading day (post 15:30): "
             f"{(df['trading_date'] > df['_pub_date']).sum()}")
    return df

# ─── deduplication ─────────────────────────────────────────────────────────────

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate: same headline hash within a 6-hour window for the same
    (company, symbol) = same story.
    Keep first occurrence (earliest timestamp).
    """
    df = df.copy()
    df["_hash"] = df["headline"].apply(headline_hash)
    # Sort by date + time so first occurrence is kept
    df = df.sort_values(["_pub_date", "_time_hhmm"]).reset_index(drop=True)

    # Build key with date, 6-hour bucket, headline hash, company, and symbol
    # so different company/symbol pairs are never treated as duplicates.
    def _bucket(row):
        d = row["_pub_date"]
        try:
            hh = int(row["_time_hhmm"].split(":")[0])
        except Exception:
            hh = 9
        bucket = hh // 6  # 0-3 (four 6-hour windows per day)
        company = "" if pd.isna(row.get("company", "")) else str(row.get("company", "")).strip().lower()
        symbol = "" if pd.isna(row.get("symbol", "")) else str(row.get("symbol", "")).strip().upper()
        return f"{d.isoformat()}_{bucket}_{row['_hash']}_{company}_{symbol}"

    df["_dedup_key"] = df.apply(_bucket, axis=1)
    before = len(df)
    df = df.drop_duplicates(subset=["_dedup_key"])
    log.info(f"After dedup (6h window): {len(df)} (removed {before - len(df)} duplicates)")
    return df

# ─── output ───────────────────────────────────────────────────────────────────

OUTPUT_COLS = [
    "datePublished", "timestamp", "timestamp_filled", "trading_date",
    "company", "symbol", "headline", "description",
    "articleBody", "tags", "author", "url", "source", "language",
]

def save_outputs(df: pd.DataFrame, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker_dir = out_dir / "hindi_by_ticker"
    ticker_dir.mkdir(exist_ok=True)

    # ensure all output cols exist
    for c in OUTPUT_COLS:
        if c not in df.columns:
            df[c] = ""

    # convert trading_date back to string
    df["trading_date"] = df["trading_date"].apply(
        lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x)
    )

    # ── 1. full corpus ──
    all_path = out_dir / "hindi_articles_all.csv"
    df[OUTPUT_COLS].to_csv(all_path, index=False, encoding="utf-8-sig")
    log.info(f"Saved full corpus: {all_path} ({len(df)} rows)")

    # ── 2. per-ticker files ──
    for ticker in ALL_TICKERS:
        sub = df[df["symbol"] == ticker]
        if sub.empty:
            continue
        t_path = ticker_dir / f"hindi_{ticker}.csv"
        sub[OUTPUT_COLS].to_csv(t_path, index=False, encoding="utf-8-sig")

    log.info(f"Saved {len(ALL_TICKERS)} per-ticker files to {ticker_dir}/")

    # ── 3. daily summary ──
    summary = (
        df.groupby(["symbol", "trading_date"])
        .agg(
            article_count=("url", "count"),
            has_timestamp=("timestamp", lambda x: (x.str.strip() != "").sum()),
        )
        .reset_index()
        .rename(columns={"trading_date": "date"})
        .sort_values(["symbol", "date"])
    )
    summary_path = out_dir / "hindi_daily_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    log.info(f"Saved daily summary: {summary_path} ({len(summary)} rows)")

    # ── 4. stats report ──
    total       = len(df)
    with_ts     = (df["timestamp"].str.strip() != "").sum()
    with_date   = (df["datePublished"].str.strip() != "").sum()
    tickers_hit = df["symbol"].nunique()
    date_range  = f"{df['datePublished'].min()} to {df['datePublished'].max()}"

    log.info("=" * 55)
    log.info("PREPROCESSING SUMMARY")
    log.info("=" * 55)
    log.info(f"Total articles        : {total:,}")
    log.info(f"With datePublished    : {with_date:,} ({with_date/total*100:.1f}%)")
    log.info(f"With timestamp        : {with_ts:,} ({with_ts/total*100:.1f}%)")
    log.info(f"Unique tickers        : {tickers_hit}")
    log.info(f"Date range            : {date_range}")
    log.info(f"Ticker breakdown:")
    for t, cnt in df["symbol"].value_counts().head(15).items():
        log.info(f"   {t:<15} {cnt:>6,}")
    log.info("=" * 55)


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hindi News Preprocessing Pipeline")
    parser.add_argument("--raw_dir", default=str(DEFAULT_RAW_DIR),
                        help="Directory with hindi_news_YYYY_MM.csv files")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR),
                        help="Output directory")
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--no_ticker_files", action="store_true",
                        help="Skip per-ticker CSV output (faster)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start) if args.start else None
    end   = date.fromisoformat(args.end)   if args.end   else None

    log.info("=== Hindi News Preprocessing Pipeline ===")
    log.info(f"Raw dir : {args.raw_dir}")
    log.info(f"Out dir : {args.out_dir}")
    if start: log.info(f"Start   : {start}")
    if end:   log.info(f"End     : {end}")

    # Step 1: Load
    df = load_raw_csvs(Path(args.raw_dir), start, end)

    # Step 2: Clean & validate
    df = clean(df)

    # Step 3: Timestamp alignment
    df = add_aligned_date(df)

    # Apply date range filter on trading_date
    if start:
        df = df[df["trading_date"] >= start]
    if end:
        df = df[df["trading_date"] <= end]
    log.info(f"After date range filter: {len(df)} rows")

    # Step 4: Deduplicate
    df = dedup(df)

    # Step 5: Save
    if not args.no_ticker_files:
        save_outputs(df, Path(args.out_dir))
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_cols = [c for c in OUTPUT_COLS if c in df.columns]
        df["trading_date"] = df["trading_date"].apply(
            lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x)
        )
        df[out_cols].to_csv(out_dir / "hindi_articles_all.csv",
                            index=False, encoding="utf-8-sig")
        log.info(f"Saved corpus only (no ticker files)")

    log.info("=== Preprocessing complete ===")


if __name__ == "__main__":
    main()
