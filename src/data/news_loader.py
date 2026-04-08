"""
src/data/news_loader.py — Load and deduplicate raw news CSVs.

Expected CSV columns (minimum):
    date, time, headline, body, lang, source, ticker

The loader:
  1. Reads all CSVs in data/raw/news/ (or a single file if ticker is given)
  2. Parses datetime = date + time into a UTC-aware timestamp
  3. Deduplicates on (url_hash OR headline_hash) + date
  4. Time-gates: drops articles with published_datetime >= T 15:30 IST
     for the purpose of predicting day T (anti-leakage)
"""

import pathlib
import hashlib
import pandas as pd
from typing import Optional, List

from src.utils.logger import get_logger
from src.utils.errors import run_stage
from src.utils.paths import RAW_DATA_DIR

log = get_logger("news_loader")

IST_OFFSET = pd.Timedelta("5h30min")
MARKET_CLOSE_IST = pd.Timedelta("15h30min")   # 3:30 PM IST


def _hash_row(row: pd.Series) -> str:
    key = str(row.get("url", "")) + str(row.get("headline", ""))
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine date + time columns into a single IST-aware datetime."""
    if "datetime" in df.columns:
        df["pub_dt"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    elif "date" in df.columns and "time" in df.columns:
        df["pub_dt"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
        )
    elif "date" in df.columns:
        df["pub_dt"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("News CSV must have 'datetime' or 'date' column.")
    return df


def _apply_leakage_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each article with the trading date it belongs to.
    Articles published after 15:30 IST on day T belong to day T+1.
    This ensures no future information leaks into today's sentiment.
    """
    if "pub_dt" not in df.columns:
        return df

    # Convert pub_dt to IST (naive → add offset, or localise)
    pub_ist = df["pub_dt"]
    if pub_ist.dt.tz is not None:
        pub_ist = pub_ist.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    time_of_day = pub_ist.dt.hour * 3600 + pub_ist.dt.minute * 60 + pub_ist.dt.second
    cutoff_secs = int(MARKET_CLOSE_IST.total_seconds())   # 55800

    # Articles after 15:30 → assign to next calendar day
    df["trade_date"] = pub_ist.dt.normalize()
    after_close      = time_of_day >= cutoff_secs
    df.loc[after_close, "trade_date"] = (
        df.loc[after_close, "trade_date"] + pd.Timedelta("1D")
    )
    return df


def load_news(
    ticker: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    news_dir: pathlib.Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """
    Load all news CSVs, deduplicate, and apply leakage gate.

    Args:
        ticker  : Load only headlines mentioning this ticker.
        tickers : Load for a list of tickers (ignored if ticker given).
        news_dir: Directory containing raw news CSVs.

    Returns:
        pd.DataFrame with columns:
            trade_date, headline, body, lang, source, ticker, pub_dt, row_hash
    """
    csv_files = sorted(news_dir.glob("*.csv"))
    if not csv_files:
        log.warning(f"No CSV files found in {news_dir}. Returning empty DataFrame.")
        return pd.DataFrame()

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            frames.append(df)
        except Exception as e:
            log.warning(f"[SKIP] Could not read {f.name}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded {len(df):,} raw articles from {len(frames)} file(s).")

    # ── Filter by ticker ──────────────────────────────────────────────────────
    scope = [ticker] if ticker else (tickers or [])
    if scope:
        scope_upper = [t.upper() for t in scope]
        if "ticker" in df.columns:
            df = df[df["ticker"].str.upper().isin(scope_upper)]
        elif "headline" in df.columns:
            pattern = "|".join(scope_upper)
            df = df[df["headline"].str.upper().str.contains(pattern, na=False)]
        log.info(f"After ticker filter {scope}: {len(df):,} articles remain.")

    # ── Deduplication ─────────────────────────────────────────────────────────
    df["row_hash"] = df.apply(_hash_row, axis=1)
    before = len(df)
    df = df.drop_duplicates(subset=["row_hash"])
    log.info(f"Deduplicated: {before - len(df):,} duplicates removed. {len(df):,} remain.")

    # ── Datetime parsing + leakage gate ──────────────────────────────────────
    df = _parse_datetime(df)
    df = _apply_leakage_gate(df)
    df = df.dropna(subset=["trade_date"])

    # ── Ensure lang column exists ─────────────────────────────────────────────
    if "lang" not in df.columns:
        df["lang"] = "unknown"
    if "source" not in df.columns:
        df["source"] = "unknown"
    if "body" not in df.columns:
        df["body"] = ""

    return df.reset_index(drop=True)
