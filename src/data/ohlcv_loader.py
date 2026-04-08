"""
src/data/ohlcv_loader.py — Load OHLCV Parquet files and compute price features.

Expected Parquet schema (minimum):
    date, open, high, low, close, volume

Output: DataFrame indexed by date with ≤11 engineered price features,
        all rolling z-score normalised (window=252 trading days).
"""

import pathlib
import numpy as np
import pandas as pd
from typing import Optional, List

from src.utils.logger import get_logger
from src.utils.paths import OHLCV_DIR

log = get_logger("ohlcv_loader")


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_return(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return (100 - 100 / (1 + rs)) / 100   # normalise to [0,1]


def _atr(high, low, close, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _bollinger_width(close: pd.Series, period: int = 20) -> pd.Series:
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (2 * std) / (ma + 1e-9)   # normalised band width


def _zscore_norm(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score normalisation (unit‑free, cross‑stock comparable)."""
    mu  = series.rolling(window, min_periods=30).mean()
    sig = series.rolling(window, min_periods=30).std()
    return (series - mu) / (sig + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "log_ret", "vol_norm", "gap_open",
    "rsi_14",
    "ma20_ratio", "ma50_ratio", "ma200_ratio",
    "bb_width",
    "atr_14_norm",
    "vol_z",
    "log_ret_z",
]


def load_ohlcv(
    ticker: str,
    cfg: dict,
    ohlcv_dir: pathlib.Path = OHLCV_DIR,
) -> pd.DataFrame:
    """
    Load OHLCV for `ticker` and compute all price features.

    Supported file formats (searched in order):
        <ticker>.parquet, <ticker>.csv, <TICKER>.parquet, <TICKER>.csv

    Returns:
        pd.DataFrame indexed by `date` (DatetimeIndex) with columns = FEATURE_COLS.
        Rows with NaN features (first ~200 trading days) are dropped.
    """
    feat_cfg = cfg.get("features", {})
    rsi_p    = feat_cfg.get("rsi_period", 14)
    ma_ps    = feat_cfg.get("ma_periods", [20, 50, 200])
    atr_p    = feat_cfg.get("atr_period", 14)
    bb_p     = feat_cfg.get("bollinger_period", 20)
    vnorm_w  = feat_cfg.get("volume_norm_window", 20)
    zw       = feat_cfg.get("zscore_window", 252)

    # ── Find file ─────────────────────────────────────────────────────────────
    candidates = [
        ohlcv_dir / f"{ticker}.parquet",
        ohlcv_dir / f"{ticker.upper()}.parquet",
        ohlcv_dir / f"{ticker}.csv",
        ohlcv_dir / f"{ticker.upper()}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"OHLCV data for '{ticker}' not found in {ohlcv_dir}.\n"
            f"Expected one of: {[p.name for p in candidates]}\n"
            f"Place your Parquet/CSV file there and retry."
        )

    log.info(f"[{ticker}] Loading OHLCV from {path.name} …")

    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path, low_memory=False)

    raw.columns = raw.columns.str.strip().str.lower()

    # ── Validate schema ───────────────────────────────────────────────────────
    required = {"date", "open", "high", "low", "close", "volume"}
    missing  = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"[{ticker}] OHLCV file missing columns: {missing}\n"
            f"Found columns: {list(raw.columns)}\n"
            f"Rename your columns to match: date, open, high, low, close, volume"
        )

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).set_index("date").sort_index()
    raw = raw[["open", "high", "low", "close", "volume"]].astype(float)

    # ── Feature engineering ───────────────────────────────────────────────────
    df = pd.DataFrame(index=raw.index)

    df["log_ret"]    = _log_return(raw["close"])
    df["vol_norm"]   = raw["volume"] / raw["volume"].rolling(vnorm_w).mean().clip(lower=1)
    df["gap_open"]   = np.log((raw["open"] / raw["close"].shift(1)).clip(lower=1e-6))
    df["rsi_14"]     = _rsi(raw["close"], rsi_p)
    df["ma20_ratio"] = raw["close"] / raw["close"].rolling(ma_ps[0]).mean().clip(lower=1e-6)
    df["ma50_ratio"] = raw["close"] / raw["close"].rolling(ma_ps[1]).mean().clip(lower=1e-6)
    df["ma200_ratio"]= raw["close"] / raw["close"].rolling(ma_ps[2]).mean().clip(lower=1e-6)
    df["bb_width"]   = _bollinger_width(raw["close"], bb_p)
    atr              = _atr(raw["high"], raw["low"], raw["close"], atr_p)
    df["atr_14_norm"]= atr / raw["close"].clip(lower=1e-6)
    df["vol_z"]      = _zscore_norm(np.log(raw["volume"].clip(lower=1)), zw)
    df["log_ret_z"]  = _zscore_norm(df["log_ret"], zw)

    # Also keep raw close for label generation (dropped before model sees it)
    df["close"]      = raw["close"]

    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    log.info(
        f"[{ticker}] {before} rows → {len(df)} after dropping NaN "
        f"(first {before - len(df)} warm-up rows removed)."
    )
    return df
