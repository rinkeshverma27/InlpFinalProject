"""
utils/features.py
Feature engineering for OHLCV data.
Computes: RSI-14, MA deltas (5/20), Realised Volatility, Regime flag,
and the Dynamic Window size.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Technical Indicator Helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI (period-day)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_ma_delta(close: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """(MA_short - MA_long) / MA_long — relative delta."""
    ma_s = close.rolling(short).mean()
    ma_l = close.rolling(long).mean()
    return (ma_s - ma_l) / ma_l.replace(0, np.nan)


def compute_realised_vol(close: pd.Series, window: int = 5) -> pd.Series:
    """Rolling std of log-returns over `window` days."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std()


def compute_regime_flag(realised_vol: pd.Series) -> pd.Series:
    """
    0 = normal  (<1.0x historical mean)
    1 = moderate (1.0–2.0x)
    2 = event    (>2.0x)
    """
    hist_mean = realised_vol.expanding().mean()
    ratio = realised_vol / hist_mean.replace(0, np.nan)
    flag = pd.cut(ratio,
                  bins=[-np.inf, 1.0, 2.0, np.inf],
                  labels=[0, 1, 2]).astype(float)
    return flag


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Window Sizer
# ──────────────────────────────────────────────────────────────────────────────

def get_window_size(realised_vol_value: float, hist_mean: float) -> int:
    """
    Returns the lookback window length for a given day's realised vol.

    RealVol < 1.0x hist_mean  → 60 days  (calm regime)
    1.0x ≤ RealVol < 2.0x     → 30 days  (moderate)
    RealVol ≥ 2.0x             → 10 days  (event / high vol)
    """
    if hist_mean == 0 or np.isnan(hist_mean):
        return 30  # safe default
    ratio = realised_vol_value / hist_mean
    if ratio < 1.0:
        return 60
    elif ratio < 2.0:
        return 30
    else:
        return 10


# ──────────────────────────────────────────────────────────────────────────────
# Main Feature Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a DataFrame with columns: [Open, High, Low, Close, Volume]
    sorted by date (oldest first) for a SINGLE stock.

    Returns the same frame augmented with:
      open_norm, high_norm, low_norm, close_norm  (min-max per stock)
      volume_log
      rsi_14
      ma_delta_5_20
      realised_vol_5
      regime_flag
      window_size                    (informational; used by Dataset)
    """
    out = df.copy()

    # ── Normalise OHLC ───────────────────────────────────────────────────────
    for col in ["Open", "High", "Low", "Close"]:
        col_min = df[col].min()
        col_max = df[col].max()
        out[f"{col.lower()}_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-8)

    # ── Log-scale volume ─────────────────────────────────────────────────────
    out["volume_log"] = np.log1p(df["Volume"])

    # ── Technical indicators ─────────────────────────────────────────────────
    out["rsi_14"]        = compute_rsi(df["Close"])
    out["ma_delta_5_20"] = compute_ma_delta(df["Close"])
    out["realised_vol_5"]= compute_realised_vol(df["Close"])
    out["regime_flag"]   = compute_regime_flag(out["realised_vol_5"])

    # ── Dynamic window size per row ──────────────────────────────────────────
    hist_means = out["realised_vol_5"].expanding().mean()
    out["window_size"] = [
        get_window_size(rv, hm)
        for rv, hm in zip(out["realised_vol_5"], hist_means)
    ]

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Label Construction
# ──────────────────────────────────────────────────────────────────────────────

def build_labels(
    df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    close_col: str = "Close",
    price_10am_col: str = "price_10am",
) -> pd.DataFrame:
    """
    Compute adjusted return label:

      raw_ret      = (price_10am_{t+1} - close_t) / close_t
      adjusted_ret = raw_ret - nifty50_overnight_ret_{t+1}

    Both `df` and `nifty_df` must be indexed by date (DatetimeIndex),
    sorted ascending, and share the same trading calendar.

    `df` must contain: Close, price_10am  (filled in at 10:05 AM)
    `nifty_df` must contain: Close, price_10am  (Nifty 50 index)
    """
    out = df.copy()

    # Nifty overnight return (index level)
    nifty_raw = (
        nifty_df[price_10am_col].shift(-1) - nifty_df[close_col]
    ) / nifty_df[close_col]

    # Stock raw return
    stock_raw = (
        df[price_10am_col].shift(-1) - df[close_col]
    ) / df[close_col]

    # Align on date index
    nifty_aligned = nifty_raw.reindex(df.index)

    out["raw_ret"]       = stock_raw
    out["nifty_ret"]     = nifty_aligned
    out["adjusted_ret"]  = stock_raw - nifty_aligned   # ← the prediction target

    return out
