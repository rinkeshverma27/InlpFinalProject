"""
src/data/data_fusion.py — In-memory alignment of Stream A (sentiment) and Stream B (price).

The "handshake" concept is implemented here as a strict left-join on trade_date.
Leakage is already baked into the sentiment Parquet by sentiment_aggregator.py.
This module only handles the join + forward-fill + sequence building.
"""

import pathlib
import numpy as np
import pandas as pd
import torch
from typing import Optional

from src.utils.logger import get_logger
from src.utils.cache import stage_exists, mark_stage_done
from src.utils.paths import SENTIMENT_DIR, FEATURES_DIR
from src.data.ohlcv_loader import load_ohlcv, FEATURE_COLS

log = get_logger("data_fusion")

SENTIMENT_COLS = [
    "pos_1d", "neg_1d", "neu_1d",
    "pos_3d", "neg_3d", "neu_3d",
    "pos_7d", "neg_7d", "neu_7d",
]


def _build_label(price_df: pd.DataFrame, horizon: int = 1, threshold: float = 0.003) -> pd.Series:
    """
    Binary label: 1 = next-day return > +threshold, 0 = next-day return < -threshold.
    Rows with |return| <= threshold are marked NaN and dropped later.
    """
    fut_ret = price_df["close"].pct_change(horizon).shift(-horizon)
    label   = pd.Series(np.nan, index=price_df.index)
    label[fut_ret >  threshold] = 1.0
    label[fut_ret < -threshold] = 0.0
    return label


def fuse_ticker(
    ticker: str,
    cfg: dict,
    force: bool = False,
) -> Optional[dict]:
    """
    Fuse sentiment + price features for one ticker into tensors.

    Checks cache first. Saves result to data/processed/features/<ticker>.pt.

    Returns:
        dict with keys:
            price_seq    : FloatTensor [N, T, 11]   — price feature sequences
            sentiment_seq: FloatTensor [N, T, 9]    — sentiment broadcast sequences
            labels       : FloatTensor [N]           — binary labels
            dates        : list[str]                 — corresponding trade dates
        or None if insufficient data.
    """
    out_path = FEATURES_DIR / f"{ticker}.pt"
    ttl      = cfg.get("cache", {}).get("fusion_ttl_days", 7)

    if stage_exists(out_path, ttl, force, cfg):
        log.info(f"[{ticker}] Loading fused tensors from cache …")
        return torch.load(out_path, weights_only=False)

    # ── Load price features ───────────────────────────────────────────────────
    try:
        price_df = load_ohlcv(ticker, cfg)
    except FileNotFoundError as e:
        log.error(str(e))
        return None

    # ── Load sentiment ────────────────────────────────────────────────────────
    sent_path = SENTIMENT_DIR / f"{ticker}.parquet"
    if not sent_path.exists():
        log.warning(
            f"[{ticker}] No sentiment data found. "
            f"Generating dummy neutral sentiment to allow Price-Only training."
        )
        # Create a dummy dataframe with neutral sentiment (pos=0, neg=0, neu=1)
        sent_df = pd.DataFrame(index=price_df.index)
        for col in SENTIMENT_COLS:
            sent_df[col] = 1.0 if "neu" in col else 0.0
    else:
        sent_df = pd.read_parquet(sent_path)
        if "date" in sent_df.columns:
            sent_df["date"] = pd.to_datetime(sent_df["date"])
            sent_df = sent_df.set_index("date")
        sent_df.index = pd.to_datetime(sent_df.index)

    # ── Join ──────────────────────────────────────────────────────────────────
    fused = price_df.join(sent_df[SENTIMENT_COLS], how="left")
    # Forward-fill missing sentiment (sparse Hindi days, weekends)
    fused[SENTIMENT_COLS] = fused[SENTIMENT_COLS].ffill(limit=5).fillna(0.0)

    # ── Labels ────────────────────────────────────────────────────────────────
    horizon   = cfg.get("training", {}).get("label_horizon_days", 1)
    threshold = cfg.get("training", {}).get("label_threshold", 0.003)
    fused["label"] = _build_label(fused, horizon, threshold)
    
    # Drop rows with NaN *features*, but keep NaN *labels* to preserve time continuity
    fused = fused.dropna(subset=FEATURE_COLS)

    # ── Dynamic window sequencing ─────────────────────────────────────────────
    from src.features.window_sizer import get_window
    feat_cfg   = cfg.get("features", {})
    w_min      = feat_cfg.get("window_min", 10)
    w_max      = feat_cfg.get("window_max", 60)
    w_scale    = feat_cfg.get("window_atr_scale", 50)
    max_window = w_max   # pad to this length for uniform tensor

    if len(fused) <= max_window:
        log.warning(f"[{ticker}] Dataframe ({len(fused)} rows) too short for max_window {max_window}.")
        return None

    price_arr   = fused[FEATURE_COLS].values.astype(np.float32)
    sent_arr    = fused[SENTIMENT_COLS].values.astype(np.float32)
    labels_arr  = fused["label"].values.astype(np.float32)
    dates_arr   = fused.index.strftime("%Y-%m-%d").tolist()

    # ATR percentile per row (for dynamic window)
    atr_series  = fused["atr_14_norm"].rank(pct=True)

    price_seqs, sent_seqs, labels, dates = [], [], [], []

    for i in range(max_window, len(fused)):
        if np.isnan(labels_arr[i]):
            continue  # Chop day, skip training sample

        window = get_window(atr_series.iloc[i], w_min, w_max, w_scale)
        start  = i - window

        p_seq  = price_arr[start:i]          # [window, 11]
        s_seq  = sent_arr[start:i]           # [window, 9]

        # Pad to max_window at the front with zeros
        pad    = max_window - window
        p_pad  = np.zeros((pad, p_seq.shape[1]), dtype=np.float32)
        s_pad  = np.zeros((pad, s_seq.shape[1]), dtype=np.float32)
        p_seq  = np.concatenate([p_pad, p_seq], axis=0)
        s_seq  = np.concatenate([s_pad, s_seq], axis=0)

        price_seqs.append(p_seq)
        sent_seqs.append(s_seq)
        labels.append(labels_arr[i])
        dates.append(dates_arr[i])

    if not price_seqs:
        log.warning(f"[{ticker}] No valid trade days found after dropping chop days.")
        return None

    result = {
        "price_seq":     torch.tensor(np.stack(price_seqs),  dtype=torch.float32),
        "sentiment_seq": torch.tensor(np.stack(sent_seqs),   dtype=torch.float32),
        "labels":        torch.tensor(labels,                 dtype=torch.float32),
        "dates":         dates,
        "ticker":        ticker,
    }

    torch.save(result, out_path)
    mark_stage_done(out_path, {"ticker": ticker, "n_samples": len(labels)})
    log.info(f"[{ticker}] Fused {len(labels)} sequences → {out_path.name}")
    return result
