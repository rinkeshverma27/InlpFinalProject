"""
src/nlp/sentiment_aggregator.py — Aggregate FinBERT and MuRIL scores into
per-ticker per-day sentiment vectors with 1/3/7-day rolling windows.

Cache: saves to data/processed/sentiment/<ticker>.parquet
Hindi sparse gate: if Hindi article count < min_threshold, carry forward
                   last valid Hindi signal (no CSV, pure in-memory state dict).
"""

import pathlib
import numpy as np
import pandas as pd
import torch
from typing import List, Optional

from src.utils.logger import get_logger
from src.utils.cache import stage_exists, mark_stage_done
from src.utils.errors import run_stage
from src.utils.paths import SENTIMENT_DIR, LOGS_DIR

log = get_logger("sentiment_agg")

SCORE_COLS = ["pos", "neg", "neu"]
OUT_COLS   = [
    "pos_1d", "neg_1d", "neu_1d",
    "pos_3d", "neg_3d", "neu_3d",
    "pos_7d", "neg_7d", "neu_7d",
]


def _source_weight(source: str, weights_cfg: dict) -> float:
    s = source.lower()
    for key, w in weights_cfg.items():
        if key in s:
            return w
    return weights_cfg.get("default", 0.5)


def _weighted_mean_scores(
    scores_df: pd.DataFrame,
    sources: List[str],
    weights_cfg: dict,
) -> np.ndarray:
    """Weighted average of [pos, neg, neu] vectors by source reliability."""
    if scores_df.empty:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)   # default: neutral

    weights = np.array([_source_weight(s, weights_cfg) for s in sources], dtype=np.float32)
    weights = weights / weights.sum()
    vals    = scores_df[SCORE_COLS].values.astype(np.float32)
    return (vals * weights[:, None]).sum(axis=0)


def aggregate_ticker(
    ticker: str,
    news_df: pd.DataFrame,
    cfg: dict,
    force: bool = False,
    device: Optional[torch.device] = None,
) -> Optional[pd.DataFrame]:
    """
    Full Stream A pipeline for one ticker:
      1. Cache check
      2. Language detect per article
      3. FinBERT (EN) / MuRIL (HI/Hinglish)
      4. Sparse Hindi gate + forward fill
      5. Source-weighted daily aggregation
      6. 1/3/7-day rolling windows
      7. Save Parquet

    Args:
        ticker   : Stock ticker string.
        news_df  : DataFrame from news_loader.load_news() filtered for this ticker.
        cfg      : Full config dict.
        force    : Bypass cache.

    Returns:
        pd.DataFrame with columns OUT_COLS, indexed by trade_date.
    """
    out_path = SENTIMENT_DIR / f"{ticker}.parquet"
    ttl      = cfg.get("cache", {}).get("sentiment_ttl_days", 1)

    if stage_exists(out_path, ttl, force, cfg):
        return pd.read_parquet(out_path)

    if news_df.empty:
        log.warning(f"[{ticker}] No news articles — returning neutral sentiment.")
        return None

    nlp_cfg     = cfg.get("nlp", {})
    lang_thresh = nlp_cfg.get("lang_confidence_threshold", 0.85)
    hi_min      = nlp_cfg.get("hindi_min_articles", 3)
    windows     = nlp_cfg.get("sentiment_windows", [1, 3, 7])
    weights_cfg = cfg.get("source_weights", {})

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Language detection ────────────────────────────────────────────────────
    from src.nlp.lang_detector import detect_batch
    headlines = news_df["headline"].fillna("").tolist()
    langs     = detect_batch(headlines, lang_thresh)
    news_df   = news_df.copy()
    news_df["detected_lang"] = langs

    # ── Score ─────────────────────────────────────────────────────────────────
    from src.nlp.finbert_scorer import score_english
    from src.nlp.muril_scorer   import score_hindi

    ckpt_dir = LOGS_DIR / "inference_checkpoints" / ticker
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    en_mask = news_df["detected_lang"] == "en"
    hi_mask = news_df["detected_lang"].isin(["hi", "hinglish"])

    en_scores = pd.DataFrame(columns=SCORE_COLS)
    hi_scores = pd.DataFrame(columns=SCORE_COLS)

    if en_mask.sum() > 0:
        en_texts  = news_df.loc[en_mask, "headline"].tolist()
        en_scores = score_english(en_texts, cfg, device=device, checkpoint_dir=ckpt_dir)
        en_scores.index = news_df.index[en_mask]

    if hi_mask.sum() > 0:
        hi_texts  = news_df.loc[hi_mask, "headline"].tolist()
        hi_scores = score_hindi(hi_texts, cfg, device=device, checkpoint_dir=ckpt_dir)
        hi_scores.index = news_df.index[hi_mask]

    scores = pd.concat([en_scores, hi_scores]).sort_index()
    news_df = news_df.join(scores, rsuffix="_score")
    news_df[SCORE_COLS] = news_df[SCORE_COLS].fillna({"pos": 0.0, "neg": 0.0, "neu": 1.0})

    # ── Daily aggregation ────────────────────────────────────────────────────
    daily_records = []
    for trade_date, day_df in news_df.groupby("trade_date"):
        en_day = day_df[day_df["detected_lang"] == "en"]
        hi_day = day_df[day_df["detected_lang"].isin(["hi", "hinglish"])]

        en_vec = _weighted_mean_scores(en_day, en_day.get("source", pd.Series()).tolist(), weights_cfg)
        hi_vec = _weighted_mean_scores(hi_day, hi_day.get("source", pd.Series()).tolist(), weights_cfg)

        # Sparse Hindi gate
        hi_count = len(hi_day)
        if hi_count < hi_min:
            hi_weight = 0.0
        else:
            hi_weight = min(1.0, hi_count / (hi_min * 2))

        en_weight  = 1.0 - hi_weight
        daily_vec  = en_weight * en_vec + hi_weight * hi_vec

        daily_records.append({
            "date":          pd.Timestamp(trade_date),
            "pos":           daily_vec[0],
            "neg":           daily_vec[1],
            "neu":           daily_vec[2],
            "hi_count":      hi_count,
            "hi_weight":     hi_weight,
        })

    if not daily_records:
        log.warning(f"[{ticker}] Aggregation produced 0 daily records.")
        return None

    daily = pd.DataFrame(daily_records).set_index("date").sort_index()

    # Sparse Hindi gate: forward-fill gap days with last valid Hindi signal
    daily[["pos", "neg", "neu"]] = daily[["pos", "neg", "neu"]].ffill(limit=5)

    # ── Rolling windows ───────────────────────────────────────────────────────
    result = pd.DataFrame(index=daily.index)
    for w in windows:
        roll = daily[["pos", "neg", "neu"]].rolling(w, min_periods=1).mean()
        result[f"pos_{w}d"] = roll["pos"]
        result[f"neg_{w}d"] = roll["neg"]
        result[f"neu_{w}d"] = roll["neu"]

    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path)
    mark_stage_done(out_path, {"ticker": ticker, "n_days": len(result)})
    log.info(f"[{ticker}] Sentiment aggregated: {len(result)} trading days → {out_path.name}")
    return result
