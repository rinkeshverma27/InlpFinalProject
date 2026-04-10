"""
src/utils/paths.py — Central path resolver.
All code imports paths from here. Never hardcode paths anywhere.
"""

import pathlib
import yaml
import os

# Project root = directory containing this file's grandparent (gemini/)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config.yaml"


def _load_cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_cfg = _load_cfg()
_data = _cfg.get("data", {})

# ── Raw inputs ───────────────────────────────────────────────────────────────
RAW_DATA_DIR      = PROJECT_ROOT / _data.get("news_dir",  "data/raw/news/")
OHLCV_DIR         = PROJECT_ROOT / _data.get("ohlcv_dir", "data/raw/ohlcv/")
DATASETS_DIR      = PROJECT_ROOT / _data.get("datasets_dir", "data/datasets/")

# ── Processed outputs ────────────────────────────────────────────────────────
SENTIMENT_DIR     = PROJECT_ROOT / _data.get("sentiment_dir", "data/processed/sentiment/")
FEATURES_DIR      = PROJECT_ROOT / _data.get("features_dir",  "data/processed/features/")
SPLITS_DIR        = PROJECT_ROOT / _data.get("splits_dir",    "data/splits/")

# ── Models ───────────────────────────────────────────────────────────────────
MODELS_DIR        = PROJECT_ROOT / "models"
PRETRAINED_DIR    = MODELS_DIR / "pretrained"
FINETUNED_DIR     = MODELS_DIR / "finetuned"
MURIL_FT_DIR      = FINETUNED_DIR / "muril_financial"
LSTM_DIR          = MODELS_DIR / "lstm"
CHECKPOINTS_DIR   = LSTM_DIR / "checkpoints"
PRODUCTION_DIR    = LSTM_DIR / "production"

# ── Logs ─────────────────────────────────────────────────────────────────────
LOGS_DIR          = PROJECT_ROOT / "logs"
PRED_LOGS_DIR     = LOGS_DIR / "predictions"
TRAIN_LOGS_DIR    = LOGS_DIR / "training"
EVAL_REPORT_DIR   = LOGS_DIR / "eval_report"


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [
        RAW_DATA_DIR, OHLCV_DIR, DATASETS_DIR,
        SENTIMENT_DIR, FEATURES_DIR, SPLITS_DIR,
        PRETRAINED_DIR, FINETUNED_DIR, MURIL_FT_DIR,
        CHECKPOINTS_DIR, PRODUCTION_DIR,
        PRED_LOGS_DIR, TRAIN_LOGS_DIR, EVAL_REPORT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
