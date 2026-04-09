"""
src/utils/paths.py
Canonical path constants for the NIFTY-NLP project.
All paths are resolved relative to the project root so the codebase
is portable across all three machines.

Usage:
    from src.utils.paths import NEWS_RAW_DIR, NEWS_PROCESSED_DIR, ...
"""

from pathlib import Path

# ── Project root (two levels up from this file: src/utils/ → src/ → project/) ──
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# ── Raw / Processed News ─────────────────────────────────────────────────────
NEWS_DIR: Path           = PROJECT_ROOT / "data" / "news"
NEWS_RAW_DIR: Path       = NEWS_DIR / "raw"
NEWS_PROCESSED_DIR: Path = NEWS_DIR / "processed"

# ── Handshake CSV (trust-weighted daily sentiment vectors) ───────────────────
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
HANDSHAKE_CSV: Path      = PROCESSED_DATA_DIR / "handshake.csv"

# ── Trust Weights (Tier 1 daily JSON store) ───────────────────────────────────
TRUST_WEIGHTS_DIR: Path = PROJECT_ROOT / "data" / "trust_weights"

# ── Model Artefacts ──────────────────────────────────────────────────────────
MODELS_DIR: Path                  = PROJECT_ROOT / "models"
MURIL_MODEL_DIR: Path             = MODELS_DIR / "muril_financial_sentiment_v1"
BINARY_LSTM_MODEL: Path           = MODELS_DIR / "prod_binary_lstm_best.pth"
SCALER_FILE: Path                 = MODELS_DIR / "prod_scaler.joblib"

# ── Dataset inputs (production-ready CSVs) ───────────────────────────────────
INPUTS_DIR: Path   = PROJECT_ROOT / "data" / "inputs"
TRAIN_CSV: Path    = INPUTS_DIR / "prod_train.csv"
TEST_CSV: Path     = INPUTS_DIR / "prod_test.csv"
SYNTHETIC_CSV: Path = INPUTS_DIR / "mega_synthetic_hindi_train.csv"

# ── Predictions output ───────────────────────────────────────────────────────
PREDICTIONS_DIR: Path = PROJECT_ROOT / "data" / "predictions"
PREDICTIONS_CSV: Path = PREDICTIONS_DIR / "production_predictions.csv"

# ── Raw scraped news (legacy / data-server path) ─────────────────────────────
SCRAPED_NEWS_CSV: Path = (
    PROJECT_ROOT / "dataset" / "raw_dataset" / "english_news_nifty50.csv"
)

# ── Ensure critical output directories exist on import ───────────────────────
for _dir in [
    NEWS_RAW_DIR,
    NEWS_PROCESSED_DIR,
    PROCESSED_DATA_DIR,
    TRUST_WEIGHTS_DIR,
    PREDICTIONS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
