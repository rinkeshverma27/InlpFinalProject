"""
data/dataset.py
PyTorch Dataset classes for the T-LSTM Dual-Stream model.

Stream B  → OHLCVDataset: sequences of engineered OHLCV features
Handshake → HandshakeDataset: NLP embeddings + sentiment from the CSV freeze point

StockDataset merges both streams into a single (features, nlp_vector, label) tuple.
"""

import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Feature columns produced by utils/features.py (used as LSTM input)
OHLCV_FEATURE_COLS = [
    "open_norm", "high_norm", "low_norm", "close_norm",
    "volume_log",
    "rsi_14",
    "ma_delta_5_20",
    "realised_vol_5",
    "regime_flag",
]

# Embedding columns in the handshake CSV
EN_EMB_COLS = [f"en_emb_{i}" for i in range(512)]
HI_EMB_COLS = [f"hi_emb_{i}" for i in range(512)]


# ──────────────────────────────────────────────────────────────────────────────
# Handshake CSV Loader
# ──────────────────────────────────────────────────────────────────────────────

class HandshakeDataset:
    """
    Loads the frozen NLP feature CSV written by Part 1 (sentiment pipeline).

    Expected CSV schema (minimum columns):
        date, ticker,
        en_sentiment, hi_sentiment,
        en_emb_0 … en_emb_511,
        hi_emb_0 … hi_emb_511,
        trust_weight_en, trust_weight_hi

    Returns a dict keyed by (date, ticker) → np.ndarray of shape (1024,)
    representing the trust-weighted concatenation of EN and HI embeddings.
    """

    def __init__(self, handshake_dir: pathlib.Path):
        self.handshake_dir = pathlib.Path(handshake_dir)
        self._index: dict[tuple, np.ndarray] = {}
        self._load_from_directory()

    def _load_from_directory(self):
        # The structure is: data/handshake/fused_{YYYY-MM-DD}.csv
        if not self.handshake_dir.exists():
            return
            
        for filepath in self.handshake_dir.glob("fused_*.csv"):
            try:
                # Expecting fused_YYYY-MM-DD.csv
                date_str = filepath.stem.split("_")[1]
                date_obj = pd.to_datetime(date_str).date()
            except Exception:
                continue
                
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                ticker = row["ticker"]

            en_emb = row[EN_EMB_COLS].values.astype(np.float32)
            hi_emb = row[HI_EMB_COLS].values.astype(np.float32)

            # Trust-weighted average of both 512-dim embeddings → 512-dim vector
            w_en = float(row.get("trust_weight_en", 1.0))
            w_hi = float(row.get("trust_weight_hi", 1.0))
            total = w_en + w_hi + 1e-8

            fused = (w_en * en_emb + w_hi * hi_emb) / total   # 512-dim

            # Also append scalar sentiment scores (→ 514-dim total)
            # The LSTM fusion concatenates this with the 256-dim LSTM output
            en_sent = float(row.get("en_sentiment", 0.0))
            hi_sent = float(row.get("hi_sentiment", 0.0))
            nlp_vec = np.concatenate([fused, [en_sent, hi_sent]])  # (514,)

        self._index[(date_obj, ticker)] = nlp_vec

    def get(self, date, ticker) -> np.ndarray:
        """Returns the NLP vector for (date, ticker), or zeros if missing."""
        key = (date if not isinstance(date, pd.Timestamp) else date.date(), ticker)
        return self._index.get(key, np.zeros(514, dtype=np.float32))


# ──────────────────────────────────────────────────────────────────────────────
# OHLCV Sequence + Label Dataset
# ──────────────────────────────────────────────────────────────────────────────

class StockDataset(Dataset):
    """
    PyTorch Dataset producing:
        ohlcv_seq  : Tensor (window_size, n_features)   — LSTM input
        nlp_vec    : Tensor (514,)                       — Fusion input
        label      : Tensor (1,)                         — adjusted_ret

    Parameters
    ----------
    feature_df   : DataFrame with OHLCV feature columns + window_size + adjusted_ret
                   indexed by DatetimeIndex, single ticker.
    ticker       : NSE ticker string (e.g. "RELIANCE")
    handshake    : HandshakeDataset instance (or None for ablation runs)
    min_window   : minimum rows needed before a sample is valid (default 60)
    """

    def __init__(
        self,
        feature_df: pd.DataFrame,
        ticker: str,
        handshake: HandshakeDataset | None = None,
        min_window: int = 60,
    ):
        self.ticker    = ticker
        self.handshake = handshake
        self.min_window = min_window

        # Drop rows that don't yet have enough history or a valid label
        df = feature_df.dropna(subset=OHLCV_FEATURE_COLS + ["adjusted_ret"])
        df = df.iloc[min_window:]  # first `min_window` rows can't form a full window

        self.df       = df.reset_index()   # keep the date column accessible
        self.features = df[OHLCV_FEATURE_COLS].values.astype(np.float32)
        self.labels   = df["adjusted_ret"].values.astype(np.float32)
        self.windows  = df["window_size"].values.astype(int)
        self.dates    = df.index if "date" not in df.columns else df["date"].values

        # Store full feature array for window slicing (prepend min_window rows)
        full_features = feature_df[OHLCV_FEATURE_COLS].ffill().fillna(0).values.astype(np.float32)
        self._full    = full_features
        self._offset  = len(feature_df) - len(df)   # index correction

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        actual_idx = self._offset + idx
        window     = int(self.windows[idx])

        # Slice the lookback window
        # For batching, we pad everything to the global MAX window (60)
        max_window = 60
        start = max(0, actual_idx - window)
        seq   = self._full[start:actual_idx]          # (window, n_feat)

        # Pad on the left to max_window
        if len(seq) < max_window:
            pad = np.zeros((max_window - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)

        ohlcv_seq = torch.tensor(seq, dtype=torch.float32)

        # NLP vector from handshake
        date_col = "date" if "date" in self.df.columns else "Date"
        date = self.df.iloc[idx][date_col] if date_col in self.df.columns else self.dates[idx]
        if self.handshake is not None:
            nlp_vec = torch.tensor(self.handshake.get(date, self.ticker), dtype=torch.float32)
        else:
            nlp_vec = torch.zeros(514, dtype=torch.float32)

        label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        return ohlcv_seq, nlp_vec, label


# ──────────────────────────────────────────────────────────────────────────────
# Walk-Forward Splitter
# ──────────────────────────────────────────────────────────────────────────────

def walk_forward_splits(
    df: pd.DataFrame,
    train_years: int = 3,         # Reduced from 17 for sample compatibility
    val_months: int = 3,
    step_months: int = 1,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generates (train_df, val_df) pairs for expanding-window walk-forward CV.

    Initial train window: first `train_years` years of data.
    Validation window: next `val_months` months.
    Each subsequent fold expands the training set by `step_months`.
    """
    df = df.sort_index()
    start = df.index.min()
    train_end = start + pd.DateOffset(years=train_years)
    splits = []

    while True:
        val_end = train_end + pd.DateOffset(months=val_months)
        if val_end > df.index.max():
            break
        train_df = df.loc[:train_end]
        val_df   = df.loc[train_end:val_end]
        splits.append((train_df, val_df))
        train_end += pd.DateOffset(months=step_months)

    return splits
