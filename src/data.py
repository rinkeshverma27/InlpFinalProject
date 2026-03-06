import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class StockDataset(Dataset):
    """
    PyTorch Dataset for a single stock's time-series data.
    Implements:
      - Feature normalization (min-max for OHLC, log for Volume) per blueprint §2.2
      - Dynamic window sizing (60/30/10 days) based on realized volatility
      - Walk-forward safe indexing (no future data leakage)
    """

    # Blueprint §2.2 Stream B features
    FEATURE_COLS = [
        'Open', 'High', 'Low', 'Close',       # min-max normalised per stock
        'Volume',                               # log-scaled
        'dist_from_sma',                        # 5-day MA delta proxy
        'vol_delta',                            # 20-day MA delta proxy
        'rsi',                                  # RSI 14-period
        'bb_width',                             # realised volatility proxy
        'Volatility_20D',                       # risk state
    ]

    def __init__(self, df, seq_length=60, normalise=True):
        self.df = df.reset_index(drop=True)
        self.seq_length = seq_length

        available = [c for c in self.FEATURE_COLS if c in self.df.columns]
        raw = self.df[available].copy()

        # ── Normalisation (blueprint §2.2) ──────────────────────────────
        if normalise:
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in raw.columns:
                    cmin, cmax = raw[col].min(), raw[col].max()
                    raw[col] = (raw[col] - cmin) / (cmax - cmin + 1e-8)
            if 'Volume' in raw.columns:
                raw['Volume'] = np.log1p(raw['Volume'])

        self.features = raw.values.astype(np.float32)
        self.targets = self.df['target_label'].values.astype(np.float32)

        # Dynamic window ratios
        if 'Volatility_20D' in self.df.columns:
            vol_mean = self.df['Volatility_20D'].mean()
            self.vol_ratios = (self.df['Volatility_20D'] / (vol_mean + 1e-8)).fillna(1.0).values
        else:
            self.vol_ratios = np.ones(len(self.df))

        # Valid indices: need at least seq_length history and a non-NaN target
        self.valid_indices = [
            i for i in range(self.seq_length, len(self.df))
            if not np.isnan(self.targets[i])
        ]

    # Blueprint §2.2 dynamic window
    def get_dynamic_window(self, idx):
        r = self.vol_ratios[idx]
        if r > 2.0:
            return 10
        elif r > 1.5:
            return 30
        return 60

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        idx = self.valid_indices[i]
        window = self.get_dynamic_window(idx)
        x_raw = self.features[idx - window: idx]

        # Left-pad to seq_length for uniform batching
        if len(x_raw) < self.seq_length:
            pad = self.seq_length - len(x_raw)
            x_raw = np.pad(x_raw, ((pad, 0), (0, 0)), constant_values=0)

        return (
            torch.tensor(x_raw, dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            window,
        )


# ── DataLoader factory ──────────────────────────────────────────────────────

def get_train_val_dataloaders(csv_path, tickers, batch_size=32, val_years=2):
    """
    Walk-forward split: train on expanding window, validate on last `val_years`.
    Year 20 (1year-test.csv) is NEVER loaded here — holdout sealed.
    """
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    df = df[df['Ticker'].isin(tickers)]
    df = df.ffill().bfill()

    max_date = df['Date'].max()
    cutoff = max_date - pd.DateOffset(years=val_years)

    train_df = df[df['Date'] < cutoff]
    val_df = df[df['Date'] >= cutoff]
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    train_ds, val_ds = [], []
    for t in tickers:
        t_df = train_df[train_df['Ticker'] == t]
        v_df = val_df[val_df['Ticker'] == t]
        if len(t_df) > 60:
            train_ds.append(StockDataset(t_df))
        if len(v_df) > 60:
            val_ds.append(StockDataset(v_df))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_ds), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(val_ds), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def get_test_dataloader(csv_path, tickers, batch_size=32):
    """Load the Year-20 holdout set (1year-test.csv) for final evaluation only."""
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    df = df[df['Ticker'].isin(tickers)]
    df = df.ffill().bfill()

    datasets = [StockDataset(df[df['Ticker'] == t]) for t in tickers
                if len(df[df['Ticker'] == t]) > 60]

    return torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(datasets), batch_size=batch_size, shuffle=False
    )
