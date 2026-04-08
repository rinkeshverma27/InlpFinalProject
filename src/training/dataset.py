"""
src/training/dataset.py — PyTorch Dataset for the dual-stream LSTM.
Loads fused tensors from data/processed/features/ and applies walk-forward splits.
"""

import pathlib
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Tuple, Optional

from src.utils.logger import get_logger
from src.utils.paths import FEATURES_DIR, SPLITS_DIR
from src.utils.cache import stage_exists, mark_stage_done

log = get_logger("dataset")


class StockSequenceDataset(Dataset):
    """
    Wraps the fused tensor dict for a single ticker.

    Returns:
        price_seq    : [T, 11] float32
        sentiment_seq: [T, 9]  float32
        label        : scalar float32  (0.0 or 1.0)
        date         : str
    """

    def __init__(self, fused_data: dict, indices: np.ndarray):
        self.price     = fused_data["price_seq"]     # [N, T, 11]
        self.sentiment = fused_data["sentiment_seq"] # [N, T, 9]
        self.labels    = fused_data["labels"]        # [N]
        self.dates     = fused_data["dates"]         # List[str]
        self.indices   = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (
            self.price[idx],
            self.sentiment[idx],
            self.labels[idx],
            self.dates[idx],
        )


def make_splits(
    fused_data: dict,
    cfg: dict,
    ticker: str,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate walk-forward train/val/test index splits.
    Saves to data/splits/<ticker>_splits.npz (cached).

    Returns:
        train_idx, val_idx, test_idx  — numpy int arrays
    """
    split_path = SPLITS_DIR / f"{ticker}_splits.npz"
    ttl        = cfg.get("cache", {}).get("splits_ttl_days", 30)

    if stage_exists(split_path, ttl, force, cfg):
        npz = np.load(split_path)
        return npz["train"], npz["val"], npz["test"]

    dates_pd = np.array(fused_data["dates"])   # ['2019-01-02', ...]

    t_cfg    = cfg.get("training", {})
    train_s  = t_cfg.get("train_start", "2019-01-01")
    train_e  = t_cfg.get("train_end",   "2021-12-31")
    val_s    = t_cfg.get("val_start",   "2022-01-01")
    val_e    = t_cfg.get("val_end",     "2022-12-31")
    test_s   = t_cfg.get("test_start",  "2023-01-01")
    test_e   = t_cfg.get("test_end",    "2023-12-31")

    train_idx = np.where((dates_pd >= train_s) & (dates_pd <= train_e))[0]
    val_idx   = np.where((dates_pd >= val_s)   & (dates_pd <= val_e))[0]
    test_idx  = np.where((dates_pd >= test_s)  & (dates_pd <= test_e))[0]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(split_path, train=train_idx, val=val_idx, test=test_idx)
    mark_stage_done(split_path, {"ticker": ticker,
                                 "n_train": len(train_idx),
                                 "n_val": len(val_idx),
                                 "n_test": len(test_idx)})
    log.info(
        f"[{ticker}] Splits: train={len(train_idx)}, "
        f"val={len(val_idx)}, test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


def collate_fn(batch):
    """Custom collate: returns tensors + list of date strings."""
    prices, sents, labels, dates = zip(*batch)
    return (
        torch.stack(prices),
        torch.stack(sents),
        torch.stack(labels),
        list(dates),
    )
