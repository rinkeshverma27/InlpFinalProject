"""
main.py
Entry point for Part 2 — T-LSTM Dual-Stream model.

Usage:
    python main.py --mode train --ticker RELIANCE --ohlcv data/sample_ohlcv.csv
    python main.py --mode predict --ticker RELIANCE --checkpoint outputs/checkpoints/model.pt
    python main.py --mode ewc_nudge --ticker RELIANCE --checkpoint outputs/checkpoints/model.pt
"""

import argparse
import pathlib
import pandas as pd
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility (Phase 8)
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    """
    Ensures that if we run the same data and settings twice, we get EXACTLY the same result.
    1. torch.manual_seed: Fixes initial weights of LSTM/NLP heads.
    2. np.random.seed: Fixes NumPy-based data noise.
    3. random.seed: Fixes standard Python randomness.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from src.utils.features import build_features, build_labels
from src.data.dataset import StockDataset, HandshakeDataset, walk_forward_splits
from src.models.tlstm import TLSTMModel
from src.training.trainer import Trainer
from src.training.predict import run_prediction


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_ohlcv_features": 14,
    "lstm_hidden_dim":  256,
    "lstm_layers":      2,
    "lstm_dropout":     0.3,
    "nlp_in_dim":       514,
    "nlp_out_dim":      512,
    "head_dropout":     0.3,
    "batch_size":       64,
    "epochs":           55,     # Production-grade 50+ training
    "patience":         10,     # Increased to match Trainer
    "ewc_epochs":       3,
    "ewc_lambda":       400.0,
    "ewc_lr":           1e-5,
    "lr":               3e-4,   # Lowered per Phase 4
    "n_mc_passes":      50,
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("  Using CPU")
    return dev


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_ohlcv(ohlcv_path: pathlib.Path, ticker: str) -> pd.DataFrame:
    """Load and filter OHLCV CSV for a single ticker."""
    df = pd.read_csv(ohlcv_path, parse_dates=["Date"])
    # Handle both base ticker and .NS suffix
    possible_tickers = [ticker, f"{ticker}.NS"]
    df = df[df["ticker"].isin(possible_tickers)].set_index("Date").sort_index()
    return df


def prepare_features(df: pd.DataFrame, nifty_df: pd.DataFrame) -> pd.DataFrame:
    """Run feature engineering and label construction."""
    feat_df = build_features(df)
    feat_df = build_labels(feat_df, nifty_df)
    return feat_df


# ──────────────────────────────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────────────────────────────

def mode_train(args, cfg, device):
    ohlcv_path = pathlib.Path(args.ohlcv)

    # Load the unified dataset
    df_raw = pd.read_csv(ohlcv_path, parse_dates=["Date"])
    
    # Filter for ticker (unified file might have multiple)
    possible_tickers = [args.ticker, f"{args.ticker}.NS"]
    ticker_df = df_raw[df_raw["ticker"].isin(possible_tickers)].sort_values("Date")
    
    if ticker_df.empty:
        raise ValueError(f"Ticker {args.ticker} not found in {ohlcv_path}")

    # Re-run feature engineering to get the exact normalized columns model expects
    # We use a dummy Nifty DF since nifty_ret_proxy is already handled or we can just pass the same df
    feat_df = build_features(ticker_df.set_index("Date"))
    
    # Ensure adjusted_ret exists (consolidator used target_label or similar)
    if "adjusted_ret" not in feat_df.columns:
        if "target_label" in feat_df.columns:
            feat_df["adjusted_ret"] = feat_df["target_label"]
        elif "adjusted_ret" in ticker_df.columns: # It was in raw but got lost in build_features?
             feat_df["adjusted_ret"] = ticker_df.set_index("Date")["adjusted_ret"]
    
    # Filter for training window (2023-2024)
    train_val_df = feat_df.loc["2023-01-01":"2024-12-31"]
    
    # Split train and validation (80/20)
    split_index = int(len(train_val_df) * 0.8)
    train_df = train_val_df.iloc[:split_index]
    val_df   = train_val_df.iloc[split_index:]

    train_ds = StockDataset(train_df, args.ticker)
    val_ds   = StockDataset(val_df,   args.ticker)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = TLSTMModel(**{k: cfg[k] for k in [
        "n_ohlcv_features", "lstm_hidden_dim", "lstm_layers",
        "lstm_dropout", "nlp_in_dim", "nlp_out_dim", "head_dropout"]
    })
    trainer = Trainer(
        model, device,
        lr=cfg["lr"], ewc_lr=cfg["ewc_lr"], ewc_lambda=cfg["ewc_lambda"],
        checkpoint_dir=pathlib.Path("outputs/checkpoints"),
    )

    best_mae = trainer.train(
        train_loader, val_loader,
        epochs=cfg["epochs"], patience=cfg["patience"],
        tag=f"{args.ticker}_full",
    )
    print(f"\n  Best val MAE: {best_mae:.5f}")

    # Compute and store Fisher matrix for future EWC nudges
    trainer.compute_and_store_fisher(train_loader)
    trainer.save_checkpoint(tag=f"{args.ticker}_full", val_mae=best_mae)


def mode_ewc_nudge(args, cfg, device):
    ohlcv_path     = pathlib.Path(args.ohlcv)
    nifty_path     = pathlib.Path(args.nifty)
    handshake_dir  = pathlib.Path(args.handshake_dir) if args.handshake_dir else None
    ckpt_path      = pathlib.Path(args.checkpoint)

    df      = load_ohlcv(ohlcv_path, args.ticker)
    nifty   = load_ohlcv(nifty_path, "NIFTY50")
    feat_df = prepare_features(df, nifty)

    handshake = HandshakeDataset(handshake_dir) if handshake_dir else None
    recent_df = feat_df.iloc[-(60 + 5 * 7):]   # approx last 7 weeks of data + 60 days lookback
    ds        = StockDataset(recent_df, args.ticker, handshake)
    loader    = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    model   = TLSTMModel(**{k: cfg[k] for k in [
        "n_ohlcv_features", "lstm_hidden_dim", "lstm_layers",
        "lstm_dropout", "nlp_in_dim", "nlp_out_dim", "head_dropout"]
    })
    trainer = Trainer(model, device, ewc_lr=cfg["ewc_lr"], ewc_lambda=cfg["ewc_lambda"])
    trainer.load_checkpoint(ckpt_path)

    accepted = trainer.ewc_nudge(loader, epochs=cfg["ewc_epochs"], tag=f"{args.ticker}_nudge")
    print(f"  EWC nudge {'ACCEPTED' if accepted else 'ROLLED BACK'}")


def mode_predict(args, cfg, device):
    p_train = pathlib.Path("data/prod_train.csv")
    p_test  = pathlib.Path(args.ohlcv)
    ckpt_path = pathlib.Path(args.checkpoint)

    # Load both to provide context (StockDataset needs 60-day lookback)
    df_train = pd.read_csv(p_train, parse_dates=["Date"])
    df_test  = pd.read_csv(p_test,  parse_dates=["Date"])
    
    # Filter for ticker
    possible_tickers = [args.ticker, f"{args.ticker}.NS"]
    t_train = df_train[df_train["ticker"].isin(possible_tickers)].sort_values("Date")
    t_test  = df_test[df_test["ticker"].isin(possible_tickers)].sort_values("Date")

    # Combine: Last 120 days of train + all test
    # (120 ensures we have enough for 60-day lookback even with NaN drops)
    feat_df = pd.concat([t_train.iloc[-120:], t_test]).set_index("Date").sort_index()

    if feat_df.empty:
        raise ValueError(f"Ticker {args.ticker} not found in prediction scope")

    # Re-run feature engineering for normalization
    feat_df = build_features(feat_df)

    if "adjusted_ret" not in feat_df.columns and "target_label" in feat_df.columns:
        feat_df["adjusted_ret"] = feat_df["target_label"]

    # Now create the dataset. We want to ONLY predict the rows that came from t_test.
    ds = StockDataset(feat_df, args.ticker)
    
    # Determine the index where the actual test period starts in the Dataset
    # ds.dates should be checked against t_test starting date
    test_start_date = t_test["Date"].min()
    valid_indices = [i for i, date in enumerate(ds.dates) if pd.to_datetime(date) >= pd.to_datetime(test_start_date)]
    
    if not valid_indices:
        print(f"  ⚠️ No valid test dates found for {args.ticker} starting from {test_start_date}")
        return

    sub_ds = torch.utils.data.Subset(ds, valid_indices)
    loader = DataLoader(sub_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = TLSTMModel(**{k: cfg[k] for k in [
        "n_ohlcv_features", "lstm_hidden_dim", "lstm_layers",
        "lstm_dropout", "nlp_in_dim", "nlp_out_dim", "head_dropout"]
    })
    
    print(f"  Loading checkpoint: {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Prepare metadata for output file
    dates   = [str(ds.dates[i]) for i in valid_indices]
    tickers = [args.ticker] * len(valid_indices)

    run_prediction(
        model=model,
        loader=loader,
        device=device,
        tickers=tickers,
        dates=dates,
        model_version=ckpt_path.stem,
        output_dir=pathlib.Path("data/predictions"),
        n_mc_passes=cfg["n_mc_passes"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="T-LSTM Dual-Stream Financial Prediction")
    p.add_argument("--mode",       choices=["train", "predict", "ewc_nudge"], required=True)
    p.add_argument("--ticker",     default="RELIANCE", help="NSE ticker symbol")
    p.add_argument("--ohlcv",      default="data/price/{ticker}/ohlcv.csv", help="Path to OHLCV CSV")
    p.add_argument("--nifty",      default="data/price/NIFTY50/ohlcv.csv",  help="Path to Nifty 50 CSV")
    p.add_argument("--handshake_dir", default="data/handshake/",            help="Dir containing fused_YYYY-MM-DD.csv")
    p.add_argument("--checkpoint", default=None,                            help="Path to model checkpoint (.pt)")
    p.add_argument("--train_years", type=int, default=17,                   help="Years for initial training window")
    
    args = p.parse_args()
    args.ohlcv = args.ohlcv.replace("{ticker}", args.ticker)
    return args


if __name__ == "__main__":
    args   = parse_args()
    cfg    = DEFAULT_CONFIG
    device = get_device()

    if args.mode == "train":
        mode_train(args, cfg, device)
    elif args.mode == "ewc_nudge":
        mode_ewc_nudge(args, cfg, device)
    elif args.mode == "predict":
        mode_predict(args, cfg, device)
