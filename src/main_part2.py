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
from torch.utils.data import DataLoader

from src.utils.features import build_features, build_labels
from src.data.dataset import StockDataset, HandshakeDataset, walk_forward_splits
from src.models.tlstm import TLSTMModel
from src.training.trainer import Trainer
from src.training.predict import run_prediction


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_ohlcv_features": 9,
    "lstm_hidden_dim":  256,
    "lstm_layers":      2,
    "lstm_dropout":     0.3,
    "nlp_in_dim":       514,
    "nlp_out_dim":      512,
    "head_dropout":     0.3,
    "batch_size":       64,
    "epochs":           50,
    "patience":         5,
    "ewc_epochs":       3,
    "ewc_lambda":       400.0,
    "ewc_lr":           1e-5,
    "lr":               1e-3,
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
    df = df[df["ticker"] == ticker].set_index("Date").sort_index()
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
    ohlcv_path   = pathlib.Path(args.ohlcv)
    nifty_path   = pathlib.Path(args.nifty)
    handshake_dir = pathlib.Path(args.handshake_dir) if args.handshake_dir else None

    df      = load_ohlcv(ohlcv_path, args.ticker)
    nifty   = load_ohlcv(nifty_path, "NIFTY50")
    feat_df = prepare_features(df, nifty)
    splits  = walk_forward_splits(feat_df, train_years=args.train_years)

    handshake = HandshakeDataset(handshake_dir) if handshake_dir else None

    # Use the LAST split for a final train/val evaluation
    train_df, val_df = splits[-1]

    train_ds = StockDataset(train_df, args.ticker, handshake)
    val_ds   = StockDataset(val_df,   args.ticker, handshake)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model   = TLSTMModel(**{k: cfg[k] for k in [
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
    ohlcv_path     = pathlib.Path(args.ohlcv)
    nifty_path     = pathlib.Path(args.nifty)
    handshake_dir  = pathlib.Path(args.handshake_dir) if args.handshake_dir else None
    ckpt_path      = pathlib.Path(args.checkpoint)

    df      = load_ohlcv(ohlcv_path, args.ticker)
    nifty   = load_ohlcv(nifty_path, "NIFTY50")
    feat_df = prepare_features(df, nifty)

    handshake = HandshakeDataset(handshake_dir) if handshake_dir else None
    ds        = StockDataset(feat_df, args.ticker, handshake)
    loader    = DataLoader(ds, batch_size=1, shuffle=False)

    model   = TLSTMModel(**{k: cfg[k] for k in [
        "n_ohlcv_features", "lstm_hidden_dim", "lstm_layers",
        "lstm_dropout", "nlp_in_dim", "nlp_out_dim", "head_dropout"]
    })

    import torch
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Get dates from the dataset (handles both 'Date' and 'date' columns)
    date_col = "date" if "date" in ds.df.columns else "Date"
    dates   = [str(d) for d in ds.df[date_col].tolist()]
    tickers = [args.ticker] * len(ds)

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
