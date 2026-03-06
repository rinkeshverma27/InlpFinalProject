"""
Prediction / Inference Script  (Blueprint §6.1)

Loads a trained checkpoint, runs MC Dropout inference on a dataset,
and writes predictions.csv matching the blueprint output schema:
  date, ticker, predicted_pct, conf_low, conf_high, direction,
  magnitude_label, model_version, actual_pct, gap
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

import torch
import pandas as pd
import numpy as np
from data import get_test_dataloader, StockDataset
from models.baseline_lstm import BaselineLSTM
from models.tlstm_hybrid import TLSTMHybrid


# ── Direction & magnitude rules (Blueprint §6.1) ────────────────────────
def get_direction(pred_pct):
    if pred_pct > 0.002:
        return "UP"
    elif pred_pct < -0.002:
        return "DOWN"
    return "NEUTRAL"


def get_magnitude(ci_width):
    if ci_width < 0.005:
        return "Strong"
    elif ci_width < 0.012:
        return "Moderate"
    return "Weak"


def main():
    parser = argparse.ArgumentParser(description="Prediction / Inference Pipeline")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "hybrid"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt checkpoint")
    parser.add_argument("--data", type=str, default="data1/1year-test.csv", help="CSV to predict on")
    parser.add_argument("--stocks", type=str, default="RELIANCE.NS,TCS.NS,HDFCBANK.NS")
    parser.add_argument("--output", type=str, default="data/predictions/predictions.csv")
    parser.add_argument("--mc-passes", type=int, default=50, help="MC Dropout forward passes for CI")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    tickers = [t.strip() for t in args.stocks.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Tickers: {tickers}")

    # ── Load raw DF for date/ticker metadata ─────────────────────────────
    raw_df = pd.read_csv(args.data)
    raw_df['Date'] = pd.to_datetime(raw_df['Date'], utc=True)
    raw_df = raw_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    raw_df = raw_df[raw_df['Ticker'].isin(tickers)]

    # ── Detect input_dim ─────────────────────────────────────────────────
    feature_cols = StockDataset.FEATURE_COLS
    available = [c for c in feature_cols if c in raw_df.columns]
    input_dim = len(available)

    # ── Build model ──────────────────────────────────────────────────────
    if args.model == "baseline":
        model = BaselineLSTM(input_dim=input_dim)
    else:
        model = TLSTMHybrid(price_input_dim=input_dim)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Extract model version from checkpoint metadata
    ckpt_path = Path(args.checkpoint)
    model_version = ckpt_path.stem  # e.g. model_2026-03-05

    # ── DataLoader ───────────────────────────────────────────────────────
    test_loader = get_test_dataloader(args.data, tickers, batch_size=args.batch_size)

    # ── Inference with MC Dropout ────────────────────────────────────────
    all_means, all_lows, all_highs, all_targets = [], [], [], []

    for x, y, _ in test_loader:
        x = x.to(device)
        if args.model == "baseline":
            mean, lo, hi = model.mc_predict(x, n_passes=args.mc_passes)
        else:
            mean, lo, hi = model.mc_predict(x, nlp_embed=None, n_passes=args.mc_passes)

        all_means.extend(mean.cpu().numpy().tolist())
        all_lows.extend(lo.cpu().numpy().tolist())
        all_highs.extend(hi.cpu().numpy().tolist())
        all_targets.extend(y.numpy().tolist())

    # ── Build metadata for each prediction row ───────────────────────────
    # We need to map predictions back to (date, ticker) pairs.
    # Construct an ordered list of (date, ticker) matching the test dataset.
    meta_rows = []
    for ticker in tickers:
        t_df = raw_df[raw_df['Ticker'] == ticker].reset_index(drop=True)
        if len(t_df) <= 60:
            continue
        # StockDataset uses valid_indices starting from seq_length=60
        ds = StockDataset(t_df)
        for vi in ds.valid_indices:
            row = t_df.iloc[vi]
            meta_rows.append({
                'date': row['Date'].strftime('%Y-%m-%d'),
                'ticker': ticker,
            })

    # Truncate to actual prediction count
    n = min(len(meta_rows), len(all_means))
    meta_rows = meta_rows[:n]
    all_means = all_means[:n]
    all_lows = all_lows[:n]
    all_highs = all_highs[:n]
    all_targets = all_targets[:n]

    # ── Build predictions.csv ────────────────────────────────────────────
    records = []
    for i, meta in enumerate(meta_rows):
        pred_pct = all_means[i]
        ci_width = all_highs[i] - all_lows[i]
        actual = all_targets[i]

        records.append({
            'date': meta['date'],
            'ticker': meta['ticker'],
            'predicted_pct': round(pred_pct, 6),
            'conf_low': round(all_lows[i], 6),
            'conf_high': round(all_highs[i], 6),
            'direction': get_direction(pred_pct),
            'magnitude_label': get_magnitude(ci_width),
            'model_version': model_version,
            'active_trust_weights': '{}',  # populated when sentiment pipeline is live
            'actual_pct': round(actual, 6),
            'gap': round(abs(pred_pct - actual), 6),
        })

    out_df = pd.DataFrame(records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # ── Summary stats ────────────────────────────────────────────────────
    mae = np.mean(np.abs(np.array(all_means) - np.array(all_targets)))
    preds_arr = np.array(all_means)
    tgts_arr = np.array(all_targets)
    dir_acc = np.mean(np.sign(preds_arr) == np.sign(tgts_arr))

    high_vol = np.abs(tgts_arr) > 0.015
    hv_acc = np.mean(np.sign(preds_arr[high_vol]) == np.sign(tgts_arr[high_vol])) if high_vol.sum() > 0 else 0

    print(f"\n{'='*50}")
    print(f"Predictions saved to: {out_path}")
    print(f"Total predictions:    {len(records)}")
    print(f"MAE:                  {mae:.4f}")
    print(f"Direction Accuracy:   {dir_acc:.3f}")
    print(f"High-Vol Accuracy:    {hv_acc:.3f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
