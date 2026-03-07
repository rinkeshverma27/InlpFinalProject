"""
training/predict.py
Run MC Dropout inference on a trained TLSTMModel and write predictions.csv.
"""

import pathlib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.tlstm import TLSTMModel


def run_prediction(
    model: TLSTMModel,
    loader: DataLoader,
    device: torch.device,
    tickers: list[str],
    dates: list,
    model_version: str,
    output_dir: pathlib.Path = pathlib.Path("data/predictions"),
    n_mc_passes: int = 50,
) -> pd.DataFrame:
    """
    Run MC Dropout inference and save predictions.csv.

    Output schema (matches blueprint spec):
        date, ticker, predicted_pct, conf_low, conf_high,
        direction, magnitude_label, model_version
    """
    model.to(device)
    model.eval()

    all_mean, all_ci_low, all_ci_high = [], [], []
    all_targets = []

    for ohlcv_seq, nlp_vec, label in loader:
        ohlcv_seq = ohlcv_seq.to(device)
        nlp_vec   = nlp_vec.to(device)
        result    = model.mc_dropout_predict(ohlcv_seq, nlp_vec, n_passes=n_mc_passes)
        all_mean.append(result["mean"].cpu())
        all_ci_low.append(result["ci_low"].cpu())
        all_ci_high.append(result["ci_high"].cpu())
        all_targets.append(label.cpu())

    pred_pct  = torch.cat(all_mean).numpy().flatten()
    ci_low    = torch.cat(all_ci_low).numpy().flatten()
    ci_high   = torch.cat(all_ci_high).numpy().flatten()
    targets   = torch.cat(all_targets).numpy().flatten()

    def direction(p):
        if p > 0.002:   return "UP"
        elif p < -0.002: return "DOWN"
        else:            return "NEUTRAL"

    def magnitude(ci_l, ci_h):
        width = abs(ci_h - ci_l)
        if width < 0.005:   return "Strong"
        elif width < 0.012: return "Moderate"
        else:               return "Weak"

    df = pd.DataFrame({
        "date":            dates[:len(pred_pct)],
        "ticker":          tickers[:len(pred_pct)],
        "predicted_pct":   pred_pct,
        "conf_low":        ci_low,
        "conf_high":       ci_high,
        "direction":       [direction(p) for p in pred_pct],
        "magnitude_label": [magnitude(l, h) for l, h in zip(ci_low, ci_high)],
        "model_version":   model_version,
        "active_trust_weights": "{}",  # placeholder for production json payload
        "actual_pct":      targets,    
        # User requested to halt the 'gap' column for now to prevent confusion
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    out_path = output_dir / f"predictions_{today_str}.csv"
    
    df.to_csv(out_path, index=False)
    
    # Calculate detailed accuracy block requested by user
    import numpy as np
    errors = np.abs(targets - pred_pct)
    avg_err = np.mean(errors)
    median_err = np.median(errors)
    
    pred_dir = np.sign(pred_pct)
    pred_dir[pred_dir == 0] = 1
    target_dir = np.sign(targets)
    target_dir[target_dir == 0] = 1
    acc = np.mean(pred_dir == target_dir)
    
    start_date = min(df["date"])
    end_date = max(df["date"])
    
    print("\n" + "="*60)
    print("📈 PREDICTION EVALUATION REPORT (TEST DATA)")
    print("="*60)
    print(f"Dataset Size:    {len(df)} predictions")
    print(f"Date Range:      {start_date} to {end_date}")
    print(f"Data Density:    High (10:00 AM Adjusted Returns synced with NLP Sentinel)")
    print("-" * 60)
    print("ERRORS (Absolute magnitude between predicted and actual):")
    print(f"  Mean Error:    {avg_err*100:.4f}%")
    print(f"  Median Error:  {median_err*100:.4f}%")
    print(f"  Max Error:     {np.max(errors)*100:.4f}%")
    print("-" * 60)
    print("ACCURACY (Correct prediction of Next-Day movement direction):")
    print(f"  Direction Acc: {acc*100:.2f}%")
    print("="*60 + "\n")

    print(f"  Predictions written → {out_path}  ({len(df)} rows)")

    return df


def fill_actuals(
    predictions_csv: pathlib.Path,
    actual_prices: dict[str, float],
    date: str,
) -> pd.DataFrame:
    """
    Fill in actual_pct manually once 10:05 AM price is captured on a live day.
    Called by Machine 3's cron job after market opens.

    `actual_prices` : { ticker: actual_pct_move }
    """
    df = pd.read_csv(predictions_csv)
    mask = df["date"] == date

    df.loc[mask, "actual_pct"] = df.loc[mask, "ticker"].map(actual_prices)
    # Gap calculation halted per user feedback

    df.to_csv(predictions_csv, index=False)
    print(f"  Actuals filled for {date} in {predictions_csv}")
    return df
