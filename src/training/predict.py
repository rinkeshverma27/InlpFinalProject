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

    for ohlcv_seq, nlp_vec, _ in loader:
        ohlcv_seq = ohlcv_seq.to(device)
        nlp_vec   = nlp_vec.to(device)
        result    = model.mc_dropout_predict(ohlcv_seq, nlp_vec, n_passes=n_mc_passes)
        all_mean.append(result["mean"].cpu())
        all_ci_low.append(result["ci_low"].cpu())
        all_ci_high.append(result["ci_high"].cpu())

    pred_pct  = torch.cat(all_mean).numpy().flatten()
    ci_low    = torch.cat(all_ci_low).numpy().flatten()
    ci_high   = torch.cat(all_ci_high).numpy().flatten()

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
        "actual_pct":      None,    # filled in at 10:05 AM next day
        "gap":             None,    # filled in at 10:05 AM next day
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    out_path = output_dir / f"predictions_{today_str}.csv"
    
    df.to_csv(out_path, index=False)
    print(f"  Predictions written → {out_path}  ({len(df)} rows)")

    return df


def fill_actuals(
    predictions_csv: pathlib.Path,
    actual_prices: dict[str, float],
    date: str,
) -> pd.DataFrame:
    """
    Fill in actual_pct and gap columns once 10:05 AM price is captured.
    Called by Machine 3's cron job after market opens.

    `actual_prices` : { ticker: actual_pct_move }
    """
    df = pd.read_csv(predictions_csv)
    mask = df["date"] == date

    df.loc[mask, "actual_pct"] = df.loc[mask, "ticker"].map(actual_prices)
    df.loc[mask, "gap"]        = (df.loc[mask, "predicted_pct"] - df.loc[mask, "actual_pct"]).abs()

    df.to_csv(predictions_csv, index=False)
    print(f"  Actuals filled for {date} in {predictions_csv}")
    return df
