"""
src/predictions.py — Batch prediction helpers that write predictions.csv files.
"""

import pathlib
from typing import Iterable, List, Optional

import pandas as pd
import torch

from src.data.data_fusion import fuse_ticker
from src.model.dual_stream_lstm import build_model
from src.model.mc_dropout import predict_single
from src.utils.logger import get_logger
from src.utils.paths import PRED_LOGS_DIR, PRODUCTION_DIR

log = get_logger("predictions")


def generate_predictions_csv(
    tickers: Iterable[str],
    cfg: dict,
    device: torch.device,
    output_path: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """
    Run latest-available prediction for each ticker and save to CSV.

    Output columns:
        ticker, date, direction, prob_up, variance, confidence, n_samples
    """
    model_path = PRODUCTION_DIR / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No production model found at {model_path}. Run `python main.py train` first."
        )

    model = build_model(cfg).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    rows: List[dict] = []
    for ticker in tickers:
        fd = fuse_ticker(ticker, cfg, force=False)
        if fd is None or len(fd["dates"]) == 0:
            log.warning(f"[{ticker}] No fused data available for prediction.")
            continue

        price_seq = fd["price_seq"][-1]
        sent_seq = fd["sentiment_seq"][-1]
        result = predict_single(model, price_seq, sent_seq, cfg, device)

        rows.append({
            "ticker": ticker,
            "date": fd["dates"][-1],
            "direction": result["direction"],
            "prob_up": result["probability"],
            "variance": result["variance"],
            "confidence": result["confidence"],
            "n_samples": len(fd["dates"]),
        })

    if not rows:
        raise ValueError("No predictions generated. Check fused data availability for the selected tickers.")

    out_path = output_path or (PRED_LOGS_DIR / "predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    log.info(f"Predictions CSV saved → {out_path}")
    return out_path
