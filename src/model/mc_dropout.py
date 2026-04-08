"""
src/model/mc_dropout.py — Monte Carlo Dropout inference.

Runs the model in train() mode (dropout active) for N forward passes.
Mean = direction probability. Variance = uncertainty estimate.
High variance → model is confused → abstain.
"""

import torch
import numpy as np
from typing import Tuple


def mc_predict(
    model,
    price_seq: torch.Tensor,
    sentiment_seq: torch.Tensor,
    n_passes: int,
    confidence_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run Monte Carlo Dropout inference.

    Args:
        model                : DualStreamLSTM instance.
        price_seq            : [B, T, 11]
        sentiment_seq        : [B, T, 9]
        n_passes             : Number of stochastic forward passes.
        confidence_threshold : Abstain if variance > (1 - threshold).

    Returns:
        mean      : [B] mean probability of UP (direction call)
        variance  : [B] variance across passes (uncertainty)
        abstain   : [B] bool tensor — True = do not act on this prediction
    """
    model.train()   # KEEP dropout active during inference
    preds = []

    with torch.no_grad():
        for _ in range(n_passes):
            p = model(price_seq, sentiment_seq)   # [B]
            preds.append(p)

    preds    = torch.stack(preds, dim=0)           # [N_passes, B]
    mean     = preds.mean(dim=0)                   # [B]
    variance = preds.var(dim=0)                    # [B]
    abstain  = variance > (1.0 - confidence_threshold)

    return mean, variance, abstain


def predict_single(
    model,
    price_seq: torch.Tensor,
    sentiment_seq: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    Convenience wrapper for predicting a single sample.

    Returns:
        dict with:
            direction   : "UP" | "DOWN" | "ABSTAIN"
            probability : float [0,1] — probability of UP
            variance    : float — MC uncertainty
            confidence  : float — min(prob, 1-prob) distance from 0.5, scaled
    """
    m_cfg    = cfg.get("model", {})
    n_passes = m_cfg.get("mc_dropout_passes", 30)
    thresh   = m_cfg.get("confidence_threshold", 0.65)

    if price_seq.dim() == 2:
        price_seq     = price_seq.unsqueeze(0)
    if sentiment_seq.dim() == 2:
        sentiment_seq = sentiment_seq.unsqueeze(0)

    price_seq     = price_seq.to(device)
    sentiment_seq = sentiment_seq.to(device)

    mean, var, abstain = mc_predict(model, price_seq, sentiment_seq, n_passes, thresh)

    prob   = mean[0].item()
    varval = var[0].item()
    do_abs = abstain[0].item()

    if do_abs:
        direction = "ABSTAIN"
    else:
        direction = "UP" if prob >= 0.5 else "DOWN"

    return {
        "direction":   direction,
        "probability": round(prob, 4),
        "variance":    round(varval, 4),
        "confidence":  round(1.0 - varval, 4),
    }
