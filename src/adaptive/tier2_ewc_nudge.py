"""
src/adaptive/tier2_ewc_nudge.py — Weekly EWC gradient nudge.

Applies a small gradient update on the last N days of data using EWC penalty
to prevent catastrophic forgetting of historical training.
"""

import torch
import torch.nn as nn
from typing import Optional

from src.utils.logger import get_logger
from src.utils.paths import PRODUCTION_DIR
from src.model.dual_stream_lstm import build_model
from src.model.ewc import EWCLoss, load_ewc_state

log = get_logger("tier2_ewc")


def run_tier2(recent_loader, cfg: dict, device: Optional[torch.device] = None):
    """
    Run a small EWC gradient update on recent data.

    Args:
        recent_loader: DataLoader of the last `tier2_nudge_days` trading days.
        cfg          : Full config dict.
        device       : torch.device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapt = cfg.get("adaptive", {})
    lamda = adapt.get("tier2_ewc_lambda", 400)
    lr    = adapt.get("tier2_nudge_lr",   1e-5)

    try:
        fisher, ref_params = load_ewc_state(PRODUCTION_DIR)
    except FileNotFoundError as e:
        log.warning(f"[TIER-2] Cannot run — {e}")
        return

    model = build_model(cfg).to(device)
    state = torch.load(PRODUCTION_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    ewc_loss = EWCLoss(model, fisher, ref_params, lamda)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    total_loss = 0.0
    for price, sent, labels, _ in recent_loader:
        price, sent, labels = price.to(device), sent.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(price, sent)
        loss = criterion(out, labels) + ewc_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    log.info(f"[TIER-2] EWC nudge complete. Total loss={total_loss:.4f}")

    # Overwrite production model with nudged weights
    state["model"] = model.state_dict()
    torch.save(state, PRODUCTION_DIR / "best_model.pt")
    log.info("[TIER-2] Production model updated with EWC nudge.")
