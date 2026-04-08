"""
src/adaptive/tier1_trust.py — Daily Bayesian source trust weight update.

Reads prediction logs from logs/predictions/<YYYY-MM-DD>.json, compares
to actual next-day market outcome, and updates source_weights in config.yaml
using exponential moving average (EMA).

Cost: milliseconds. Does NOT retrain any model.
"""

import json
import pathlib
import yaml
from datetime import date, timedelta
from typing import Optional

from src.utils.logger import get_logger
from src.utils.paths import PRED_LOGS_DIR, PROJECT_ROOT

log = get_logger("tier1_trust")

CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def _save_cfg(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def run_tier1(target_date: Optional[str] = None):
    """
    Update source trust weights based on yesterday's prediction accuracy.

    Args:
        target_date: "YYYY-MM-DD" prediction date to evaluate (default: yesterday).
    """
    cfg     = _load_cfg()
    decay   = cfg.get("adaptive", {}).get("tier1_decay", 0.95)
    weights = cfg.get("source_weights", {})

    if target_date is None:
        target_date = str(date.today() - timedelta(days=1))

    log_file = PRED_LOGS_DIR / f"{target_date}.json"
    if not log_file.exists():
        log.warning(
            f"[TIER-1] No prediction log for {target_date} at {log_file}.\n"
            f"Tier-1 update skipped — no data to learn from yet."
        )
        return

    with open(log_file) as f:
        preds = json.load(f)

    source_correct: dict[str, list] = {}
    for pred in preds:
        src     = pred.get("source", "default")
        correct = pred.get("correct", None)
        if correct is None:
            continue
        source_correct.setdefault(src, []).append(int(correct))

    if not source_correct:
        log.info("[TIER-1] No labeled outcomes found in prediction log — skipping.")
        return

    updates = []
    for src, outcomes in source_correct.items():
        accuracy = sum(outcomes) / len(outcomes)
        matched_key = next((k for k in weights if k in src.lower()), "default")
        old_w = weights.get(matched_key, 0.5)
        new_w = decay * old_w + (1 - decay) * accuracy
        new_w = round(max(0.1, min(1.0, new_w)), 4)   # clamp to [0.1, 1.0]
        weights[matched_key] = new_w
        updates.append(f"  {matched_key}: {old_w:.4f} → {new_w:.4f} (acc={accuracy:.2%})")

    cfg["source_weights"] = weights
    _save_cfg(cfg)
    log.info(f"[TIER-1] Source weights updated for {target_date}:\n" + "\n".join(updates))
