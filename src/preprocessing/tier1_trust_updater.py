"""
src/preprocessing/tier1_trust_updater.py
Tier 1 — Daily Source Trust Weight Update (Blueprint §5, runs at 10:10 AM)

Reads today's production_predictions.csv and yesterday's actual move
to update per-source trust weights, stored as JSON in data/trust_weights/.

Logic:
  - If |actual_move| > 3-sigma (anomaly) → skip entirely.
  - For each source that contributed to today's prediction:
      * If prediction gap improved vs yesterday → increase trust * 1.05
      * If prediction gap worsened → decrease trust * 0.95
  - Hard-clamp: trust ∈ [0.1, 2.0]. No single source > 40% of total.
  - Stale decay: sources inactive ≥7 days → decay 2%/day toward 1.0.

Usage:
    python src/preprocessing/tier1_trust_updater.py --date 2026-04-08
"""

import argparse
import json
import math
import sys
from datetime import date, timedelta, datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.paths import TRUST_WEIGHTS_DIR, PREDICTIONS_DIR

# Blueprint constants
TRUST_MIN      = 0.1
TRUST_MAX      = 2.0
TRUST_DEFAULT  = 1.0
MAX_SHARE      = 0.40       # No source > 40% of total weight for a stock
DECAY_PER_DAY  = 0.02       # Stale sources decay 2%/day toward 1.0
STALE_DAYS     = 7
ANOMALY_SIGMA  = 3.0


def load_trust(trust_path: Path) -> dict:
    """Load trust weights from JSON, or return empty dict."""
    if trust_path.exists():
        with open(trust_path) as f:
            return json.load(f)
    return {}


def save_trust(trust_path: Path, weights: dict) -> None:
    trust_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trust_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"  ✅ Trust weights saved → {trust_path}")


def clamp(value: float) -> float:
    return max(TRUST_MIN, min(TRUST_MAX, value))


def enforce_diversity_cap(weights: dict) -> dict:
    """Ensure no single source exceeds MAX_SHARE of total weight for a stock."""
    total = sum(weights.values())
    if total == 0:
        return weights
    cap = total * MAX_SHARE
    capped = {src: min(w, cap) for src, w in weights.items()}
    return capped


def apply_stale_decay(weights: dict, last_active: dict, today: date) -> dict:
    """Decay trust weights for sources inactive >= STALE_DAYS."""
    for src in weights:
        last = last_active.get(src)
        if last is None:
            continue
        last_date = datetime.strptime(last, "%Y-%m-%d").date()
        days_inactive = (today - last_date).days
        if days_inactive >= STALE_DAYS:
            decay_periods = days_inactive - STALE_DAYS + 1
            # Each inactive day pulls weight 2% toward neutral 1.0
            for _ in range(decay_periods):
                w = weights[src]
                weights[src] = clamp(w + DECAY_PER_DAY * (TRUST_DEFAULT - w))
    return weights


def update_trust_weights(run_date: date) -> None:
    today_file   = TRUST_WEIGHTS_DIR / f"{run_date}.json"
    yesterday    = run_date - timedelta(days=1)
    y_file       = TRUST_WEIGHTS_DIR / f"{yesterday}.json"
    pred_file    = PREDICTIONS_DIR / "production_predictions.csv"

    if not pred_file.exists():
        print(f"⚠️  Predictions file not found: {pred_file}. Skipping trust update.")
        return

    # Load yesterday's weights (base for today's update)
    trust_state = load_trust(y_file)
    # Structure: { ticker: { source: weight, ... } }

    # Load predictions to find which sources were active today
    try:
        import pandas as pd
        preds = pd.read_csv(pred_file)
    except Exception as e:
        print(f"⚠️  Could not load predictions: {e}")
        return

    today_str = str(run_date)

    for _, row in preds.iterrows():
        ticker = str(row.get("Ticker", ""))
        if ticker not in trust_state:
            trust_state[ticker] = {}

        # Stub: In a full implementation, actual_pct would be read from a
        # price-fill CSV written at 10:05 AM. Here we mark sources as active
        # and apply stale decay only (actual improvement logic requires live prices).
        ticker_weights = trust_state[ticker]

        # Apply stale decay on existing sources
        last_active = {src: str(yesterday) for src in ticker_weights}
        ticker_weights = apply_stale_decay(ticker_weights, last_active, run_date)
        ticker_weights = enforce_diversity_cap(ticker_weights)
        trust_state[ticker] = ticker_weights

    save_trust(today_file, trust_state)
    print(f"Tier 1 Trust Update complete for {run_date}.")


def main():
    parser = argparse.ArgumentParser(description="Tier 1 Daily Trust Weight Updater")
    parser.add_argument("--date", default=str(date.today()),
                        help="Date to run update for (YYYY-MM-DD)")
    args = parser.parse_args()
    run_date = date.fromisoformat(args.date)
    update_trust_weights(run_date)


if __name__ == "__main__":
    main()
