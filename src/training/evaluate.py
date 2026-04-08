"""
src/training/evaluate.py — Full performance matrix for directional prediction.

Sections:
  1. Overall accuracy, precision, recall, F1 (macro + per-class)
  2. Directional accuracy at multiple confidence tiers
  3. Abstention rate & coverage
  4. Per-ticker breakdown
  5. Temporal analysis: rolling 30-day accuracy
  6. Calibration: Expected Calibration Error (ECE) + reliability diagram data
  7. Confusion matrix
  8. Save full report to logs/eval_report/
"""

import pathlib
import json
import numpy as np
import pandas as pd
from typing import Optional, List

from src.utils.logger import get_logger
from src.utils.paths import EVAL_REPORT_DIR

log = get_logger("evaluate")


# ─────────────────────────────────────────────────────────────────────────────
# Core metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))

def _precision_recall_f1(y_true, y_pred, cls):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = ((y_pred == cls) & (y_true == cls)).sum()
    fp = ((y_pred == cls) & (y_true != cls)).sum()
    fn = ((y_pred != cls) & (y_true == cls)).sum()
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return float(p), float(r), float(f1)

def _ece(probs, labels, n_bins=10):
    """Expected Calibration Error (ECE)."""
    probs  = np.array(probs)
    labels = np.array(labels)
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc   = labels[mask].mean()
        conf  = probs[mask].mean()
        ece  += mask.sum() / n * abs(acc - conf)
    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    results_df: pd.DataFrame,
    cfg: dict,
    split_name: str = "test",
    output_dir: Optional[pathlib.Path] = None,
) -> dict:
    """
    Compute the full performance matrix.

    Args:
        results_df  : DataFrame with columns:
                        date, ticker, label (0/1), prob (float), variance (float),
                        direction ("UP"/"DOWN"/"ABSTAIN"), split ("train"/"val"/"test")
        cfg         : Full config dict.
        split_name  : Which split to evaluate ("val" or "test").
        output_dir  : Where to save the JSON report (default: logs/eval_report/).

    Returns:
        dict — full performance matrix (also printed to console + saved as JSON).
    """
    df = results_df[results_df["split"] == split_name].copy()
    if df.empty:
        log.warning(f"No rows found for split='{split_name}'.")
        return {}

    eval_cfg   = cfg.get("evaluation", {})
    conf_tiers = eval_cfg.get("confidence_tiers", [0.55, 0.65, 0.75, 0.85])
    roll_win   = eval_cfg.get("rolling_window_days", 30)
    cal_bins   = eval_cfg.get("calibration_bins", 10)

    # Columns needed
    df["pred_binary"] = (df["direction"] == "UP").astype(int)
    df["abstained"]   = (df["direction"] == "ABSTAIN")
    df_active         = df[~df["abstained"]]

    # ── 1. Overall metrics ────────────────────────────────────────────────────
    overall_acc = _accuracy(df_active["label"], df_active["pred_binary"])
    p_up, r_up, f1_up   = _precision_recall_f1(df_active["label"], df_active["pred_binary"], 1)
    p_dn, r_dn, f1_dn   = _precision_recall_f1(df_active["label"], df_active["pred_binary"], 0)
    macro_f1 = (f1_up + f1_dn) / 2

    # ── 2. Confidence tier accuracy ───────────────────────────────────────────
    tier_metrics = {}
    for thresh in conf_tiers:
        high_conf = df[(df["variance"] <= (1 - thresh)) & ~df["abstained"]]
        if len(high_conf) == 0:
            tier_metrics[f"acc@conf>{thresh}"] = None
        else:
            tier_metrics[f"acc@conf>{thresh}"] = round(
                _accuracy(high_conf["label"], (high_conf["direction"] == "UP").astype(int)), 4
            )
            tier_metrics[f"n@conf>{thresh}"] = len(high_conf)

    # ── 3. Abstention rate ────────────────────────────────────────────────────
    abstention_rate = float(df["abstained"].mean())
    coverage        = 1.0 - abstention_rate

    # ── 4. Per-ticker breakdown ───────────────────────────────────────────────
    per_ticker = {}
    for ticker, tdf in df_active.groupby("ticker"):
        ta  = _accuracy(tdf["label"], tdf["pred_binary"])
        tf1 = (_precision_recall_f1(tdf["label"], tdf["pred_binary"], 1)[2] +
               _precision_recall_f1(tdf["label"], tdf["pred_binary"], 0)[2]) / 2
        per_ticker[ticker] = {
            "n_predictions": len(tdf),
            "accuracy":      round(ta, 4),
            "macro_f1":      round(tf1, 4),
            "abstention_rate": round(
                df[df["ticker"] == ticker]["abstained"].mean(), 4
            ),
        }

    # ── 5. Temporal analysis (rolling accuracy) ───────────────────────────────
    df_active_sorted = df_active.sort_values("date").copy()
    df_active_sorted["date"] = pd.to_datetime(df_active_sorted["date"])
    df_active_sorted["correct"] = (
        df_active_sorted["pred_binary"] == df_active_sorted["label"]
    ).astype(int)
    
    # Needs a DatetimeIndex to support '30D' offset strings
    rolling_acc = (
        df_active_sorted.set_index("date")["correct"]
        .rolling(f"{roll_win}D")
        .mean()
        .dropna()
    )
    temporal = {
        "rolling_window_days": roll_win,
        "rolling_accuracy_min":  round(float(rolling_acc.min()), 4),
        "rolling_accuracy_max":  round(float(rolling_acc.max()), 4),
        "rolling_accuracy_mean": round(float(rolling_acc.mean()), 4),
    }

    # ── 6. Calibration (ECE) ──────────────────────────────────────────────────
    ece_val = _ece(df_active["prob"].values, df_active["label"].values, cal_bins)

    # Reliability diagram data (for plotting)
    bins     = np.linspace(0, 1, cal_bins + 1)
    rel_data = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df_active["prob"] >= lo) & (df_active["prob"] < hi)
        if mask.sum() > 0:
            rel_data.append({
                "bin_mid": round((lo + hi) / 2, 2),
                "mean_confidence": round(float(df_active.loc[mask, "prob"].mean()), 4),
                "accuracy":        round(float(df_active.loc[mask, "label"].mean()), 4),
                "count":           int(mask.sum()),
            })

    # ── 7. Confusion matrix ───────────────────────────────────────────────────
    y_true = np.array(df_active["label"])
    y_pred = np.array(df_active["pred_binary"])
    cm = {
        "TP": int(((y_pred == 1) & (y_true == 1)).sum()),
        "FP": int(((y_pred == 1) & (y_true == 0)).sum()),
        "TN": int(((y_pred == 0) & (y_true == 0)).sum()),
        "FN": int(((y_pred == 0) & (y_true == 1)).sum()),
    }

    # ── Assemble report ───────────────────────────────────────────────────────
    report = {
        "split":            split_name,
        "n_total":          len(df),
        "n_active":         len(df_active),
        "abstention_rate":  round(abstention_rate, 4),
        "coverage":         round(coverage, 4),
        "overall": {
            "directional_accuracy": round(overall_acc, 4),
            "macro_f1":             round(macro_f1, 4),
            "UP":   {"precision": round(p_up, 4), "recall": round(r_up, 4), "f1": round(f1_up, 4)},
            "DOWN": {"precision": round(p_dn, 4), "recall": round(r_dn, 4), "f1": round(f1_dn, 4)},
        },
        "confidence_tier_accuracy": tier_metrics,
        "per_ticker":       per_ticker,
        "temporal":         temporal,
        "calibration": {
            "ece":              round(ece_val, 4),
            "reliability_data": rel_data,
        },
        "confusion_matrix": cm,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"  PERFORMANCE MATRIX — split={split_name}")
    log.info("=" * 60)
    log.info(f"  Overall Accuracy  : {overall_acc:.2%}")
    log.info(f"  Macro F1          : {macro_f1:.4f}")
    log.info(f"  Abstention Rate   : {abstention_rate:.2%}  (coverage={coverage:.2%})")
    log.info(f"  ECE (calibration) : {ece_val:.4f}  (lower=better, 0=perfect)")
    for tier, acc in tier_metrics.items():
        if "acc@" in tier and acc is not None:
            n_key = tier.replace("acc@", "n@")
            n = tier_metrics.get(n_key, "?")
            log.info(f"  {tier:25s}: {acc:.2%}  (n={n})")
    log.info("-" * 60)
    for t, m in per_ticker.items():
        log.info(f"  [{t:12s}] acc={m['accuracy']:.2%} f1={m['macro_f1']:.4f} abstain={m['abstention_rate']:.2%}")
    log.info("=" * 60)

    # ── Save JSON report ──────────────────────────────────────────────────────
    out_dir = output_dir or EVAL_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"report_{split_name}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info(f"  Full report saved → {report_path}")

    return report
