"""
src/utils/plotting.py — Lightweight plotting helpers for training and evaluation.

All plots are written to disk using a non-interactive backend so they can be
generated inside CLI runs and restricted environments.
"""

import pathlib
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

from src.utils.logger import get_logger

log = get_logger("plotting")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except Exception as exc:  # pragma: no cover - environment dependent
    plt = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _ready() -> bool:
    if plt is None:
        log.warning(f"[PLOTS] matplotlib unavailable, skipping plots: {_IMPORT_ERROR}")
        return False
    return True


def plot_training_history(history: Sequence[Mapping], out_dir: pathlib.Path) -> None:
    """Generate training loss/accuracy/LR graphs from epoch history."""
    if not history or not _ready():
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]
    lr = [row.get("lr", np.nan) for row in history]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(epochs, train_loss, label="train_loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="val_loss", linewidth=2)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train_acc", linewidth=2)
    axes[1].plot(epochs, val_acc, label="val_acc", linewidth=2)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, lr, label="learning_rate", linewidth=2, color="tab:green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, label="train_loss", linewidth=2)
    ax.plot(epochs, val_loss, label="val_loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Epochs")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_vs_epochs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation_bundle(report: Mapping, out_dir: pathlib.Path) -> None:
    """Generate several evaluation plots from the saved report dict."""
    if not report or not _ready():
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_per_ticker_accuracy(report, out_dir)
    _plot_confidence_tiers(report, out_dir)
    _plot_reliability(report, out_dir)
    _plot_rolling_accuracy(report, out_dir)
    _plot_confusion_matrix(report, out_dir)


def _plot_per_ticker_accuracy(report: Mapping, out_dir: pathlib.Path) -> None:
    data = report.get("per_ticker", {})
    if not data:
        return

    tickers = list(data.keys())
    acc = [100 * data[t]["accuracy"] for t in tickers]
    abstain = [100 * data[t]["abstention_rate"] for t in tickers]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tickers))
    width = 0.38
    ax.bar(x - width / 2, acc, width, label="Accuracy %")
    ax.bar(x + width / 2, abstain, width, label="Abstention %")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=35, ha="right")
    ax.set_ylabel("Percent")
    ax.set_title(f"Per-Ticker Metrics ({report.get('split', 'test')})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"per_ticker_{report.get('split', 'test')}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confidence_tiers(report: Mapping, out_dir: pathlib.Path) -> None:
    tier_data = report.get("confidence_tier_accuracy", {})
    keys = [k for k in tier_data.keys() if k.startswith("acc@") and tier_data[k] is not None]
    if not keys:
        return

    labels = keys
    vals = [100 * tier_data[k] for k in keys]
    counts = [tier_data.get(k.replace("acc@", "n@"), 0) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, vals, color="tab:orange")
    ax.set_ylabel("Accuracy %")
    ax.set_title("Accuracy by Confidence / Variance Tier")
    ax.grid(axis="y", alpha=0.3)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"n={count}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"confidence_tiers_{report.get('split', 'test')}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_reliability(report: Mapping, out_dir: pathlib.Path) -> None:
    rel = report.get("calibration", {}).get("reliability_data", [])
    if not rel:
        return

    x = [row["mean_confidence"] for row in rel]
    y = [row["accuracy"] for row in rel]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(x, y, marker="o", linewidth=2, label="Observed")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability Diagram ({report.get('split', 'test')})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"reliability_{report.get('split', 'test')}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_rolling_accuracy(report: Mapping, out_dir: pathlib.Path) -> None:
    temporal = report.get("temporal", {})
    series = temporal.get("rolling_accuracy_series", [])
    if not series:
        return

    dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in series]
    values = [100 * row["accuracy"] for row in series]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(dates, values, linewidth=2, color="tab:purple")
    ax.set_ylabel("Accuracy %")
    ax.set_title(f"{temporal.get('rolling_window_days', 30)}-Day Rolling Accuracy")
    ax.grid(alpha=0.3)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / f"rolling_accuracy_{report.get('split', 'test')}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(report: Mapping, out_dir: pathlib.Path) -> None:
    cm = report.get("confusion_matrix", {})
    if not cm:
        return

    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred DOWN", "Pred UP"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True DOWN", "True UP"])
    ax.set_title(f"Confusion Matrix ({report.get('split', 'test')})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{report.get('split', 'test')}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
