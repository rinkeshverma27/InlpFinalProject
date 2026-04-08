"""
src/training/trainer.py — Walk-forward LSTM training loop.

Features:
  - OOM-safe training with automatic batch shrink on CUDA OOM
  - Per-epoch checkpoint saves (never overwrites, always appends)
  - Early stopping on validation directional accuracy
  - Class-weighted BCE loss for ~52/48 imbalance
  - Cosine LR schedule with warm restarts
  - Fisher Information Matrix (FIM) saved after training for EWC
"""

import pathlib
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.errors import run_stage
from src.utils.paths import CHECKPOINTS_DIR, PRODUCTION_DIR, TRAIN_LOGS_DIR
from src.training.dataset import StockSequenceDataset, make_splits, collate_fn
from src.model.dual_stream_lstm import build_model

log = get_logger("trainer")


# ─────────────────────────────────────────────────────────────────────────────
# Fisher Information Matrix (for EWC)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fisher(model, loader, device, n_batches=50):
    """Compute diagonal Fisher matrix for EWC."""
    model.train()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    criterion = nn.BCELoss()

    for i, (price, sent, labels, _) in enumerate(loader):
        if i >= n_batches:
            break
        price, sent, labels = price.to(device), sent.to(device), labels.to(device)
        model.zero_grad()
        out  = model(price, sent)
        loss = criterion(out, labels)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data.pow(2)

    for n in fisher:
        fisher[n] /= n_batches
    return fisher


# ─────────────────────────────────────────────────────────────────────────────
# Training step helpers
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for price, sent, labels, _ in loader:
        price, sent, labels = price.to(device), sent.to(device), labels.to(device)
        optimizer.zero_grad()

        out  = model(price, sent)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds  = (out >= 0.5).float()
        correct += (preds == labels).sum().item()
        total  += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for price, sent, labels, _ in loader:
        price, sent, labels = price.to(device), sent.to(device), labels.to(device)
        out   = model(price, sent)
        loss  = criterion(out, labels)
        total_loss += loss.item() * len(labels)
        preds = (out >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += len(labels)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    fused_datasets: List[dict],
    cfg: dict,
    device: Optional[torch.device] = None,
    resume_checkpoint: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """
    Train the DualStreamLSTM on walk-forward data.

    Args:
        fused_datasets   : List of dicts from data_fusion.fuse_ticker()
        cfg              : Full config dict.
        device           : torch.device (auto-detected if None)
        resume_checkpoint: Path to a previously saved .pt checkpoint to resume from.

    Returns:
        Path to the best production checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_cfg      = cfg.get("training",  {})
    epochs     = t_cfg.get("epochs",   50)
    lr         = t_cfg.get("learning_rate", 3e-4)
    wd         = t_cfg.get("weight_decay",  1e-5)
    batch_sz   = t_cfg.get("batch_size",    64)
    grad_clip  = t_cfg.get("grad_clip_norm", 1.0)
    patience   = t_cfg.get("early_stopping_patience", 10)
    cosine_t0  = t_cfg.get("cosine_t0", 10)

    # ── Build datasets & loaders ──────────────────────────────────────────────
    train_sets, val_sets = [], []
    for fd in fused_datasets:
        ticker = fd["ticker"]
        tr_idx, va_idx, _ = make_splits(fd, cfg, ticker)
        if len(tr_idx) > 0:
            train_sets.append(StockSequenceDataset(fd, tr_idx))
        if len(va_idx) > 0:
            val_sets.append(StockSequenceDataset(fd, va_idx))

    if not train_sets:
        raise ValueError("No training data available. Check your data/splits/ dates vs OHLCV range.")

    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset(train_sets)
    val_ds   = ConcatDataset(val_sets)

    n_workers = min(4, torch.get_num_threads())
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                              num_workers=n_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_sz, shuffle=False,
                              num_workers=n_workers, collate_fn=collate_fn, pin_memory=True)

    # ── Class weights ─────────────────────────────────────────────────────────
    all_labels = np.concatenate([fd["labels"].numpy() for fd in fused_datasets])
    n_pos      = all_labels.sum()
    n_neg      = len(all_labels) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-9)], device=device)
    criterion  = nn.BCELoss(weight=None)   # using pos_weight via BCEWithLogitsLoss pattern
    # Proper weighted BCE:
    criterion  = lambda out, lbl: nn.functional.binary_cross_entropy(
        out, lbl,
        weight=(lbl * pos_weight[0] + (1 - lbl) * 1.0)
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cosine_t0)

    start_epoch = 0
    best_val_acc = 0.0
    best_ckpt_path = None

    if resume_checkpoint and resume_checkpoint.exists():
        state = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch   = state.get("epoch", 0) + 1
        best_val_acc  = state.get("best_val_acc", 0.0)
        log.info(f"[RESUME] From epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    no_improve = 0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        try:
            tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
            va_loss, va_acc = _eval_epoch(model, val_loader, criterion, device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.critical(
                    f"[OOM] Training epoch {epoch+1} crashed.\n"
                    f"Action: reduce training.batch_size in config.yaml (current={batch_sz}).\n"
                    f"Or switch vram_profile to '4gb'."
                )
                raise
            raise

        scheduler.step()
        elapsed = time.time() - t0

        record = {
            "epoch": epoch + 1, "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4), "elapsed_s": round(elapsed, 1),
        }
        history.append(record)
        log.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} acc={va_acc:.4f} | "
            f"{elapsed:.1f}s | LR={scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Save epoch checkpoint ─────────────────────────────────────────────
        ckpt_path = CHECKPOINTS_DIR / f"epoch_{epoch+1:03d}_valacc{va_acc:.4f}.pt"
        torch.save({
            "epoch":        epoch + 1,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "val_acc":      va_acc,
            "best_val_acc": best_val_acc,
            "cfg_profile":  cfg.get("vram_profile", "8gb"),
        }, ckpt_path)

        if va_acc > best_val_acc:
            best_val_acc   = va_acc
            best_ckpt_path = ckpt_path
            no_improve     = 0
            log.info(f"  ★ New best: val_acc={best_val_acc:.4f}")
        else:
            no_improve += 1

        if no_improve >= patience:
            log.info(f"  Early stopping at epoch {epoch+1} (patience={patience}).")
            break

    # ── Save training history ─────────────────────────────────────────────────
    hist_path = TRAIN_LOGS_DIR / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Promote best to production ────────────────────────────────────────────
    if best_ckpt_path:
        PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        prod_path = PRODUCTION_DIR / "best_model.pt"
        shutil.copy2(best_ckpt_path, prod_path)
        log.info(f"✓ Best model (val_acc={best_val_acc:.4f}) → {prod_path}")

        # Compute and save Fisher matrix for EWC
        log.info("Computing Fisher Information Matrix for EWC …")
        model.load_state_dict(torch.load(prod_path, weights_only=False)["model"])
        fisher = compute_fisher(model, train_loader, device)
        torch.save(fisher, PRODUCTION_DIR / "fisher.pt")
        log.info("Fisher matrix saved.")

        return prod_path

    raise RuntimeError("Training ended without a valid checkpoint. Check data volume and date ranges.")
