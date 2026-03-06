"""
training/trainer.py
Core training loop, EWC weekly nudge, and evaluation functions.
"""

import pathlib
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.tlstm import TLSTMModel, compute_fisher_matrix


# ──────────────────────────────────────────────────────────────────────────────
# EWC Penalty
# ──────────────────────────────────────────────────────────────────────────────

def ewc_penalty(
    model: TLSTMModel,
    fisher: dict[str, torch.Tensor],
    old_params: dict[str, torch.Tensor],
    ewc_lambda: float = 400.0,
) -> torch.Tensor:
    """
    Elastic Weight Consolidation penalty.
    Penalises deviations from `old_params` by the Fisher diagonal importance.
    Only applied to output_head parameters (Tier-2 nudge scope).
    """
    loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if name in fisher and name in old_params:
            fisher_diag = fisher[name].to(param.device)
            old_p       = old_params[name].to(param.device)
            loss = loss + (fisher_diag * (param - old_p) ** 2).sum()
    return (ewc_lambda / 2) * loss


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages full and EWC (nudge) training of TLSTMModel.

    Parameters
    ----------
    model        : TLSTMModel
    device       : torch.device
    lr           : base learning rate (full training)
    ewc_lr       : learning rate for Tier-2 weekly nudge (default 1e-5)
    ewc_lambda   : EWC regularisation strength (default 400)
    checkpoint_dir : where to save model checkpoints
    """

    def __init__(
        self,
        model: TLSTMModel,
        device: torch.device,
        lr: float = 1e-3,
        ewc_lr: float = 1e-5,
        ewc_lambda: float = 400.0,
        checkpoint_dir: pathlib.Path = pathlib.Path("models/checkpoints/stage2A"),
    ):
        self.model          = model.to(device)
        self.device         = device
        self.lr             = lr
        self.ewc_lr         = ewc_lr
        self.ewc_lambda     = ewc_lambda
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = nn.HuberLoss(delta=0.01)

        # EWC state — populated after full retrain
        self.fisher: dict[str, torch.Tensor] = {}
        self.old_params: dict[str, torch.Tensor] = {}

    # ── Full Training ────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for ohlcv_seq, nlp_vec, label in loader:
            ohlcv_seq = ohlcv_seq.to(self.device)
            nlp_vec   = nlp_vec.to(self.device)
            label     = label.to(self.device)

            optimizer.zero_grad()
            pred = self.model(ohlcv_seq, nlp_vec)
            loss = self.loss_fn(pred, label)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * ohlcv_seq.size(0)

        return total_loss / len(loader.dataset)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 5,
        tag: str = "model",
    ) -> float:
        """
        Full training run with early stopping.
        Returns the best validation MAE achieved.
        """
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

        best_val_mae = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_mae    = self.evaluate(val_loader)["mae"]
            scheduler.step(val_mae)

            print(f"Epoch {epoch:3d} | train_loss={train_loss:.5f} | val_mae={val_mae:.5f}")

            if val_mae < best_val_mae:
                best_val_mae     = val_mae
                patience_counter = 0
                best_state       = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.save_checkpoint(tag, epoch, val_mae)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_val_mae

    # ── Tier-2 EWC Weekly Nudge ───────────────────────────────────────────────

    def ewc_nudge(
        self,
        nudge_loader: DataLoader,
        epochs: int = 3,
        patience: int = 2,
        tag: str = "ewc_nudge",
    ) -> bool:
        """
        Fine-tunes ONLY the last 2 FC layers using EWC regularisation.
        Backbone (LSTM + NLP projection) is frozen.
        Rolls back if validation MAE worsens (>2% tolerance).

        Returns True if nudge was accepted, False if rolled back.
        """
        if not self.fisher:
            raise RuntimeError("Fisher matrix not computed. Call compute_and_store_fisher() first.")

        # Snapshot current weights for rollback
        pre_nudge_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        pre_nudge_mae   = self.evaluate(nudge_loader)["mae"]

        # Snapshot old_params for EWC penalty
        self.old_params = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if "output_head" in name
        }

        # Freeze everything except output_head
        for name, p in self.model.named_parameters():
            p.requires_grad = "output_head" in name

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.ewc_lr,
            weight_decay=1e-4,
        )

        best_mae = pre_nudge_mae
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            for ohlcv_seq, nlp_vec, label in nudge_loader:
                ohlcv_seq = ohlcv_seq.to(self.device)
                nlp_vec   = nlp_vec.to(self.device)
                label     = label.to(self.device)

                optimizer.zero_grad()
                pred     = self.model(ohlcv_seq, nlp_vec)
                task_loss= self.loss_fn(pred, label)
                penalty  = ewc_penalty(self.model, self.fisher, self.old_params, self.ewc_lambda)
                loss     = task_loss + penalty
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            val_mae = self.evaluate(nudge_loader)["mae"]
            print(f"  EWC nudge epoch {epoch} | val_mae={val_mae:.5f}")
            if val_mae < best_mae:
                best_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Unfreeze all parameters
        for p in self.model.parameters():
            p.requires_grad = True

        # Rollback if MAE worsened by more than 2% tolerance
        tolerance = 1.02
        if best_mae > pre_nudge_mae * tolerance:
            print(f"  EWC nudge ROLLED BACK: {best_mae:.5f} > {pre_nudge_mae:.5f} * {tolerance}")
            self.model.load_state_dict(pre_nudge_state)
            return False

        self.save_checkpoint(tag)
        return True

    def compute_and_store_fisher(self, dataloader: DataLoader, n_samples: int = 1000):
        """Compute Fisher matrix after a full retrain. Call before any EWC nudges."""
        self.fisher = compute_fisher_matrix(self.model, dataloader, self.device, n_samples)
        print(f"  Fisher matrix computed for {len(self.fisher)} parameter groups.")

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """
        Compute MAE and Direction Accuracy on a data loader.
        Uses deterministic forward pass (no MC dropout).
        """
        self.model.eval()
        preds_all, labels_all = [], []

        for ohlcv_seq, nlp_vec, label in loader:
            ohlcv_seq = ohlcv_seq.to(self.device)
            nlp_vec   = nlp_vec.to(self.device)
            pred      = self.model(ohlcv_seq, nlp_vec).cpu()
            preds_all.append(pred)
            labels_all.append(label)

        preds_all  = torch.cat(preds_all).numpy().flatten()
        labels_all = torch.cat(labels_all).numpy().flatten()

        mae      = float(np.mean(np.abs(preds_all - labels_all)))
        dir_acc  = float(np.mean(np.sign(preds_all) == np.sign(labels_all)))

        return {"mae": mae, "direction_accuracy": dir_acc}

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str, epoch: int = 0, val_mae: float = 0.0):
        # Format: models/checkpoints/{stage}/model_{YYYY-MM-DD}.pt
        import pandas as pd
        date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        path = self.checkpoint_dir / f"model_{date_str}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "fisher": self.fisher,
            "epoch": epoch,
            "val_mae": val_mae,
        }, path)
        print(f"  Checkpoint saved → {path}")
        return path

    def load_checkpoint(self, path: pathlib.Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.fisher = ckpt.get("fisher", {})
        print(f"  Loaded checkpoint from {path} | val_mae={ckpt.get('val_mae', '?')}")
