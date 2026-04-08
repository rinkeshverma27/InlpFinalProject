"""
src/nlp/muril_finetune.py — Knowledge distillation: FinBERT (teacher) → MuRIL (student).

Strategy:
  1. Run FinBERT on Hindi/Hinglish examples from data/datasets/hindi_hinglish_financial.csv
     to generate soft probability labels (temperature-scaled).
  2. Fine-tune MuRIL student with combined loss:
       L = alpha * KL(soft_targets || student_probs) + (1-alpha) * CE(hard_label, student_logits)
  3. Freeze bottom `muril_freeze_layers` encoder layers to preserve multilingual representations.
  4. Save to models/finetuned/muril_financial/

Cache: skips entirely if checkpoint exists (unless --force).
"""

import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from src.utils.logger import get_logger
from src.utils.errors import run_stage, oom_safe_run, retry
from src.utils.cache import stage_exists, mark_stage_done
from src.utils.paths import DATASETS_DIR, MURIL_FT_DIR, PRETRAINED_DIR, CHECKPOINTS_DIR

log = get_logger("muril_finetune")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HindiDistillDataset(Dataset):
    def __init__(self, texts, hard_labels, soft_labels, tokenizer, max_len=128):
        self.texts       = texts
        self.hard_labels = hard_labels           # int [0,1,2]
        self.soft_labels = soft_labels           # float32 [N, 3]
        self.tokenizer   = tokenizer
        self.max_len     = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "hard_label":     torch.tensor(self.hard_labels[idx], dtype=torch.long),
            "soft_label":     torch.tensor(self.soft_labels[idx], dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Teacher inference (FinBERT on Hindi/Hinglish text)
# ─────────────────────────────────────────────────────────────────────────────

def _get_soft_labels(texts, cfg, device, temperature=4.0) -> np.ndarray:
    """Use FinBERT to generate soft probability labels for Hindi/Hinglish text."""
    from src.nlp.finbert_scorer import score_english
    log.info(f"Generating soft labels with FinBERT teacher (T={temperature}) …")
    scores = score_english(texts, cfg, device=device)
    probs  = scores[["pos", "neg", "neu"]].values.astype(np.float32)

    # Temperature scaling: sharpen/soften teacher distribution
    logits    = np.log(probs + 1e-9) / temperature
    soft_labs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return soft_labs


# ─────────────────────────────────────────────────────────────────────────────
# Main fine-tuning function
# ─────────────────────────────────────────────────────────────────────────────

def finetune_muril(cfg: dict, force: bool = False, device: Optional[torch.device] = None):
    """
    Fine-tune MuRIL via knowledge distillation from FinBERT.
    Saves checkpoint to models/finetuned/muril_financial/.
    """
    done_flag = MURIL_FT_DIR / "config.json"   # HF saves this at the end
    ttl       = 365   # model checkpoint valid for a year unless forced

    if stage_exists(done_flag, ttl, force, cfg):
        log.info("[CACHE HIT] Fine-tuned MuRIL exists — skipping.")
        return

    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    nlp_cfg    = cfg.get("nlp", {})
    train_cfg  = cfg.get("training", {})
    model_id   = nlp_cfg.get("muril_model_id", "google/muril-base-cased")
    freeze_n   = nlp_cfg.get("muril_freeze_layers", 8)
    temp       = nlp_cfg.get("distill_temperature", 4.0)
    alpha      = nlp_cfg.get("distill_alpha", 0.7)
    max_samp   = nlp_cfg.get("distill_max_samples", 5000)
    lr         = train_cfg.get("muril_lr", 2e-5)
    epochs     = train_cfg.get("muril_epochs", 5)
    batch_sz   = train_cfg.get("muril_batch_size", 16)
    warmup_r   = train_cfg.get("muril_warmup_ratio", 0.1)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load dataset ──────────────────────────────────────────────────────────
    hi_csv = DATASETS_DIR / "hindi_hinglish_financial.csv"
    if not hi_csv.exists():
        raise FileNotFoundError(
            f"Hindi/Hinglish dataset not found: {hi_csv}\n"
            f"Run: python3 data/datasets/generate_datasets.py"
        )

    df = pd.read_csv(hi_csv)
    # Merge label values: positive→0, neutral→1, negative→2
    label_map = {"positive": 0, "neutral": 1, "negative": 2}
    if "label_name" in df.columns:
        df["hard_int"] = df["label_name"].map(label_map)
    else:
        # Fall back to numeric label column (1→0, 0→1, -1→2)
        df["hard_int"] = df["label"].map({1: 0, 0: 1, -1: 2})

    df = df.dropna(subset=["hard_int", "text"])
    df["hard_int"] = df["hard_int"].astype(int)

    if len(df) > max_samp:
        df = df.sample(max_samp, random_state=42)

    texts       = df["text"].tolist()
    hard_labels = df["hard_int"].tolist()

    # ── Generate soft labels from FinBERT teacher ─────────────────────────────
    soft_labels = _get_soft_labels(texts, cfg, device, temperature=temp)

    # ── Load MuRIL student ────────────────────────────────────────────────────
    cache_dir = PRETRAINED_DIR / "muril"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading MuRIL student: {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_id, cache_dir=cache_dir, num_labels=3,
        ignore_mismatched_sizes=True,
    )

    # ── Freeze bottom N encoder layers ────────────────────────────────────────
    for name, param in model.named_parameters():
        if "encoder.layer" in name:
            layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_idx < freeze_n:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"MuRIL trainable parameters: {trainable:,}  (frozen first {freeze_n} layers)")

    model.to(device)

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataset    = HindiDistillDataset(texts, hard_labels, soft_labels, tokenizer)
    loader     = DataLoader(dataset, batch_size=batch_sz, shuffle=True, num_workers=2)

    optimizer  = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    total_steps = len(loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * warmup_r), total_steps
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    CKPT_DIR = CHECKPOINTS_DIR / "muril_finetune"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_loss  = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            hard_label = batch["hard_label"].to(device)
            soft_label = batch["soft_label"].to(device)

            logits     = model(input_ids=input_ids, attention_mask=attn_mask).logits
            log_probs  = F.log_softmax(logits / temp, dim=-1)

            # Distillation loss (KL divergence with soft targets)
            kl_loss    = F.kl_div(log_probs, soft_label, reduction="batchmean")
            # Hard label cross-entropy
            ce_loss    = F.cross_entropy(logits, hard_label)

            loss = alpha * kl_loss + (1 - alpha) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        log.info(f"  Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f}")

        # Save epoch checkpoint
        ckpt_path = CKPT_DIR / f"epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = ckpt_path

    # ── Save final model ──────────────────────────────────────────────────────
    MURIL_FT_DIR.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    model.save_pretrained(MURIL_FT_DIR)
    tokenizer.save_pretrained(MURIL_FT_DIR)
    mark_stage_done(done_flag, {"best_loss": best_loss, "epochs": epochs})
    log.info(f"✓ MuRIL fine-tuning complete. Saved to {MURIL_FT_DIR}")
