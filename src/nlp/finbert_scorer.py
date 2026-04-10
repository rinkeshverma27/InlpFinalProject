"""
src/nlp/finbert_scorer.py — Batched FinBERT inference (English financial sentiment).

Returns full [pos, neg, neu] probability vector — never just argmax.
Supports int8 quantization for 4GB VRAM mode.
Includes OOM-safe batch runner and inference checkpointing.
"""

import pathlib
import numpy as np
import pandas as pd
import torch
from typing import List, Optional

from src.utils.logger import get_logger
from src.utils.errors import oom_safe_run, retry
from src.utils.paths import PRETRAINED_DIR

log = get_logger("finbert")

_LABELS = ["positive", "negative", "neutral"]


@retry(max_retries=3, backoff=10.0, exceptions=(OSError, ConnectionError, Exception))
def _load_finbert(model_id: str, use_int8: bool, device: torch.device):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    cache_dir = PRETRAINED_DIR / "finbert"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading FinBERT: {model_id} (int8={use_int8}) …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_id, cache_dir=cache_dir
    )

    if use_int8 and device.type == "cpu":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        log.info("FinBERT: int8 quantisation applied (CPU).")

    model.to(device).eval()
    return tokenizer, model


def _infer_batch(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run inference on a list of texts. Returns [N, 3] float32 array."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i + batch_size]
        enc   = tokenizer(
            chunk, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [B, 3]
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def score_english(
    texts: List[str],
    cfg: dict,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[pathlib.Path] = None,
) -> pd.DataFrame:
    """
    Run FinBERT on a list of English headlines.

    Args:
        texts         : List of English financial headlines.
        cfg           : Full config dict.
        device        : torch.device (auto-detected if None).
        checkpoint_dir: Save partial results here to survive crashes.

    Returns:
        pd.DataFrame with columns [pos, neg, neu] — shape [N, 3].
        Values are softmax probabilities summing to 1.
    """
    if not texts:
        return pd.DataFrame(columns=["pos", "neg", "neu"])

    nlp_cfg  = cfg.get("nlp", {})
    model_id = nlp_cfg.get("finbert_model_id", "ProsusAI/finbert")
    use_int8 = cfg.get("model", {}).get("use_int8_inference", False)
    batch_sz = nlp_cfg.get("finbert_batch_size", 32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = _load_finbert(model_id, use_int8, device)

    # Checkpoint: resume from partial results if they exist
    ckpt_path = None
    offset    = 0
    partial   = []
    if checkpoint_dir:
        ckpt_path = checkpoint_dir / "finbert_partial.npy"
        if ckpt_path.exists():
            partial = [np.load(ckpt_path)]
            offset  = partial[0].shape[0]
            log.info(f"[FINBERT] Resuming from checkpoint: {offset} rows already done.")

    remaining = texts[offset:]
    if remaining:
        probs = oom_safe_run(
            _infer_batch, remaining, tokenizer, model, device,
            batch_size=batch_sz, cfg=cfg,
        )
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            np.save(ckpt_path, np.concatenate(partial + [probs], axis=0))
        partial.append(probs)

    all_probs = np.concatenate(partial, axis=0) if partial else np.zeros((0, 3))

    # FinBERT label order: positive=0, negative=1, neutral=2
    # Re-map to our canonical [pos, neg, neu]
    id2label  = model.config.id2label if hasattr(model, "config") else {0: "positive", 1: "negative", 2: "neutral"}
    col_order = ["pos", "neg", "neu"]
    label_map = {}
    for idx, lbl in id2label.items():
        l = lbl.lower()
        if "pos" in l:   label_map[idx] = 0
        elif "neg" in l: label_map[idx] = 1
        else:            label_map[idx] = 2

    reordered = np.zeros_like(all_probs)
    for src, dst in label_map.items():
        if src < all_probs.shape[1]:
            reordered[:, dst] = all_probs[:, src]

    return pd.DataFrame(reordered, columns=col_order)
