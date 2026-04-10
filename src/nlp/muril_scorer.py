"""
src/nlp/muril_scorer.py — Batched MuRIL inference for Hindi / Hinglish sentiment.

Uses the fine-tuned checkpoint from models/finetuned/muril_financial/.
Falls back to base MuRIL if fine-tuned checkpoint is missing (with a clear warning).
"""

import pathlib
import numpy as np
import pandas as pd
import torch
from typing import List, Optional

from src.utils.logger import get_logger
from src.utils.errors import oom_safe_run, retry
from src.utils.paths import PRETRAINED_DIR, MURIL_FT_DIR

log = get_logger("muril_scorer")


@retry(max_retries=3, backoff=10.0, exceptions=(OSError, ConnectionError, Exception))
def _load_muril(use_finetuned: bool, model_id: str, use_int8: bool, device: torch.device):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    cache_dir = PRETRAINED_DIR / "muril"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if use_finetuned and MURIL_FT_DIR.exists() and any(MURIL_FT_DIR.iterdir()):
        load_path = str(MURIL_FT_DIR)
        log.info(f"Loading fine-tuned MuRIL from {load_path} …")
    else:
        load_path = model_id
        log.warning(
            f"Fine-tuned MuRIL not found at {MURIL_FT_DIR}.\n"
            f"Falling back to base {model_id}.\n"
            f"Run: python main.py finetune-muril  to create the fine-tuned checkpoint."
        )

    tokenizer = AutoTokenizer.from_pretrained(load_path, cache_dir=cache_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(
        load_path, cache_dir=cache_dir, num_labels=3
    )

    if use_int8 and device.type == "cpu":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    model.to(device).eval()
    return tokenizer, model


def _infer_batch(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    all_probs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i + batch_size]
        enc   = tokenizer(
            chunk, padding=True, truncation=True,
            max_length=128, return_tensors="pt"   # 128 is enough for Hindi headlines
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def score_hindi(
    texts: List[str],
    cfg: dict,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[pathlib.Path] = None,
) -> pd.DataFrame:
    """
    Run MuRIL on Hindi / Hinglish headlines.

    Returns:
        pd.DataFrame with columns [pos, neg, neu], shape [N, 3].
    """
    if not texts:
        return pd.DataFrame(columns=["pos", "neg", "neu"])

    nlp_cfg  = cfg.get("nlp", {})
    model_id = nlp_cfg.get("muril_model_id", "google/muril-base-cased")
    use_int8 = cfg.get("model", {}).get("use_int8_inference", False)
    batch_sz = nlp_cfg.get("muril_batch_size", 32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = _load_muril(True, model_id, use_int8, device)

    # Checkpoint resume
    offset, partial = 0, []
    if checkpoint_dir:
        ckpt = checkpoint_dir / "muril_partial.npy"
        if ckpt.exists():
            partial = [np.load(ckpt)]
            offset  = partial[0].shape[0]
            log.info(f"[MuRIL] Resuming from checkpoint: {offset} rows done.")

    remaining = texts[offset:]
    if remaining:
        probs = oom_safe_run(
            _infer_batch, remaining, tokenizer, model, device,
            batch_size=batch_sz, cfg=cfg,
        )
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            np.save(ckpt, np.concatenate(partial + [probs], axis=0))
        partial.append(probs)

    all_probs = np.concatenate(partial, axis=0) if partial else np.zeros((0, 3))
    return pd.DataFrame(all_probs, columns=["pos", "neg", "neu"])
