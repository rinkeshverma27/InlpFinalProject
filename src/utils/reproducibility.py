"""
src/utils/reproducibility.py — Helpers for reproducible experiment runs.
"""

import os
import random
from typing import Optional

import numpy as np
import torch

from src.utils.logger import get_logger

log = get_logger("repro")


def get_seed(cfg: dict, override: Optional[int] = None) -> int:
    """Resolve the experiment seed from config, with optional CLI override."""
    if override is not None:
        return int(override)
    return int(cfg.get("reproducibility", {}).get("seed", 42))


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed Python, NumPy, and PyTorch.
    When deterministic=True, prefer stable kernels over peak performance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    log.info(
        f"[REPRO] seed={seed} deterministic={deterministic} "
        f"cuda={'yes' if torch.cuda.is_available() else 'no'}"
    )


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a seeded torch.Generator for DataLoader shuffling."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
