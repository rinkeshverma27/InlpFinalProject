"""
src/adaptive/tier3_retrain.py — Monthly full retrain on rolling 3-year window.

Archives old production model before overwriting.
"""

import shutil
import pathlib
from datetime import datetime
from typing import List, Optional

from src.utils.logger import get_logger
from src.utils.paths import PRODUCTION_DIR, CHECKPOINTS_DIR

log = get_logger("tier3_retrain")


def run_tier3(fused_datasets: List[dict], cfg: dict, device=None):
    """
    Full retrain on a rolling window. Archives current production model first.

    Args:
        fused_datasets: All fused ticker data (from data_fusion.fuse_ticker).
        cfg           : Full config dict.
        device        : torch.device.
    """
    # Archive current production model
    if (PRODUCTION_DIR / "best_model.pt").exists():
        archive_dir = CHECKPOINTS_DIR / f"archive_{datetime.now().strftime('%Y%m%d_%H%M')}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for f in PRODUCTION_DIR.iterdir():
            shutil.copy2(f, archive_dir / f.name)
        log.info(f"[TIER-3] Current production model archived → {archive_dir}")

    # Override training dates for rolling window
    import yaml
    adapt_cfg = cfg.get("adaptive", {})
    window_yr = adapt_cfg.get("tier3_window_years", 3)

    from dateutil.relativedelta import relativedelta
    from datetime import date
    end_date   = date.today()
    start_date = end_date - relativedelta(years=window_yr)
    cfg["training"]["train_start"] = start_date.strftime("%Y-%m-%d")
    cfg["training"]["train_end"]   = (end_date - relativedelta(months=6)).strftime("%Y-%m-%d")
    cfg["training"]["val_start"]   = (end_date - relativedelta(months=6)).strftime("%Y-%m-%d")
    cfg["training"]["val_end"]     = (end_date - relativedelta(months=3)).strftime("%Y-%m-%d")

    log.info(
        f"[TIER-3] Starting monthly retrain. "
        f"Window: {cfg['training']['train_start']} → {cfg['training']['train_end']}"
    )

    from src.training.trainer import train
    best_ckpt = train(fused_datasets, cfg, device=device)
    log.info(f"[TIER-3] Monthly retrain complete. Best model: {best_ckpt}")
