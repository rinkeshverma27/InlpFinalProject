"""
src/utils/cache.py — Stage-level cache checks with TTL and graceful OOM guard.
"""

import pathlib
import time
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional

from src.utils.logger import get_logger

log = get_logger("cache")

_META_SUFFIX = ".cache_meta.json"


def _meta_path(output_path: pathlib.Path) -> pathlib.Path:
    return output_path.parent / (output_path.name + _META_SUFFIX)


def stage_exists(
    output_path: pathlib.Path,
    ttl_days: int,
    force: bool = False,
    cfg: Optional[dict] = None,
) -> bool:
    """
    Return True if `output_path` exists, is younger than ttl_days, and force=False.

    Args:
        output_path : Path to the stage's primary output file or directory.
        ttl_days    : Max age in days before the cache is considered stale.
        force       : If True, always return False (recompute).
        cfg         : Full config dict — checks cache.force_rerun global override.

    Returns:
        True  → skip this stage (cache is valid)
        False → run this stage
    """
    # Global override via config
    if cfg and cfg.get("cache", {}).get("force_rerun", False):
        log.debug(f"[CACHE BYPASS] force_rerun=true in config: {output_path.name}")
        return False

    if force:
        log.debug(f"[CACHE BYPASS] --force flag: {output_path.name}")
        return False

    if not output_path.exists():
        log.debug(f"[CACHE MISS] Not found: {output_path}")
        return False

    # Check age
    mtime     = output_path.stat().st_mtime
    age_days  = (time.time() - mtime) / 86400
    if age_days > ttl_days:
        log.info(
            f"[CACHE STALE] {output_path.name} is {age_days:.1f} days old "
            f"(TTL={ttl_days}d). Will recompute."
        )
        return False

    log.info(f"[CACHE HIT] {output_path.name} ({age_days:.1f}d old) — skipping stage.")
    return True


def mark_stage_done(output_path: pathlib.Path, meta: Optional[dict] = None):
    """Write a sidecar .cache_meta.json with timestamp and optional metadata."""
    record = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output_path),
        **(meta or {}),
    }
    try:
        _meta_path(output_path).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log.warning(f"[CACHE] Could not write meta for {output_path.name}: {e}")


def read_stage_meta(output_path: pathlib.Path) -> dict:
    """Read sidecar metadata for a cached stage (empty dict if missing)."""
    mp = _meta_path(output_path)
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def content_hash(path: pathlib.Path, chunk_size: int = 65536) -> str:
    """MD5 of a file's contents — useful for detecting silent corruption."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()
