"""
src/utils/errors.py — Graceful failure wrappers for every pipeline stage.

Design goals:
  1. Never crash silently — always print a detailed, actionable error.
  2. CUDA OOM → automatically shrink batch size and retry.
  3. Transient errors (network, disk) → exponential back-off retry.
  4. CPU RAM spikes → warn with current RSS before crashing.
  5. Every public decorator / context manager is importable from here.
"""

import functools
import os
import sys
import time
import traceback
from typing import Callable, Any

from src.utils.logger import get_logger

log = get_logger("errors")


# ─────────────────────────────────────────────────────────────────────────────
# Memory monitoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cpu_ram_gb() -> float:
    """Current process RSS in GB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except ImportError:
        return 0.0


def _gpu_vram_gb() -> float:
    """Currently allocated GPU VRAM in GB (0 if no GPU / torch not loaded)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0


def check_memory(cfg: dict):
    """Warn if RAM/VRAM thresholds are approached."""
    res   = cfg.get("resilience", {})
    cpu_w = res.get("cpu_ram_warn_gb", 6.0)
    gpu_w = res.get("gpu_vram_warn_gb", 3.5)

    cpu_used = _cpu_ram_gb()
    gpu_used = _gpu_vram_gb()

    if cpu_used > cpu_w:
        log.warning(
            f"⚠  CPU RAM usage {cpu_used:.2f} GB exceeds warning threshold "
            f"{cpu_w:.1f} GB. Consider reducing batch_size or num_workers."
        )
    if gpu_used > gpu_w:
        log.warning(
            f"⚠  GPU VRAM usage {gpu_used:.2f} GB exceeds warning threshold "
            f"{gpu_w:.1f} GB. Consider switching to vram_profile: '4gb' in config.yaml."
        )


# ─────────────────────────────────────────────────────────────────────────────
# OOM-safe batch runner
# ─────────────────────────────────────────────────────────────────────────────

def oom_safe_run(fn: Callable, *args, batch_size: int, cfg: dict, **kwargs) -> Any:
    """
    Call fn(*args, batch_size=batch_size, **kwargs).
    On CUDA OOM, shrink batch_size by shrink_factor and retry up to max_retries.
    On continued failure, raise with a clear diagnostic message.
    """
    res          = cfg.get("resilience", {})
    max_retries  = res.get("max_retries", 3)
    shrink       = res.get("oom_batch_shrink_factor", 0.5)
    min_bs       = res.get("oom_min_batch_size", 1)
    current_bs   = batch_size

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, batch_size=current_bs, **kwargs)
        except RuntimeError as e:
            err_str = str(e)
            is_oom  = "out of memory" in err_str.lower() or "CUDA out of memory" in err_str
            if not is_oom or attempt >= max_retries:
                raise
            new_bs = max(min_bs, int(current_bs * shrink))
            log.warning(
                f"[OOM] CUDA out-of-memory on attempt {attempt + 1}/{max_retries}.\n"
                f"       Current batch_size={current_bs} → retrying with {new_bs}.\n"
                f"       GPU VRAM in use: {_gpu_vram_gb():.2f} GB\n"
                f"       Tip: set vram_profile: '4gb' in config.yaml for permanent fix."
            )
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            current_bs = new_bs
            time.sleep(1)

    raise RuntimeError(
        f"[OOM] Function '{fn.__name__}' failed after {max_retries} retries.\n"
        f"GPU VRAM: {_gpu_vram_gb():.2f} GB | CPU RAM: {_cpu_ram_gb():.2f} GB\n"
        f"Action: switch vram_profile to '4gb' in config.yaml, or reduce "
        f"nlp.finbert_batch_size to 4–8."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Retry decorator for transient failures (network, I/O)
# ─────────────────────────────────────────────────────────────────────────────

def retry(max_retries: int = 3, backoff: float = 5.0, exceptions=(Exception,)):
    """
    Decorator: retry the function up to max_retries times on transient errors.
    Prints a detailed error + back-off message on each failure.
    """
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_retries:
                        log.error(
                            f"[RETRY EXHAUSTED] '{fn.__name__}' failed after "
                            f"{max_retries} retries.\n"
                            f"Last error ({type(e).__name__}): {e}\n"
                            f"CPU RAM: {_cpu_ram_gb():.2f} GB | "
                            f"GPU VRAM: {_gpu_vram_gb():.2f} GB\n"
                            f"Full traceback:\n{traceback.format_exc()}"
                        )
                        raise
                    wait = backoff * (2 ** attempt)
                    log.warning(
                        f"[RETRY {attempt + 1}/{max_retries}] '{fn.__name__}' raised "
                        f"{type(e).__name__}: {e}\n"
                        f"Retrying in {wait:.1f}s …"
                    )
                    time.sleep(wait)
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Stage-level graceful failure wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(stage_name: str, fn: Callable, *args, **kwargs) -> Any:
    """
    Run a pipeline stage function with full error catching and diagnostics.
    On failure: logs detailed error + memory snapshot and re-raises.
    Does NOT swallow errors — the pipeline will halt cleanly.
    """
    log.info(f"▶  Starting stage: {stage_name}")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        log.info(
            f"✓  Stage '{stage_name}' completed in {elapsed:.1f}s | "
            f"CPU RAM: {_cpu_ram_gb():.2f} GB | GPU VRAM: {_gpu_vram_gb():.2f} GB"
        )
        return result
    except KeyboardInterrupt:
        log.warning(f"⚡ Stage '{stage_name}' interrupted by user (Ctrl+C). "
                    f"Partial outputs (if any) are in their output directory.")
        sys.exit(0)
    except MemoryError:
        log.critical(
            f"[FATAL] Stage '{stage_name}' hit a CPU MemoryError.\n"
            f"CPU RAM in use: {_cpu_ram_gb():.2f} GB\n"
            f"Action: reduce training.batch_size in config.yaml, or close other "
            f"applications to free RAM.\n{traceback.format_exc()}"
        )
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log.critical(
                f"[FATAL OOM] Stage '{stage_name}' crashed with CUDA OOM.\n"
                f"GPU VRAM allocated: {_gpu_vram_gb():.2f} GB\n"
                f"Action: set vram_profile: '4gb' in config.yaml OR reduce "
                f"nlp.finbert_batch_size to 4.\n{traceback.format_exc()}"
            )
        else:
            log.error(
                f"[ERROR] Stage '{stage_name}' raised RuntimeError.\n"
                f"CPU RAM: {_cpu_ram_gb():.2f} GB | GPU VRAM: {_gpu_vram_gb():.2f} GB\n"
                f"{traceback.format_exc()}"
            )
        raise
    except Exception:
        log.error(
            f"[ERROR] Stage '{stage_name}' failed unexpectedly.\n"
            f"CPU RAM: {_cpu_ram_gb():.2f} GB | GPU VRAM: {_gpu_vram_gb():.2f} GB\n"
            f"{traceback.format_exc()}"
        )
        raise
