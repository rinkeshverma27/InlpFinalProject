#!/usr/bin/env python3
"""
main.py — Central CLI orchestrator for the Stock Direction Predictor pipeline.

Usage:
  python main.py <command> [options]

Individual stage commands (run in order):
  generate-data      Generate training datasets (Hindi/Hinglish + English)
  finetune-muril     Fine-tune MuRIL via FinBERT distillation
  sentiment          Compute sentiment scores for all tickers (Stream A)
  features           Compute price features for all tickers (Stream B)
  fuse               Fuse sentiment + price into model-ready tensors
  train              Train the dual-stream LSTM
  eval               Evaluate on val or test split (full performance matrix)
  predict            Run inference on new data for a specific ticker/date

Adaptive learning commands (run after deployment):
  tier1              Daily source trust weight update
  tier2              Weekly EWC gradient nudge
  tier3              Monthly full retrain

Full pipeline shortcut:
  run-all            Runs: sentiment → features → fuse → train → eval (in sequence)
                     Skips any stage whose output is already cached.

Global options:
  --config PATH      Path to config.yaml (default: config.yaml)
  --force            Bypass all caches — recompute everything
  --device cpu|cuda  Override device (default: auto-detect)
  --ticker TICKER    Restrict to a single ticker (where applicable)
  --profile 4gb|8gb|full  Override vram_profile from config
"""

import argparse
import sys
import pathlib
import torch

# ── Bootstrap sys.path so src/ imports work ──────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import ensure_dirs
from src.utils.errors import run_stage

log = get_logger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _device(args) -> torch.device:
    if hasattr(args, "device") and args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_tickers(cfg: dict, args) -> list:
    if hasattr(args, "ticker") and args.ticker:
        return [args.ticker.upper()]
    t = cfg.get("tickers", {})
    active = t.get("active", "subset_5")
    tlist  = t.get(active)
    if tlist is None:
        # Load all from OHLCV dir
        from src.utils.paths import OHLCV_DIR
        tlist = [p.stem.upper() for p in OHLCV_DIR.glob("*.parquet")]
        tlist += [p.stem.upper() for p in OHLCV_DIR.glob("*.csv")]
    return sorted(set(tlist))


# ─────────────────────────────────────────────────────────────────────────────
# Stage handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_generate_data(args, cfg):
    """Generate Hindi/Hinglish + English training datasets."""
    gen_script = ROOT / "data" / "datasets" / "generate_datasets.py"
    if not gen_script.exists():
        log.error(f"Generator script not found: {gen_script}")
        sys.exit(1)
    import subprocess
    extra = ["--force"] if args.force else []
    subprocess.run([sys.executable, str(gen_script)] + extra, check=True)


def cmd_finetune_muril(args, cfg):
    """Fine-tune MuRIL via FinBERT knowledge distillation."""
    from src.nlp.muril_finetune import finetune_muril
    run_stage("finetune-muril", finetune_muril, cfg, args.force, _device(args))


def cmd_sentiment(args, cfg):
    """Compute sentiment for all (or one) ticker(s)."""
    from src.data.news_loader import load_news
    from src.nlp.sentiment_aggregator import aggregate_ticker
    from src.utils.paths import RAW_DATA_DIR

    tickers = _get_tickers(cfg, args)
    device  = _device(args)
    log.info(f"Sentiment stage — tickers: {tickers}")

    for ticker in tickers:
        news_df = run_stage(f"load_news:{ticker}", load_news,
                            ticker=ticker, news_dir=RAW_DATA_DIR)
        run_stage(f"sentiment:{ticker}", aggregate_ticker,
                  ticker, news_df, cfg, args.force, device)


def cmd_features(args, cfg):
    """Compute OHLCV price features for all (or one) ticker(s)."""
    from src.data.ohlcv_loader import load_ohlcv
    from src.utils.paths import FEATURES_DIR

    tickers = _get_tickers(cfg, args)
    log.info(f"Features stage — tickers: {tickers}")

    for ticker in tickers:
        run_stage(f"features:{ticker}", load_ohlcv, ticker, cfg)


def cmd_fuse(args, cfg):
    """Fuse sentiment + price into model-ready tensors."""
    from src.data.data_fusion import fuse_ticker
    tickers = _get_tickers(cfg, args)
    log.info(f"Fuse stage — tickers: {tickers}")
    results = []
    for ticker in tickers:
        fd = run_stage(f"fuse:{ticker}", fuse_ticker, ticker, cfg, args.force)
        if fd is not None:
            results.append(fd)
    log.info(f"Fuse complete: {len(results)}/{len(tickers)} tickers succeeded.")
    return results


def cmd_train(args, cfg):
    """Train the dual-stream LSTM."""
    from src.data.data_fusion import fuse_ticker
    from src.training.trainer import train

    tickers = _get_tickers(cfg, args)
    fused   = []
    for ticker in tickers:
        fd = fuse_ticker(ticker, cfg, force=False)
        if fd is not None:
            fused.append(fd)

    if not fused:
        log.error("No fused data available. Run `python main.py fuse` first.")
        sys.exit(1)

    resume = pathlib.Path(args.resume) if hasattr(args, "resume") and args.resume else None
    run_stage("train", train, fused, cfg, device=_device(args), resume_checkpoint=resume)


def cmd_eval(args, cfg):
    """Full performance matrix on val or test split."""
    from src.data.data_fusion import fuse_ticker
    from src.model.dual_stream_lstm import build_model
    from src.model.mc_dropout import mc_predict
    from src.training.dataset import StockSequenceDataset, make_splits, collate_fn
    from src.training.evaluate import evaluate
    from src.utils.paths import PRODUCTION_DIR
    from torch.utils.data import DataLoader

    split    = getattr(args, "split", "test")
    tickers  = _get_tickers(cfg, args)
    device   = _device(args)
    m_cfg    = cfg.get("model", {})
    n_passes = m_cfg.get("mc_dropout_passes", 30)
    thresh   = m_cfg.get("confidence_threshold", 0.65)

    model_path = PRODUCTION_DIR / "best_model.pt"
    if not model_path.exists():
        log.error(f"No production model found at {model_path}. Run `python main.py train` first.")
        sys.exit(1)

    model = build_model(cfg).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    rows = []
    for ticker in tickers:
        fd = fuse_ticker(ticker, cfg, force=False)
        if fd is None:
            continue
        tr_idx, va_idx, te_idx = make_splits(fd, cfg, ticker)
        idx_map = {"train": tr_idx, "val": va_idx, "test": te_idx}

        for sname, sidx in idx_map.items():
            if len(sidx) == 0:
                continue
            ds     = StockSequenceDataset(fd, sidx)
            loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

            for price, sent, labels, dates in loader:
                price, sent = price.to(device), sent.to(device)
                mean, var, abstain = mc_predict(model, price, sent, n_passes, thresh)
                for i in range(len(labels)):
                    direction = "ABSTAIN" if abstain[i].item() else ("UP" if mean[i].item() >= 0.5 else "DOWN")
                    rows.append({
                        "date":      dates[i],
                        "ticker":    ticker,
                        "label":     int(labels[i].item()),
                        "prob":      round(mean[i].item(), 4),
                        "variance":  round(var[i].item(), 4),
                        "direction": direction,
                        "split":     sname,
                    })

    results_df = __import__("pandas").DataFrame(rows)
    run_stage("evaluate", evaluate, results_df, cfg, split_name=split)


def cmd_predict(args, cfg):
    """Predict direction for a specific ticker (latest data)."""
    from src.data.data_fusion import fuse_ticker
    from src.model.dual_stream_lstm import build_model
    from src.model.mc_dropout import predict_single
    from src.utils.paths import PRODUCTION_DIR

    ticker = args.ticker.upper() if hasattr(args, "ticker") and args.ticker else None
    if not ticker:
        log.error("Provide --ticker TICKER for predict command.")
        sys.exit(1)

    device     = _device(args)
    model_path = PRODUCTION_DIR / "best_model.pt"
    if not model_path.exists():
        log.error("No production model. Run `python main.py train` first.")
        sys.exit(1)

    model = build_model(cfg).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    fd = fuse_ticker(ticker, cfg, force=False)
    if fd is None:
        log.error(f"No fused data for {ticker}.")
        sys.exit(1)

    # Use the last available sequence
    price_seq = fd["price_seq"][-1]
    sent_seq  = fd["sentiment_seq"][-1]
    date_str  = fd["dates"][-1]

    result = predict_single(model, price_seq, sent_seq, cfg, device)
    print(f"\n{'='*50}")
    print(f"  Prediction for {ticker} — trade date: {date_str}")
    print(f"  Direction   : {result['direction']}")
    print(f"  P(UP)       : {result['probability']:.4f}")
    print(f"  Confidence  : {result['confidence']:.4f}  (variance={result['variance']:.4f})")
    print(f"{'='*50}\n")


def cmd_tier1(args, cfg):
    from src.adaptive.tier1_trust import run_tier1
    date_arg = getattr(args, "date", None)
    run_stage("tier1", run_tier1, date_arg)


def cmd_tier2(args, cfg):
    from src.data.data_fusion import fuse_ticker
    from src.training.dataset import StockSequenceDataset, make_splits, collate_fn
    from src.adaptive.tier2_ewc_nudge import run_tier2
    from torch.utils.data import DataLoader, ConcatDataset

    tickers = _get_tickers(cfg, args)
    recent_ds = []
    for ticker in tickers:
        fd = fuse_ticker(ticker, cfg, force=False)
        if fd is None: continue
        idx = list(range(max(0, len(fd["labels"]) - cfg["adaptive"].get("tier2_nudge_days", 5) * 5), len(fd["labels"])))
        if idx:
            recent_ds.append(StockSequenceDataset(fd, __import__("numpy").array(idx)))

    if not recent_ds:
        log.warning("[TIER-2] No recent data available.")
        return

    loader = DataLoader(ConcatDataset(recent_ds), batch_size=32, shuffle=True, collate_fn=collate_fn)
    run_stage("tier2", run_tier2, loader, cfg, _device(args))


def cmd_tier3(args, cfg):
    from src.data.data_fusion import fuse_ticker
    from src.adaptive.tier3_retrain import run_tier3

    tickers = _get_tickers(cfg, args)
    fused   = [fd for t in tickers if (fd := fuse_ticker(t, cfg, force=False)) is not None]
    run_stage("tier3", run_tier3, fused, cfg, _device(args))


def cmd_run_all(args, cfg):
    """
    Full pipeline: sentiment → features → fuse → train → eval.
    Cache-aware: each stage is skipped if already done.
    """
    log.info("=" * 60)
    log.info("  RUN-ALL: Full pipeline starting")
    log.info("=" * 60)
    cmd_sentiment(args, cfg)
    cmd_features(args, cfg)
    cmd_fuse(args, cfg)
    cmd_train(args, cfg)
    # Evaluate on validation split after training
    args.split = "val"
    cmd_eval(args, cfg)
    log.info("=" * 60)
    log.info("  RUN-ALL complete. Run `python main.py eval --split test` for final test metrics.")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI parser
# ─────────────────────────────────────────────────────────────────────────────

COMMANDS = {
    "generate-data":  cmd_generate_data,
    "finetune-muril": cmd_finetune_muril,
    "sentiment":      cmd_sentiment,
    "features":       cmd_features,
    "fuse":           cmd_fuse,
    "train":          cmd_train,
    "eval":           cmd_eval,
    "predict":        cmd_predict,
    "tier1":          cmd_tier1,
    "tier2":          cmd_tier2,
    "tier3":          cmd_tier3,
    "run-all":        cmd_run_all,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Stock Direction Predictor — Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("command",    choices=list(COMMANDS), help="Pipeline stage to run")
    p.add_argument("--config",   default="config.yaml",  help="Path to config.yaml")
    p.add_argument("--force",    action="store_true",     help="Bypass all caches")
    p.add_argument("--device",   default=None,            help="cpu or cuda")
    p.add_argument("--ticker",   default=None,            help="Restrict to one ticker")
    p.add_argument("--profile",  default=None,            choices=["4gb","8gb","full"],
                   help="Override vram_profile from config")
    p.add_argument("--split",    default="test",          choices=["train","val","test"],
                   help="Split to evaluate (used by eval command)")
    p.add_argument("--resume",   default=None,            help="Path to checkpoint to resume from")
    p.add_argument("--date",     default=None,            help="Target date for tier1 (YYYY-MM-DD)")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Load config
    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        log.error(f"config.yaml not found at {cfg_path}. Are you in the project root?")
        sys.exit(1)

    cfg = load_config(cfg_path, profile_override=args.profile)

    # Create all directories
    ensure_dirs()

    log.info(f"Command: {args.command} | Profile: {cfg.get('vram_profile','8gb')} | Force: {args.force}")

    COMMANDS[args.command](args, cfg)


if __name__ == "__main__":
    main()
