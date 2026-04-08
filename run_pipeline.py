#!/usr/bin/env python3
"""
run_pipeline.py
Orchestrates the entire NIFTY-NLP workflow.

Usage:
    conda activate nifty-rtx5060
    python run_pipeline.py
    python run_pipeline.py --skip-nlp          # Skip NLP if handshake CSVs exist
    python run_pipeline.py --skip-train-lstm   # Skip LSTM training (use saved model)
    python run_pipeline.py --skip-sentiment-train  # Skip MuRIL fine-tuning
"""

import argparse
import subprocess
import sys
import pathlib
import datetime

ROOT = pathlib.Path(__file__).resolve().parent


def run_step(script_path: str, args: list[str], description: str) -> None:
    """Run one pipeline step as a subprocess, halt on failure."""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING: {description}")
    print(f"📦 COMMAND: python {script_path} {' '.join(args)}")
    try:
        subprocess.run([sys.executable, script_path] + args, check=True)
    except subprocess.CalledProcessError:
        print(f"\n❌ ERROR in {script_path}. Pipeline halted.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY-NLP Master Pipeline")
    parser.add_argument(
        "--skip-nlp",
        action="store_true",
        help="Skip NLP scoring steps (requires en/hi_sentiment CSVs to exist)",
    )
    parser.add_argument(
        "--skip-sentiment-train",
        action="store_true",
        help="Skip MuRIL fine-tuning (use pre-trained model in models/muril_financial_sentiment_v1/)",
    )
    parser.add_argument(
        "--skip-train-lstm",
        action="store_true",
        help="Skip LSTM training (requires models/prod_binary_lstm_best.pth to exist)",
    )
    parser.add_argument(
        "--train-path",
        default=str(ROOT / "data" / "inputs" / "prod_train.csv"),
        help="Path to LSTM training CSV",
    )
    parser.add_argument(
        "--test-path",
        default=str(ROOT / "data" / "inputs" / "prod_test.csv"),
        help="Path to LSTM test CSV",
    )
    parser.add_argument(
        "--model-out",
        default=str(ROOT / "models" / "prod_binary_lstm_best.pth"),
        help="Output path for trained LSTM model",
    )
    parser.add_argument(
        "--scaler-out",
        default=str(ROOT / "models" / "prod_scaler.joblib"),
        help="Output path for StandardScaler",
    )
    args = parser.parse_args()

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    print("\n" + "=" * 80)
    print("📊 NIFTY-NLP PIPELINE STARTING")
    print(f"   Date: {today_str}")
    print("=" * 80)

    # ── STAGE A: MuRIL Sentiment Model ────────────────────────────────────────
    if not args.skip_sentiment_train:
        run_step(
            str(ROOT / "src" / "scripts" / "train_sentiment_model.py"),
            [],
            "Stage A: Fine-tune MuRIL on Synthetic Hindi/Hinglish Financial Data",
        )
    else:
        print("\n⏩ SKIPPING MuRIL fine-tuning (--skip-sentiment-train).")

    # ── STAGE B: NLP Scoring (FinBERT + MuRIL → Handshake CSV) ──────────────
    if not args.skip_nlp:
        run_step(
            str(ROOT / "src" / "sentiment" / "analyzer_en.py"),
            [],
            "Stage B-1: FinBERT English Sentiment Scoring",
        )
        run_step(
            str(ROOT / "src" / "sentiment" / "analyzer_hi.py"),
            [],
            "Stage B-2: MuRIL Hindi/Hinglish Sentiment Scoring",
        )
        run_step(
            str(ROOT / "src" / "preprocessing" / "signal_merging.py"),
            [],
            "Stage B-3: Trust-Weight Fusion → Handshake CSV",
        )
    else:
        print("\n⏩ SKIPPING NLP scoring (--skip-nlp). Using existing sentiment CSVs.")

    # ── STAGE C: LSTM Training ────────────────────────────────────────────────
    if not args.skip_train_lstm:
        run_step(
            str(ROOT / "src" / "scripts" / "train.py"),
            [
                "--train-path", args.train_path,
                "--test-path",  args.test_path,
                "--model-out",  args.model_out,
                "--scaler-out", args.scaler_out,
            ],
            "Stage C: Train Binary LSTM (up to 100 epochs, early-stop @ patience=15)",
        )
    else:
        print("\n⏩ SKIPPING LSTM training (--skip-train-lstm). Using saved model.")

    # ── STAGE D: Prediction ───────────────────────────────────────────────────
    run_step(
        str(ROOT / "src" / "scripts" / "predict.py"),
        [
            "--model-path",  args.model_out,
            "--scaler-path", args.scaler_out,
            "--test-path",   args.test_path,
            "--output-path", str(ROOT / "data" / "predictions" / "production_predictions.csv"),
        ],
        "Stage D: MC Dropout Prediction (50 forward passes per ticker)",
    )

    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(
        f"👉 Check data/predictions/production_predictions.csv for the latest signals."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
