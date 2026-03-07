# Running the T-LSTM Pipeline

This guide explains how to use `src/main.py` to train and evaluate the updated Dual-Stream model.

## Prerequisites

Ensure you are in the internal project directory and have activated the `nifty-nlp` environment:

```bash
conda activate nifty-nlp
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

## 1. Full Training

Train the model on the consolidated 2023-2025 dataset for a specific ticker. This will also save a checkpoint and compute the Fisher matrix for future updates.

```bash
python src/main.py --mode train \
    --ticker RELIANCE \
    --ohlcv data/price/RELIANCE/ohlcv.csv \
    --nifty data/price/NIFTY50/ohlcv.csv
```

## 2. Generate Predictions

Produce next-day predictions and a detailed evaluation report using a trained checkpoint.

```bash
python src/main.py --mode predict \
    --ticker RELIANCE \
    --ohlcv data/price/RELIANCE/ohlcv.csv \
    --nifty data/price/NIFTY50/ohlcv.csv \
    --checkpoint outputs/checkpoints/model_2026-03-07.pt
```

## 3. EWC Nudge (Incremental Update)

Fine-tune the model on the latest available days without forgetting prior training weights.

```bash
python src/main.py --mode ewc_nudge \
    --ticker RELIANCE \
    --ohlcv data/price/RELIANCE/ohlcv.csv \
    --nifty data/price/NIFTY50/ohlcv.csv \
    --checkpoint outputs/checkpoints/model_2026-03-07.pt
```

---
**Note:** All consolidated tickers (15 total) follow the same path pattern: `data/price/{TICKER}/ohlcv.csv`.
