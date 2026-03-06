# Nifty 50 News-Driven Prediction Engine

## Overview
This platform predicts the 10:00 AM next-day price move for Nifty 50 stocks by fusing multilingual news sentiment (Hindi + English) with 20 years of historical OHLCV data. Output is a regression float with a confidence interval. The primary model is a T-LSTM Dual Stream Architecture with 3 tiers of adaptive learning (Trust Weights, Weekly Elastic Weight Consolidation, and Monthly Walk-forward Retraining).

## Architecture Status: COMPLETE - Version 1.0 (March 2026)

### Directory Structure
- `configs/`: YAML Configuration files.
- `data/`: Storage for Kaggle historical datasets, handshake CSV, prediction labels, and JSON trust weights.
- `logs/`: Application execution logs.
- `models/`: Checkpoints and serialized model objects.
- `notebooks/`: Exploration & reporting notebooks.
- `src/`: Complete source code pipeline.
  - `data_pipeline/`: Data ingestion, daily CSV downloads, RSS scraping.
  - `evaluation/`: Gap metrics, comparative benchmarks, tracking.
  - `models/`: PyTorch definitions of Dual-Stream T-LSTM, Huber Head, EWC logic.
  - `nlp/`: FinBERT (English), MuRIL/IndicBERT (Hindi), and `lxyuan/distilbert-base-multilingual` component for extracting multilingual headlines.
  - `training/`: Forward passes, loops, monthly walk-forward execution, Tier 1/2 nudges.
  - `utils/`: Pathing via pathlib, safe logging, global configs.
- `tests/`: Automated unit and integration testing.

### Setup Instructions
Run this identically on Machine 1, Machine 2, and Machine 3:
```bash
conda env create -f environment.yml
conda activate nifty-nlp
```

### Staged Roadmap (Stage 0 Focus)
Currently completing **Stage 0** which involves:
1. Building structure & conda `environment.yml`.
2. Verifying directories and pathlib structure mapping.
3. Loading & asserting `data/` storage integrity (Targeting Machine 3 Kaggle loads).
4. Testing baseline architectures setup.
5. Integrates dynamic distilbert Sentinel multilingal tracking into pipeline processing.
5. Deployed standalone Multilingual Stock Sentiment component utilizing distilbert to detect market polarity across Hindi, Hinglish, and English dummy datasets.
