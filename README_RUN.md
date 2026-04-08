# NIFTY-NLP — Run Guide

## Prerequisites

```bash
conda activate nifty-rtx5060
cd /path/to/InlpFinalProject
```

---

## Quick Start (Full Pipeline)

```bash
# First-ever run (trains MuRIL + LSTM, scores news, generates predictions)
python run_pipeline.py

# Skip MuRIL training (model already in models/muril_financial_sentiment_v1/)
python run_pipeline.py --skip-sentiment-train

# Skip LSTM training + NLP (use saved model + existing handshake CSVs)
python run_pipeline.py --skip-sentiment-train --skip-nlp --skip-train-lstm
```

---

## Step-by-Step (Manual)

### Step 1 — Generate Synthetic Hindi Training Data (one-time)
```bash
python src/scripts/generate_mega_synthetic.py
# → data/inputs/mega_synthetic_hindi_train.csv  (8,000 samples)
```

### Step 2 — Fine-tune MuRIL Sentiment Model (one-time, Machine 1 GPU)
```bash
python src/scripts/train_sentiment_model.py
# → models/muril_financial_sentiment_v1/  (fine-tuned model)
```

### Step 3 — Run English Sentiment Scoring (daily, Machine 2)
```bash
python src/sentiment/analyzer_en.py
# → data/news/processed/en_sentiment.csv
```

### Step 4 — Run Hindi Sentiment Scoring (daily, Machine 1)
```bash
python src/sentiment/analyzer_hi.py
# → data/news/processed/hi_sentiment.csv
```

### Step 5 — Generate Trust-Weighted Handshake CSV (daily, Machine 1)
```bash
python src/preprocessing/signal_merging.py
# → data/processed/handshake.csv
```

### Step 6 — Train Binary LSTM (one-time / monthly retrain)
```bash
python src/scripts/train.py \
  --train-path data/inputs/prod_train.csv \
  --test-path  data/inputs/prod_test.csv \
  --model-out  models/prod_binary_lstm_best.pth \
  --scaler-out models/prod_scaler.joblib
```

### Step 7 — Generate Predictions (daily, 9:05 PM)
```bash
python src/scripts/predict.py
# → data/predictions/production_predictions.csv
```

### Step 8 — Update Trust Weights (daily, 10:10 AM next day)
```bash
python src/preprocessing/tier1_trust_updater.py --date $(date +%Y-%m-%d)
# → data/trust_weights/YYYY-MM-DD.json
```

---

## Launch Streamlit Dashboard

```bash
conda activate nifty-rtx5060
streamlit run streamlit_gui/app.py
```

---

## Key File Locations

| Artifact | Path |
|---|---|
| MuRIL model | `models/muril_financial_sentiment_v1/` |
| LSTM model | `models/prod_binary_lstm_best.pth` |
| Scaler | `models/prod_scaler.joblib` |
| Training data | `data/inputs/prod_train.csv` |
| Test data | `data/inputs/prod_test.csv` |
| Synthetic Hindi data | `data/inputs/mega_synthetic_hindi_train.csv` |
| Daily predictions | `data/predictions/production_predictions.csv` |
| Handshake CSV | `data/processed/handshake.csv` |
| Trust weights | `data/trust_weights/{date}.json` |
| Hindi sentiment | `data/news/processed/hi_sentiment.csv` |
| English sentiment | `data/news/processed/en_sentiment.csv` |
| Raw news (scraped) | `dataset/raw_dataset/english_news_nifty50.csv` |