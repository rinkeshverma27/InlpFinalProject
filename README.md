# InlpFinalProject — News-Driven Financial Market Prediction

**Team YRP | IIIT Hyderabad**

| Member | Roll No. | 
|---|---|
| Yogendra Patel | 2025201098 | 
| Rinkesh Verma | 2025201070 | 
| Prabhash Padhan | 2025201089 | 

---

## What this project does

Most stock prediction systems either ignore news entirely and work only with price history, or they read only English text and miss the large Hindi-language financial media that Indian retail investors rely on. This project builds an end-to-end pipeline that:

1. Scrapes English financial news from Moneycontrol, NSE, and other sources via RSS
2. Scores each article using **FinBERT** (a finance-specific sentiment model)
3. Merges daily sentiment scores with OHLCV price data and technical indicators
4. Trains a **3-layer Binary LSTM** to predict whether each of 15 Nifty-50 stocks will go **UP or DOWN** the next trading day
5. Outputs predictions with confidence scores so you can filter for high-signal calls

Hindi sentiment via **MuRIL** is built and validated but not yet merged into training — that is the immediate next step.

**Current results:** 60.42% ± 2.95% mean directional accuracy on a Jan–Feb 2025 test set. High-confidence predictions (p > 0.6) hit 67.11% ± 0.63%.

---

## Project structure

```
InlpFinalProject/
│
├── data/
│   ├── inputs/
│   │   ├── prod_train.csv          # Training data (Jan 2023 – Dec 2024, 7365 rows)
│   │   └── prod_test.csv           # Test data (Jan – Feb 2025, 300 rows)
│   └── predictions/
│       └── production_predictions.csv   # Model output with confidence scores
│
├── models/
│   ├── prod_binary_lstm_best.pth   # Best trained LSTM checkpoint (seed 42)
│   └── prod_scaler.joblib          # StandardScaler fitted on training data
│
├── src/
│   ├── gathering/                  # RSS scraping utilities
│   ├── preprocessing/
│   │   ├── cleaning.py             # OHLCV cleaning, pilot selection, label construction
│   │   ├── feature_engineering.py  # RSI, MACD, Bollinger Band Width, SMA distance
│   │   └── signal_merging.py       # Merges sentiment scores with price features
│   ├── sentiment/
│   │   ├── analyzer_en.py          # FinBERT inference on English articles
│   │   ├── analyzer_hi.py          # MuRIL inference on Hindi/Hinglish articles
│   │   ├── news_classifier.py      # Ticker-to-article matching
│   │   └── synthetic_data_gen.py   # Generates synthetic Hindi training data for MuRIL
│   ├── modeling/
│   │   ├── lstm_binary.py          # BinaryLSTM model definition (PyTorch)
│   │   └── dataloader.py           # ProductionStockDataset (sliding window)
│   └── scripts/
│       ├── train.py                # Training loop with multi-seed runs and early stopping
│       ├── predict.py              # Inference script — generates predictions CSV
│       └── train_sentiment_model.py # Fine-tunes MuRIL on Hindi financial data
│
├── scripts/
│   ├── batch_run.py                # Runs train + predict for multiple tickers in one go
│   └── check_baseline.py           # Computes market majority-class baseline for comparison
│
├── streamlit_gui/
│   └── app.py                      # Interactive dashboard for running and viewing predictions
│
├── environment.yml                 # Conda environment spec
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/rinkeshverma27/InlpFinalProject.git
cd InlpFinalProject
```

### 2. Create the conda environment

The project uses Python 3.10 with PyTorch 2.3.1 and CUDA 12.1. If you are on a machine without a GPU, PyTorch will fall back to CPU automatically (training will be slower).

```bash
conda env create -f environment.yml
conda activate nifty-nlp
```

### 3. Verify the setup

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## How to run

### Step 1 — Clean market data and select pilot stocks

```bash
python src/preprocessing/cleaning.py
```

This reads the raw Nifty-50 OHLCV data, audits each stock for data quality (10+ years of history, fewer than 10 gaps), selects the top 15 stocks by trading volume, and constructs the market-corrected binary labels. It outputs `prod_train.csv` and `prod_test.csv` under `data/inputs/`.

The market-corrected label is calculated as:

```
target = tomorrow's stock return - tomorrow's Nifty-50 index return
label  = 1 (UP) if target > +0.5%
         0 (DOWN) if target < -0.5%
         (neutral days dropped as noise)
```

Subtracting the index return removes broad market moves and isolates stock-specific signals driven by company news.

### Step 2 — Run English sentiment (FinBERT)

```bash
python src/sentiment/analyzer_en.py
```

Reads `data/news/news_articles_timestamped.csv`, runs FinBERT on all English headlines in batches of 16, and saves sentiment scores to `data/news/processed/en_sentiment.csv`. Each score is a continuous value in `[-1, +1]` — positive means the article is bullish, negative means bearish, and the magnitude reflects model confidence.

If no news CSV exists, the script runs on a small set of built-in dummy articles for testing.

> **GPU note:** FP16 half-precision is used automatically if CUDA is available, keeping VRAM under 3.5 GB.

### Step 3 — Run Hindi sentiment (MuRIL) *(in progress)*

```bash
python src/sentiment/analyzer_hi.py
```

Same structure as the English pipeline but uses MuRIL (`google/muril-base-cased`), which handles Hindi and Hinglish natively. The base MuRIL model does not have a fine-tuned sentiment head yet — fine-tuning on Hindi financial data is ongoing. The script currently falls back to a multilingual distilBERT student model for pipeline validation.

### Step 4 — Merge sentiment with price data

```bash
python src/preprocessing/signal_merging.py
```

For each (ticker, date) pair, takes the confidence-weighted average of all article sentiment scores for that day and joins with the OHLCV feature matrix. Days with no news coverage get `0.0` in the sentiment columns. The output is the final 17-feature input for the LSTM.

### Step 5 — Train the model

```bash
python src/scripts/train.py \
  --train-path data/inputs/prod_train.csv \
  --test-path  data/inputs/prod_test.csv \
  --model-out  models/prod_binary_lstm_best.pth \
  --scaler-out models/prod_scaler.joblib
```

This runs three independent training runs (seeds 42, 43, 44) and saves the best checkpoint by validation accuracy. Training runs for up to 100 epochs with early stopping (patience = 15). You will see output like:

```
Using Device: cuda
Train samples after filtering: 4522
Test samples after filtering: 214
=== Seed 42 ===
--> Improvement! Best Acc: 56.25%
Epoch 10/100 | Loss: 3.4533 | Val Acc: 56.25%
...
Early stopping at epoch 53
✅ Production Model Result (Seed 42):
Base Accuracy: 62.50%
🔥 HIGH CONFIDENCE ACCURACY (p>0.6): 66.67%

📊 Multi-Seed Summary
Base Accuracy Mean±Std: 60.42% ± 2.95%
High Confidence Accuracy Mean±Std: 67.11% ± 0.63%
```

### Step 6 — Generate predictions

```bash
python src/scripts/predict.py \
  --model-path  models/prod_binary_lstm_best.pth \
  --scaler-path models/prod_scaler.joblib \
  --test-path   data/inputs/prod_test.csv \
  --output-path data/predictions/production_predictions.csv
```

For each of the 15 pilot stocks, the script takes the most recent 10-day window from the test data, runs a forward pass, and saves a CSV with columns: `Ticker`, `Date`, `Prediction`, `Confidence`, `Probability_UP`, `Probability_DOWN`, `Significant_Signal`.

The `Significant_Signal` column is `YES` when confidence exceeds 60% — those are the high-quality calls worth paying attention to.

Example output:

```
Ticker,Date,Prediction,Confidence,Probability_UP,Probability_DOWN,Significant_Signal
KOTAKBANK.NS,2025-01-28,DOWN,80.99%,19.01%,80.99%,YES
RELIANCE.NS,2025-01-28,DOWN,78.29%,21.71%,78.29%,YES
INFY.NS,2025-01-28,DOWN,76.55%,23.45%,76.55%,YES
BPCL.NS,2025-01-28,UP,65.74%,65.74%,34.26%,YES
```

### Run everything at once (batch mode)

```bash
python scripts/batch_run.py
```

Runs train and predict back-to-back for a set of tickers in one command. Edit the `top_tickers` list inside the script to change which tickers are processed.

### Check the majority-class baseline

```bash
python scripts/check_baseline.py
```

Reads a predictions CSV and computes the natural UP/DOWN split for the test period. Prints whether the model's accuracy beats the simple majority-class baseline.

---

## Streamlit dashboard

```bash
conda activate nifty-nlp
streamlit run streamlit_gui/app.py
```

Opens a browser UI where you can:
- Browse and select train/test/model/scaler files using dropdowns
- Trigger training or prediction runs directly from the interface
- View and filter the predictions CSV in an interactive table

---

## Dataset details

Both CSVs contain the same 17 columns:

| Column | Description |
|---|---|
| `ticker` | NSE ticker symbol (e.g. `RELIANCE`) |
| `Date` | Trading date |
| `Daily_Return` | Day's percentage price change |
| `Volatility_20D` | Rolling 20-day return volatility |
| `MA_50`, `MA_200` | 50-day and 200-day moving averages |
| `PE_Ratio`, `Forward_PE` | Valuation multiples |
| `Price_to_Book`, `Dividend_Yield`, `Beta` | Additional fundamentals |
| `nifty_ret_proxy` | Nifty-50 index return (used for market-corrected label) |
| `rsi` | RSI-14 momentum indicator |
| `macd_diff` | MACD histogram value |
| `bb_width` | Bollinger Band Width (volatility) |
| `dist_from_sma` | Distance of price from 20-day SMA |
| `vol_delta` | Day-over-day volume change |
| `en_sentiment` | FinBERT English sentiment score in [-1, +1] |
| `hi_sentiment` | MuRIL Hindi sentiment score in [-1, +1] (currently 0.0 placeholder) |

**15 pilot stocks:** RELIANCE, HDFCBANK, ICICIBANK, INFY, KOTAKBANK, TATASTEEL, SBIN, NTPC, BAJFINANCE, POWERGRID, ONGC, WIPRO, ITC, BPCL, BHARTIARTL.

These 15 were selected from all 50 Nifty constituents based on: 10+ years of clean price history, fewer than 10 data gaps, and highest average daily trading volume (volume is a proxy for news density).

---

## Model architecture

The `BinaryLSTM` model (`src/modeling/lstm_binary.py`):

```
Input: (batch, 10 days, 17 features)
       ↓
  LSTM Layer 1  →  256 hidden units
       ↓  Dropout(0.3)
  LSTM Layer 2  →  256 hidden units
       ↓  Dropout(0.3)
  LSTM Layer 3  →  256 hidden units
       ↓
  Final hidden state h_n  (256-dim)
       ↓
  Linear(256 → 128)  +  ReLU
       ↓  Dropout(0.3)
  Linear(128 → 64)   +  ReLU
       ↓
  Linear(64 → 2)
       ↓
  Softmax  →  P(UP),  P(DOWN)
```

**Training config:** Adam lr=5e-4, CrossEntropy loss, 100 epochs max, early stopping patience=15, multi-seed (42 / 43 / 44).

---

## Results

| Run | Base Accuracy | High-Confidence (p > 0.6) |
|---|---|---|
| Seed 42 | 62.50% | 66.67% |
| Seed 43 | 56.25% | 66.67% |
| Seed 44 | 62.50% | — |
| **Mean ± Std** | **60.42% ± 2.95%** | **67.11% ± 0.63%** |
| Previous model (mixed data) | 53.09% | — |
| Random baseline | 50.00% | — |

The jump from 53% to 62% came mainly from restricting training to post-COVID data (2023+). Pre-COVID and post-COVID markets behave differently enough in terms of volatility, volumes, and news-price dynamics that mixing them confuses the model.

---

## What's next

- **Hindi integration:** Fine-tune MuRIL on real Hindi financial news plus the synthetic dataset, then replace the zero-filled `hi_sentiment` column with actual scores
- **Ablation study:** Compare technical-only vs. +English vs. +Hindi vs. +both to quantify each source's contribution
- **Attention layer:** Add attention over the 10 LSTM hidden states so the model can focus on the most informative days in the window
- **Time-weighted sentiment:** Give higher weight to articles published close to market open rather than averaging all articles equally
- **Broader coverage:** Extend to more Nifty-50 stocks beyond the current 15 pilots

---

## Key dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | 3.10 | — |
| PyTorch | 2.3.1 | LSTM model, GPU training |
| transformers | 4.42.4 | FinBERT and MuRIL inference |
| scikit-learn | 1.5.1 | Scaling, evaluation |
| pandas | 2.2.2 | Data handling |
| yfinance | 0.2.40 | Market data collection |
| feedparser | 6.0.11 | RSS feed parsing |
| streamlit | latest | Dashboard |

Full list with pinned versions in `environment.yml`.

---

## Academic context

This project is part of the Introduction to NLP (INLP) course at IIIT Hyderabad. It is a work in progress — the midterm report is available in the repo.
