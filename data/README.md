# Production Dataset (Post-COVID)

This directory contains the finalized, consolidated datasets for the T-LSTM Dual-Stream model. All legacy and intermediate files have been discarded to focus strictly on the 2023–2025 regime.

## 📂 Final Files

### 1. `prod_train.csv` (7,365 rows)

- **Timeframe:** Jan 1, 2023 – Dec 31, 2024
- **Tickers:** All 15 pilot tickers (RELIANCE, HDFCBANK, ICICIBANK, INFOSYS, etc.)
- **Content:** Fused OHLCV data + Sequential News Sentiment (English & Hindi).
- **Purpose:** Full training and validation set for Phase 8+ models.

### 2. `prod_test.csv` (300 rows)

- **Timeframe:** Jan 1, 2025 – Feb 28, 2025
- **Content:** Out-of-sample test data used for project validation (Market Edge Tests).

## 🛠️ Feature Set (11 Features)

The following features are available in both CSVs and are fed into the T-LSTM:

1. `Open`, `High`, `Low`, `Close`, `Volume` (Standard Price Action)
2. `Daily_Return` (Shifted target proxy)
3. `en_sentiment` (Pre-processed English News Sentiment)
4. `hi_sentiment` (Pre-processed Hindi News Sentiment)
5. `rsi`, `macd_diff`, `bb_width` (Technical Indicators)

---

**Note:** For future use, simply load these files and filter by the `ticker` column in your training script.
