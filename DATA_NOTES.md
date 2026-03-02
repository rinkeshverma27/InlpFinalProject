# 📦 DATA NOTES — News-Driven Financial Prediction Engine
**Reference for all data requirements, formats, and sources**

---

## 1. Historical Price Data

### Source
- **Dataset:** Kaggle 20-Year Nifty 50 OHLCV dataset
- **Storage:** Machine 3 SSD (primary), backed up to Git LFS
- **Stocks:** 15 Nifty 50 stocks (3 pilot stocks first)

### Expected Schema — Raw OHLCV
```
data/price/{TICKER}/ohlcv.csv
```

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | DATE (YYYY-MM-DD) | Trading day | 2025-03-01 |
| `ticker` | STRING | NSE stock symbol | RELIANCE |
| `open` | FLOAT | Opening price (INR) | 2840.50 |
| `high` | FLOAT | Intraday high (INR) | 2875.00 |
| `low` | FLOAT | Intraday low (INR) | 2820.00 |
| `close` | FLOAT | Closing price (INR) | 2865.25 |
| `volume` | INT | Shares traded | 4820000 |
| `nifty50_close` | FLOAT | Nifty 50 index close | 22350.10 |
| `nifty50_open` | FLOAT | Nifty 50 index open | 22290.00 |

### Data Split — Walk-Forward
| Split | Years | Usage |
|-------|-------|-------|
| **Training** | Years 1–17 (expanding) | Walk-forward expanding window |
| **Validation** | Years 18–19 (3-month rolling) | Rolling validation window |
| **HOLDOUT** | Year 20 | 🔒 SEALED — never used until Stage 3 |

> **CRITICAL:** Walk-forward only. Zero random train/test splits. Year 20 is physically sealed.

### Label Construction Formula
```python
# Step 1: Raw return
raw_ret = (price_10am_t1 - close_t0) / close_t0

# Step 2: Confound-corrected label (removes overnight market drift)
nifty50_overnight_ret = (nifty50_open_t1 - nifty50_close_t0) / nifty50_close_t0
adjusted_ret = raw_ret - nifty50_overnight_ret

# Output: continuous float, e.g. +0.018 means +1.8% stock-specific move
```

**Why confound-correct?** Removes SGX Nifty / US market drift so the label reflects only stock-specific news impact.

### Stream B LSTM Features (per timestep)
| Feature | Transform | Notes |
|---------|-----------|-------|
| Open, High, Low, Close | Min-max normalised per stock | 4 features |
| Volume | Log-scaled | `log1p(volume)` |
| 5-day MA delta | `close - ma5` | Momentum |
| 20-day MA delta | `close - ma20` | Trend |
| Realised volatility | 5-day rolling std of returns | Risk state |
| RSI (14-period) | Standard RSI formula | Momentum |
| Market regime flag | 0=normal, 1=moderate, 2=event | Integer |

### Dynamic Window Sizes
| Condition | Window | Trigger |
|-----------|--------|---------|
| RealVol < 1.0x historical avg | 60 days | Normal market |
| RealVol 1.5–2.0x avg | 30 days | Elevated volatility |
| RealVol > 2.0x avg | 10 days | Event mode |

---

## 2. News Data (RSS Scraping)

### News Sources
| Source | Language | Coverage | Fallback |
|--------|----------|---------|---------|
| Economic Times | English | All 15 stocks | LiveMint |
| Moneycontrol | English | All 15 stocks | LiveMint |
| Business Standard | English | All 15 stocks | NDTV Profit |
| NDTV Profit (EN) | English | All 15 stocks | Hindu BusinessLine |
| NSE Official Announcements | English | All 15 stocks (authoritative) | BSE Announcements |
| Navbharat Times | Hindi | Large-cap stocks | Dainik Bhaskar |
| Amar Ujala | Hindi | Large-cap stocks | Dainik Bhaskar |
| Zee Business Hindi | Hindi | Financial-specific | NDTV Profit Hindi |

### Raw Article Schema
```
data/news/raw/{YYYY-MM-DD}/{source}_{ticker}.jsonl
```

Each line is a JSON object:
```json
{
  "article_id": "ET_RELIANCE_20260301_001",
  "source": "economic_times",
  "language": "en",
  "ticker": "RELIANCE",
  "headline": "Reliance Industries reports record Q3 profit...",
  "body": "...",
  "published_at": "2026-03-01T20:15:00+05:30",
  "scraped_at": "2026-03-01T20:18:32+05:30",
  "url": "https://..."
}
```

### Timestamp Alignment Rule
- Articles published **8 PM to 11:59 PM (Day T)** → inform prediction for **10 AM Day T+1**
- Articles published **12 AM to 7:59 AM (Day T+1)** → informed prediction for **10 AM Day T+1**
- Articles after **8 AM** on T+1 → **excluded** (post-open, too late)
- Market holidays: time window extends as needed — test this explicitly

### Deduplication Rule
- Deduplicate on `(source, ticker, headline)` within the 8PM–10AM window
- Same story reprinted across sources: keep all (trust weights handle source quality)

---

## 3. Sentiment Scores — CSV Handshake Format

### FinBERT English Sentiment CSV (Machine 2 → Machine 1)
```
data/handshake/en_sentiment_{YYYY-MM-DD}.csv
```

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | DATE | Trading day this sentiment informs | 2026-03-02 |
| `ticker` | STRING | NSE stock code | RELIANCE |
| `source` | STRING | News source name | economic_times |
| `language` | STRING | `en` | en |
| `sentiment_score` | FLOAT | FinBERT score in [-1, +1] | +0.72 |
| `sentiment_label` | STRING | positive / negative / neutral | positive |
| `confidence` | FLOAT | Softmax confidence [0, 1] | 0.89 |
| `article_count` | INT | # articles aggregated | 3 |
| `embedding_512` | FLOAT[512] | Mean-pooled CLS embedding | [...] |

### MuRIL Hindi Sentiment CSV (Machine 1 internal)
```
data/handshake/hi_sentiment_{YYYY-MM-DD}.csv
```
Same schema as EN sentiment CSV with `language = "hi"`.

**Hindi Sparse Rule:** If `article_count < 2` in past 3 trading days for a stock → Hindi trust weight set to `0.0`, English normalised to `1.0` for that stock on that day.

### Fused Handshake CSV (Trust-weighted, Machine 1)
```
data/handshake/fused_{YYYY-MM-DD}.csv
```

| Column | Type | Description |
|--------|------|-------------|
| `date` | DATE | Trading day |
| `ticker` | STRING | NSE stock code |
| `fused_sentiment` | FLOAT | Trust-weighted combined sentiment [-1, +1] |
| `en_weight` | FLOAT | Trust weight for English sources |
| `hi_weight` | FLOAT | Trust weight for Hindi sources |
| `fused_embed_768` | FLOAT[768] | Concat [en_embed_512, hi_embed_256] |
| `hindi_sparse_flag` | BOOL | True if Hindi weight set to 0 |

---

## 4. Trust Weights

### Storage Format
```
data/trust_weights/{YYYY-MM-DD}.json
```

```json
{
  "date": "2026-03-02",
  "version": "v1",
  "stocks": {
    "RELIANCE": {
      "economic_times": 1.40,
      "moneycontrol": 1.15,
      "navbharat_times": 0.80,
      "amar_ujala": 0.95,
      "nse_announcements": 1.60
    },
    "TCS": { ... }
  }
}
```

**Trust weight rules:**
- Bounds: `[0.1, 2.0]` — hard clamped
- No single source > 40% of total weight per stock
- Sources inactive 7+ days: decay toward `1.0` at 2%/day
- Skip update if `|actual_move| > 3σ` historical std (anomaly guard)
- Files are **immutable once written** — committed to Git nightly

---

## 5. Model Checkpoints

### Naming Convention
```
models/checkpoints/{stage}/{ticker_or_all}/model_{YYYY-MM-DD}.pt
```

| File | Description |
|------|-------------|
| `model_{date}.pt` | Full model weights |
| `fisher_matrix_{date}.pt` | Fisher Information Matrix for EWC |
| `optimizer_{date}.pt` | Optimizer state (for resuming) |
| `metadata_{date}.json` | MAE, direction acc, dataset range, git hash |

### Metadata JSON (per checkpoint)
```json
{
  "checkpoint_hash": "v2.3-20260301",
  "created_at": "2026-03-01T23:00:00+05:30",
  "training_range": "2006-01-01 to 2026-02-28",
  "stocks": ["RELIANCE", "TCS", "INFY"],
  "val_mae": 0.0128,
  "val_direction_acc": 0.561,
  "deployed": true,
  "stage": "2A"
}
```

---

## 6. Predictions Output

### predictions.csv — Daily Production Output
```
data/predictions/predictions_{YYYY-MM-DD}.csv
```

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `date` | DATE | Trading day of prediction | 2026-03-15 |
| `ticker` | STRING | NSE stock code | RELIANCE |
| `predicted_pct` | FLOAT | Predicted adjusted % move at 10 AM | +0.018 (+1.8%) |
| `conf_low` | FLOAT | 95% CI lower bound | +0.014 |
| `conf_high` | FLOAT | 95% CI upper bound | +0.022 |
| `direction` | STRING | UP / DOWN / NEUTRAL | UP |
| `magnitude_label` | STRING | Strong / Moderate / Weak (CI width) | Moderate |
| `model_version` | STRING | Checkpoint hash | v2.3-20260301 |
| `active_trust_weights` | JSON | Source weights used | {"ET":1.4,"NBT":0.8} |
| `actual_pct` | FLOAT | Filled at 10:05 AM next day | +0.021 |
| `gap` | FLOAT | |pred_pct - actual_pct| (filled 10:05 AM) | 0.003 |

**Confidence interval method:** MC Dropout with 50 forward passes → `mean ± 1.96 * std = 95% CI`

**Direction rule:**
- `predicted_pct > +0.002` → UP
- `predicted_pct < -0.002` → DOWN
- Otherwise → NEUTRAL

**Magnitude rule (based on CI width = conf_high - conf_low):**
- CI width < 0.005 → Strong (tight confidence)
- CI width 0.005–0.012 → Moderate
- CI width > 0.012 → Weak (uncertain)

---

## 7. Data Needed — Stage-by-Stage Summary

| Stage | Data Required | Format | Notes |
|-------|--------------|--------|-------|
| Stage 0 | Kaggle 20yr OHLCV | CSV (OHLCV schema above) | Load on M3 SSD, audit first |
| Stage 0 | 50 sample Hindi headlines | Plain text / JSONL | For MuRIL vs IndicBERT eval |
| Stage 0 | 5 trading day timestamps | Manual records | For alignment testing |
| Stage 0 | 10 known high-news days | From Kaggle + news archive | For label spot-check |
| Stage 1 | 3 pilot stocks full history | Subset of OHLCV | RELIANCE, TCS, + 1 Hindi-heavy |
| Stage 1 | 500 financial Hindi samples | JSONL (headline + label) | For fine-tuning if needed |
| Stage 2A | Live RSS feeds | JSONL (raw article schema) | Scrape from 8 PM nightly |
| Stage 2A | FinBERT / MuRIL models | HuggingFace pretrained | Download and cache locally |
| Stage 2B | All 15 stocks OHLCV | Full OHLCV CSV | Expand from pilot subset |
| Stage 3 | Year 20 holdout | Sealed subset of OHLCV | DO NOT OPEN before Stage 3 |

---

## 8. Directory Structure (Recommended)

```
InlpFinalProject/
├── data/
│   ├── price/
│   │   └── {TICKER}/
│   │       └── ohlcv.csv
│   ├── news/
│   │   └── raw/
│   │       └── {YYYY-MM-DD}/
│   │           └── {source}_{ticker}.jsonl
│   ├── handshake/
│   │   ├── en_sentiment_{YYYY-MM-DD}.csv
│   │   ├── hi_sentiment_{YYYY-MM-DD}.csv
│   │   └── fused_{YYYY-MM-DD}.csv
│   ├── trust_weights/
│   │   └── {YYYY-MM-DD}.json
│   └── predictions/
│       └── predictions_{YYYY-MM-DD}.csv
├── models/
│   └── checkpoints/
│       └── {stage}/
│           └── model_{YYYY-MM-DD}.pt
├── src/
├── environment.yml
├── config.yml
├── PROJECT_LOG.md
└── DATA_NOTES.md
```

---

*Last updated: 2026-03-02 | Blueprint v1.0*
