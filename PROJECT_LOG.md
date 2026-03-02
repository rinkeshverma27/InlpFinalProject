# 📋 PROJECT LOG — News-Driven Financial Prediction Engine
**Blueprint v1.0 | March 2026 | Status: PLANNING COMPLETE → EXECUTION STARTING**

---

## 🎯 Project Objective

Predict the **10:00 AM next-day price move** (as a regression float + 95% CI) for **10–15 Nifty 50 stocks** by fusing:
- **Multilingual news sentiment** (Hindi + English via FinBERT / MuRIL)
- **20 years of historical OHLCV price data** (Kaggle dataset)

Output: `predicted_pct` (e.g. `+1.8% ± 0.4%`) + direction badge + daily `predictions.csv` committed to Git.

---

## 🏗️ Architecture Overview

```
Stream A (NLP, nightly)         Stream B (Price, Machine 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━      ━━━━━━━━━━━━━━━━━━━━━━━━━━
RSS Scrape (M3, 8 PM)           Kaggle OHLCV 20yr dataset
  ↓                               ↓
FinBERT EN scoring (M2)         LSTM (60/30/10-day dynamic window)
  +                               ↓
MuRIL HI scoring (M1)           256-dim hidden state
  ↓                                       ↘
Trust-weight Fusion (M1) ──→ 512-dim embed ──→ Concat (768-dim)
                                            ↓
                                    FC 768→128 (ReLU, Dropout 0.3)
                                    FC 128→1  (Linear)
                                    Huber Loss (δ=0.01)
                                    MC Dropout → 95% CI
```

---

## 🖥️ Hardware Allocation

| Machine | Specs | VRAM Budget | Role |
|---------|-------|------------|------|
| **Machine 1** | RTX 5060, 8GB VRAM, 32GB RAM | ~6.5 / 8 GB | MuRIL Hindi NLP, T-LSTM training, EWC nudge, monthly retrain, Fisher matrix |
| **Machine 2** | RTX 3050, 4GB VRAM, 16GB RAM | ~3.2 / 4 GB | FinBERT English inference (FP16, batch=16), EN sentiment CSV |
| **Machine 3** | RTX 3050, 4GB VRAM, 8GB RAM | ~0.5 / 4 GB | RSS scraping, dedup, timestamp alignment, Kaggle OHLCV serving, cron scheduler |

---

## 📅 Staged Roadmap — Full Stage-wise Breakdown

---

### ⬜ STAGE 0 — Foundation & Environment Setup
**Timeline:** Weeks 1–2
**Exit Criteria:** All 3 machines cold-start identically from `environment.yml`

#### Tasks
| # | Task | Machine | Priority | Status |
|---|------|---------|---------|--------|
| S0-1 | Create `environment.yml` with pinned versions: `torch`, `transformers`, `pandas`, `ta-lib`, `yfinance`. Test cold-start on all 3 machines. | All | P0 | ⬜ TODO |
| S0-2 | Load Kaggle 20-year Nifty 50 OHLCV dataset on Machine 3 SSD. Audit for data gaps, missing columns, split errors. | M3 | P0 | ⬜ TODO |
| S0-3 | Validate RSS feed coverage — confirm 2+ Hindi and 2+ English sources for each of the 3 pilot stocks. | M3 | P0 | ⬜ TODO |
| S0-4 | Evaluate MuRIL vs IndicBERT on 50 sample financial Hindi headlines. Record which handles Hinglish code-switching better. | M1 | P0 | ⬜ TODO |
| S0-5 | Test 8 PM to 10 AM timestamp alignment on 5 real trading days (include 1 market holiday). | M3 | P0 | ⬜ TODO |
| S0-6 | Build confound-corrected label function. Spot-check on 10 known high-news days from Kaggle historical data. | M1 | P0 | ⬜ TODO |
| S0-7 | Train price-only LSTM baseline on 3 pilot stocks. Record MAE. THIS NUMBER IS THE PERMANENT BENCHMARK. | M1 | P0 | ⬜ TODO |

#### Requirements from Previous Stage
- None (this is the first stage)

#### Deliverables for Next Stage
- `environment.yml` (pinned, tested on all 3 machines)
- Kaggle dataset loaded and audited on M3 SSD
- RSS feed validation report (per stock, per source)
- MuRIL vs IndicBERT eval result — winner chosen
- Timestamp alignment test results
- `label.py` — confound-corrected label function
- **Baseline MAE number recorded** (permanent benchmark)

---

### ⬜ STAGE 1 — Pilot: 3 Stocks, Baseline + Hindi NLP
**Timeline:** Month 1
**Exit Criteria:** Baseline MAE recorded. Hindi NLP model winner confirmed.

#### Tasks
| # | Task | Machine | Priority | Status |
|---|------|---------|---------|--------|
| S1-1 | Select 3 pilot stocks from Nifty 50 (recommend: 1 large-cap high-news, 1 mid-cap, 1 Hindi-heavy coverage) | Team | P0 | ⬜ TODO |
| S1-2 | Run price-only LSTM on pilot stocks using walk-forward CV (Years 1-17 train, Years 18-19 val) | M1 | P0 | ⬜ TODO |
| S1-3 | Validate label construction: adjusted_ret = raw_ret - nifty50_overnight_ret on pilot stocks | M1 | P0 | ⬜ TODO |
| S1-4 | Fine-tune winning Hindi NLP model on 500 financial Hindi samples if needed | M1 | P1 | ⬜ TODO |
| S1-5 | Set up predictions.csv schema and daily Git commit workflow | M3 | P1 | ⬜ TODO |
| S1-6 | Document Baseline MAE per pilot stock in evaluation log | Team | P0 | ⬜ TODO |

#### Requirements FROM Stage 0
- environment.yml working on all machines
- Kaggle OHLCV dataset clean and on M3
- Hindi NLP model chosen (MuRIL or IndicBERT)
- Label function validated
- RSS feeds confirmed for 3 pilot stocks

#### Deliverables for Next Stage
- Price-only baseline MAE per pilot stock (the 5% improvement gate for Stage 2)
- Hindi NLP model chosen and optionally fine-tuned
- predictions.csv schema implemented
- Walk-forward CV logic tested and audited

---

### ⬜ STAGE 2A — Full T-LSTM + Hybrid on 3 Pilot Stocks
**Timeline:** Months 2–4
**Exit Criteria:** Hybrid model beats price-only baseline MAE by >= 5% on pilot stocks

#### Tasks
| # | Task | Machine | Priority | Status |
|---|------|---------|---------|--------|
| S2A-1 | Build full T-LSTM dual-stream model (Stream A NLP + Stream B LSTM + Fusion) | M1 | P0 | ⬜ TODO |
| S2A-2 | Implement CSV Handshake pipeline: RSS → FinBERT → MuRIL → trust-weight fusion → handshake CSV | M1/M2/M3 | P0 | ⬜ TODO |
| S2A-3 | Implement FinBERT English scoring with FP16, batch=16 on Machine 2 | M2 | P0 | ⬜ TODO |
| S2A-4 | Implement MuRIL Hindi scoring on Machine 1 (parallel to FinBERT) | M1 | P0 | ⬜ TODO |
| S2A-5 | Implement Tier 1 Daily Source Trust Weight Update (10:10 AM, trust_weights/{date}.json) | M1 | P1 | ⬜ TODO |
| S2A-6 | Implement Hindi sparse rule: if < 2 Hindi articles in past 3 days → Hindi weight = 0.0 | M1 | P1 | ⬜ TODO |
| S2A-7 | Implement MC Dropout (50 forward passes) for 95% confidence interval | M1 | P1 | ⬜ TODO |
| S2A-8 | Run hybrid model on 3 pilot stocks. Compare MAE vs baseline. | M1 | P0 | ⬜ TODO |
| S2A-9 | Implement dynamic LSTM window sizing based on realised volatility | M1 | P1 | ⬜ TODO |

#### Requirements FROM Stage 1
- Baseline MAE per pilot stock (the gate target)
- Hindi NLP model ready
- Label function validated + predictions.csv schema ready
- Walk-forward CV logic audited — NO data leakage

#### Deliverables for Next Stage
- Full T-LSTM hybrid model running on 3 pilot stocks
- CSV Handshake pipeline operational
- Tier 1 daily trust weight system live
- Gate check: hybrid MAE >= 5% better than baseline

---

### ⬜ STAGE 2B — Scale to All 15 Stocks + Tier 2 EWC
**Timeline:** Months 5–8
**Exit Criteria:** All 15 stocks live. 30-day MAE slope is negative.

#### Tasks
| # | Task | Machine | Priority | Status |
|---|------|---------|---------|--------|
| S2B-1 | Expand data pipeline and model to all 15 Nifty 50 stocks | All | P0 | ⬜ TODO |
| S2B-2 | Implement Tier 2 Weekly EWC Weight Nudge (Friday 11 PM, LR=1e-5, 3 epochs, EWC λ=400) | M1 | P0 | ⬜ TODO |
| S2B-3 | Compute Fisher Information Matrix after Stage 2A training (reused for EWC nudges) | M1 | P0 | ⬜ TODO |
| S2B-4 | Implement EWC rollback gate (+2% MAE tolerance) | M1 | P1 | ⬜ TODO |
| S2B-5 | Implement Market Regime Detector (normal / moderate / event) | M1 | P1 | ⬜ TODO |
| S2B-6 | Build Team Dashboard v1 (daily view: predicted move, CI bar, direction badge, actual vs predicted) | Team | P2 | ⬜ TODO |
| S2B-7 | Validate diversity guard: no single source > 40% of total trust weight | M1 | P1 | ⬜ TODO |
| S2B-8 | Validate stale decay: sources inactive 7+ days decay to 1.0 at 2%/day | M1 | P1 | ⬜ TODO |
| S2B-9 | Implement anomaly guard: skip Tier 1 update if |actual_move| > 3σ historical std | M1 | P1 | ⬜ TODO |

#### Requirements FROM Stage 2A
- Hybrid model working, gate passed (>=5% better than baseline)
- CSV Handshake pipeline stable
- Tier 1 trust weight system tested
- Fisher Information Matrix available

#### Deliverables for Next Stage
- All 15 stocks live in production pipeline
- Tier 2 EWC weekly nudge operational with rollback
- Dashboard v1 live for team monitoring
- 30-day MAE slope confirmed negative

---

### ⬜ STAGE 2C — Full Automation + Dashboard v2
**Timeline:** Months 9–12
**Exit Criteria:** Fully automated. No daily manual intervention required.

#### Tasks
| # | Task | Machine | Priority | Status |
|---|------|---------|---------|--------|
| S2C-1 | Implement Tier 3 Monthly Full Walk-Forward Retrain (last Sunday of month, ~4-6 hrs) | M1 | P0 | ⬜ TODO |
| S2C-2 | Implement deployment gate: new MAE must beat live MAE by >= 2% to deploy | M1 | P0 | ⬜ TODO |
| S2C-3 | Recompute Fisher Information Matrix post-each monthly retrain | M1 | P0 | ⬜ TODO |
| S2C-4 | Implement FREEZE_ALL_UPDATES emergency flag in config.yml | M1 | P1 | ⬜ TODO |
| S2C-5 | Build Full Cron Scheduler on Machine 3 (8PM RSS, 8:30PM NLP, 9PM fusion, 9:05PM inference, 10:10AM Tier1, Friday 11PM Tier2, monthly Sunday Tier3) | M3 | P0 | ⬜ TODO |
| S2C-6 | Build Dashboard v2 (trend view: 7-day/30-day MAE, slope chart, trust weight history, nudge win/rollback rate, monthly deploy log) | Team | P2 | ⬜ TODO |
| S2C-7 | Audit all train/test split calls — zero random splits allowed, walk-forward only | Team | P0 | ⬜ TODO |
| S2C-8 | Implement predictions.csv auto-fill at 10:05 AM (actual_pct + gap columns) | M3 | P1 | ⬜ TODO |

#### Requirements FROM Stage 2B
- All 15 stocks live
- Tier 2 EWC stable
- Regime detector working
- Dashboard v1 in use

#### Deliverables for Stage 3
- Fully automated pipeline (no manual intervention)
- All 3 tiers operational
- Emergency freeze mechanism available
- Dashboard v2 with trend analysis

---

### ⬜ STAGE 3 — Final Evaluation & Research Documentation
**Timeline:** Year 2–3
**Exit Criteria:** Final MAE vs sealed Year 20 holdout reported.

#### Tasks
| # | Task | Machine | Priority | Status |
|---|------|---------|---------|--------|
| S3-1 | Unseal Year 20 holdout dataset (FIRST TIME — irreversible) | M3 | P0 | ⬜ TODO |
| S3-2 | Run final model evaluation on Year 20 holdout. Report MAE, Direction Accuracy, High-Vol Accuracy | M1 | P0 | ⬜ TODO |
| S3-3 | Compare hybrid vs price-only baseline on holdout | M1 | P0 | ⬜ TODO |
| S3-4 | Document research findings, architecture decisions, and ablation studies | Team | P1 | ⬜ TODO |
| S3-5 | Write academic paper / final project report | Team | P2 | ⬜ TODO |

#### Requirements FROM Stage 2C
- Fully automated system running for 6+ months
- Year 20 holdout NEVER touched during all prior stages
- All model versions + trust weights committed to Git (reproducible)

---

## 📊 Evaluation Metrics & Success Criteria

| Metric | Type | Definition | Stage 1 Target | Stage 2 Target |
|--------|------|-----------|---------------|---------------|
| **MAE** (PRIMARY KPI) | Regression | Mean |pred_pct - actual_pct| | < 1.5% | < 0.8% |
| **Direction Accuracy** | Classification | sign(pred) == sign(actual) | > 52% | > 57% |
| **High-Vol Accuracy** | Classification | Dir. acc when |move| > 1.5% | > 50% | > 55% |
| **Hybrid vs Baseline** | Comparative | Hybrid MAE vs price-only LSTM | Baseline set | >= 5% better |
| **Trust Entropy** | Health | Shannon entropy of trust weights | Monitor | > 60% max |
| **Nudge Win Rate** | Health | % weekly nudges that improved MAE | Monitor | > 60% |
| **Gap Trend Slope** | Health | Linear slope of 30-day rolling MAE | Monitor | Negative |

**MANDATORY GATE:** Train price-only LSTM baseline in Stage 1. Record MAE. Hybrid MUST beat this by >= 5% to justify the full NLP infrastructure at Stage 2.

---

## ⚠️ Risk Register

| Risk | Level | Mitigation |
|------|-------|-----------|
| Data leakage via random train/test split | CRITICAL | Walk-forward CV only. Audit every split call before training. |
| SGX Nifty overnight confounding | HIGH | Subtract Nifty 50 overnight return from every label. Non-negotiable. |
| Catastrophic forgetting in EWC nudge | HIGH | EWC λ=400. Rollback gate (+2% MAE tolerance). Skip on anomaly weeks. |
| Trust weight collapse to one source | MEDIUM | Hard bounds [0.1, 2.0]. No source > 40% of total weight per stock. |
| IndicBERT poor on Hinglish finance text | MEDIUM | Evaluate MuRIL first. Fine-tune on 500 financial Hindi samples if needed. |
| FinBERT VRAM OOM on Machine 2 | MEDIUM | FP16 + batch=16 + 512-token truncation. CPU offload as fallback. |
| Monthly retrain time grows too long | LOW | Mixed precision + checkpointing. Limit to 8 stocks if retrain > 8 hours. |

---

## 🔗 Stage Dependency Map

```
Stage 0 --> Stage 1 --> Stage 2A --> Stage 2B --> Stage 2C --> Stage 3
  |              |            |             |              |
  |              |            |             |              +-- Full automation
  |              |            |             +---- 15 stocks + EWC
  |              |            +------ Hybrid model live (gate: +5%)
  |              +----------- Baseline MAE (permanent gate number)
  +-------------------------- Environment + data ready (all machines)
```

**Key gates that BLOCK forward progress:**
- S0 → S1: environment.yml working on all 3 machines
- S1 → S2A: Baseline MAE recorded + Hindi NLP winner confirmed
- S2A → S2B: Hybrid >= 5% better than baseline on pilot stocks
- S2B → S2C: All 15 stocks stable, MAE slope negative
- S2C → S3: Year 20 holdout untouched for entire duration

---

## 📅 How We Will Proceed — Execution Plan

### Immediate Actions (This Week = Stage 0)
1. Create `environment.yml` — pin all dependencies, test on M1/M2/M3
2. Audit Kaggle dataset — load on M3 SSD, check gaps and schema
3. RSS validation — verify source coverage for 3 pilot stocks
4. MuRIL vs IndicBERT eval — run on 50 Hindi headlines, pick winner
5. Timestamp alignment test — 5 trading days including 1 holiday
6. Build `label.py` — confound-corrected label, spot-check 10 days
7. Train baseline LSTM — record MAE per pilot stock as permanent benchmark

### Working Conventions
- All file paths via `pathlib` — zero hardcoded strings
- All train/test splits walk-forward only — audit before any training call
- `predictions.csv` committed to Git daily (auto)
- `trust_weights/{date}.json` committed nightly (immutable, versioned)
- Year 20 holdout: treat as physically sealed — do NOT load or peek
- Emergency freeze: `FREEZE_ALL_UPDATES = True` in `config.yml` for budget/election days

---

## 📝 Session Log

| Date | Session | Work Done | Next |
|------|---------|-----------|------|
| 2026-03-02 | Project Init | Blueprint read, project log created, data notes created | Start Stage 0 — S0-1: Create environment.yml |

---

*Last updated: 2026-03-02 | Blueprint v1.0 | Architecture: COMPLETE*
