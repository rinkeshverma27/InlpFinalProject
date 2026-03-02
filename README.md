# News-Driven Financial Prediction Engine
**Nifty 50 | Multilingual NLP (Hindi + English) | T-LSTM Hybrid | 3-Tier Adaptive Learning**
> Blueprint v1.0 — March 2026

---

## 📋 Project Overview

This system predicts the **10:00 AM next-day price move** for 10–15 Nifty 50 stocks by fusing multilingual news sentiment (FinBERT English + MuRIL Hindi) with 20 years of OHLCV price data using a **T-LSTM dual-stream architecture**.

Output: `predicted_pct` (e.g. `+1.8% ± 0.4%`) + direction badge (UP/DOWN/NEUTRAL) + daily `predictions.csv` committed to Git.

---

## 🖥️ Hardware Roles

| Machine | GPU | Role |
|---------|-----|------|
| Machine 1 | RTX 5060 (8GB) | T-LSTM training, MuRIL Hindi NLP, EWC nudge, monthly retrain |
| Machine 2 | RTX 3050 (4GB) | FinBERT English inference |
| Machine 3 | RTX 3050 (4GB) | RSS scraping, data serving, cron scheduler |

---

## ⚙️ Setup — All Team Members Must Follow This

### Prerequisites
Install these **system-level** dependencies before creating the conda environment:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libta-lib-dev \
    libssl-dev \
    libffi-dev \
    git \
    curl

# Verify ta-lib system library is present
ldconfig -p | grep libta
```

> **Windows users:** Install TA-Lib from the [prebuilt wheels](https://github.com/cgohlke/talib-build/releases) — do NOT use conda for TA-Lib on Windows.

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-org>/InlpFinalProject.git
cd InlpFinalProject
```

---

### Step 2 — Create the Conda Environment

```bash
# Create from the pinned environment file
conda env create -f environment.yml

# Activate
conda activate nifty-nlp
```

> This will take 5–10 minutes the first time. All versions are pinned — **do not upgrade packages without team discussion**.

---

### Step 3 — Verify Your Environment

```bash
conda activate nifty-nlp
python scripts/verify_env.py
```

You should see:
```
────────────────────────────────────────────────────
  Summary
────────────────────────────────────────────────────

  🎉  X/X checks passed
  Environment is ready. You're good to go!
```

Fix any ❌ FAIL items before proceeding. See the **Troubleshooting** section below.

---

### Step 4 — Configure Your Machine Role

Open `config.yml` and confirm the `stage` and your machine role is correct. **Do not change any other values without team agreement.**

---

## 🔄 Updating the Environment

When a teammate updates `environment.yml` and pushes:

```bash
git pull
conda activate nifty-nlp
conda env update -f environment.yml --prune
python scripts/verify_env.py
```

The `--prune` flag removes packages that are no longer in `environment.yml`.

---

## 📁 Project Structure

```
InlpFinalProject/
├── data/
│   ├── price/            # OHLCV CSVs (gitignored — store on M3 SSD)
│   ├── news/raw/         # Scraped articles JSONL (gitignored)
│   ├── handshake/        # Sentiment CSVs (gitignored)
│   ├── trust_weights/    # ✅ Committed to Git (versioned JSON)
│   └── predictions/      # ✅ Committed to Git (daily CSV)
├── models/
│   └── checkpoints/      # Model weights .pt (gitignored — large files)
├── src/                  # Source code modules
├── scripts/
│   └── verify_env.py     # Environment verification
├── logs/                 # Runtime logs (gitignored)
├── environment.yml       # ✅ Pinned conda environment
├── config.yml            # ✅ Project configuration
├── PROJECT_LOG.md        # ✅ Stage-wise task tracker
├── DATA_NOTES.md         # ✅ Data format reference
├── .gitignore
└── README.md
```

---

## 🚀 Current Stage — Stage 0 (Weeks 1–2)

**7 actions must be completed before writing any model code:**

| # | Action | Owner | Status |
|---|--------|-------|--------|
| S0-1 | Create & test `environment.yml` on all 3 machines | All | 🔄 In Progress |
| S0-2 | Load Kaggle 20yr OHLCV on M3 SSD, audit quality | M3 owner | ⬜ TODO |
| S0-3 | Validate RSS feeds for 3 pilot stocks | M3 owner | ⬜ TODO |
| S0-4 | MuRIL vs IndicBERT eval on 50 Hindi headlines | M1 owner | ⬜ TODO |
| S0-5 | 8PM → 10AM timestamp alignment test (5 trading days) | M3 owner | ⬜ TODO |
| S0-6 | Build confound-corrected label function | M1 owner | ⬜ TODO |
| S0-7 | Train price-only LSTM baseline, record MAE | M1 owner | ⬜ TODO |

Full stage plan: see **[PROJECT_LOG.md](PROJECT_LOG.md)**
Data formats: see **[DATA_NOTES.md](DATA_NOTES.md)**

---

## 📏 Coding Conventions

- **All file paths:** use `pathlib.Path` — **zero hardcoded strings**
- **All train/test splits:** walk-forward only — **never random splits**
- **Year 20 holdout:** never load, never peek — physically sealed until Stage 3
- **GPU memory:** FP16 always on Machine 2, mixed precision on Machine 1 for retrains
- **Commits:** daily auto-commit for `predictions/` and `trust_weights/` via cron

---

## ⚠️ Emergency Freeze

On Union Budget day, election results, surprise RBI decisions, or global market shocks:

```yaml
# config.yml
adaptive_learning:
  FREEZE_ALL_UPDATES: true
```

Push this change immediately. Reset to `false` manually 2–3 trading days after market normalises.

---

## 🛠️ Troubleshooting

### TA-Lib install fails
```bash
# Make sure system library is installed first
sudo apt-get install libta-lib-dev
# Then re-create env
conda env remove -n nifty-nlp
conda env create -f environment.yml
```

### CUDA not detected
```bash
# Check CUDA driver version
nvidia-smi
# Must be >= 12.1 for PyTorch 2.3.1
# If driver is older, update it before creating the environment
```

### PyTorch version mismatch warning
The verify script accepts minor version mismatches as warnings. Only hard ❌ FAIL items need fixing.

### HuggingFace Hub unreachable
Models are downloaded on first use and cached at `~/.cache/huggingface`. Run on a machine with internet access first, then models can be used offline.

---

## 📝 Contributing

1. Pull latest: `git pull origin main`
2. Create branch: `git checkout -b feature/your-task`
3. Work on your task (see PROJECT_LOG.md for current stage tasks)
4. Run verify: `python scripts/verify_env.py`
5. PR to main — at least one team member must review

**Never commit:**
- Raw data files (OHLCV CSVs, news JSONL)
- Model weights (`.pt`, `.pth`, `.safetensors`)
- Secrets or API keys

---

*Blueprint v1.0 | Architecture: COMPLETE | Execution: Stage 0*
