# Dual-Stream Stock Direction Predictor \U0001f4c8\U0001f916

This project is a production-ready, highly resilient machine learning pipeline designed to predict the directional movement (`UP` or `DOWN`) of Nifty-50 stocks. 

It accomplishes this by fusing two distinct data streams into a custom **Dual-Stream LSTM Engine**:
1. **Stream A (NLP / Sentiment)**: Cross-lingual financial news sentiment extracted from English, Hindi, and Hinglish news using FinBERT and a custom fine-tuned MuRIL transformer.
2. **Stream B (Technical)**: Historical price data (OHLCV) with engineered technical indicators (RSI, Bollinger Bands, Moving Averages).

---

## \U0001f6e0\ufe0f Project Requirements & Setup

### 1. Environment Setup
To run the project, you must install the required dependencies inside your Python environment (Conda is recommended).
```bash
# Install all required data science & deep learning libraries
pip install -r requirements.txt
```

### 2. Data Requirements
The pipeline requires raw data to be placed in specific directories. 
- **Stock Historical Data**: Place individual CSV files for each stock inside `data/raw/ohlcv/` (e.g., `RELIANCE.csv`, `HDFCBANK.csv`). The CSV schema must contain: `date, open, high, low, close, volume`.
- **News Data**: Place scraped JSON/CSV news feeds inside `data/raw/news/`.
*(Note: If news data is missing, the pipeline gracefully falls back to Price-Only prediction).*

---

## \u2699\ufe0f System Configuration (Tailor to your Machine)

The heart of the project is the `config.yaml` file. **You do not need to edit any Python code to change how the system runs.** Open `config.yaml` to modify the pipeline according to your machine's capabilities.

### 1. Modifying Memory / GPU Profiles
If you are running on a laptop or a machine with low Video RAM (VRAM), the NLP models might cause "Out of Memory" (OOM) crashes. You can easily fix this by changing the active profile at the very top of `config.yaml`:
```yaml
vram_profile: "8gb"   # Change to "4gb" for laptops, or "full" for heavy GPUs
```
*The `4gb` profile will automatically reduce batch sizes, apply Int8 inference quantization, and shrink the LSTM layers to fit.*

### 2. Changing the Active Stocks
By default, the pipeline runs on a small 5-stock subset for fast testing. To run the full Nifty-50, edit the `active` key in `config.yaml`:
```yaml
tickers:
  active: "subset_5"   # Change to "nifty_10" or "nifty_50"
```

---

## \U0001f680 How to Run the Pipeline

The orchestrator (`main.py`) provides a robust Command Line Interface (CLI). Every command features automatic **caching** (so it doesn't re-run things it already finished) and **OOM back-off** (it safely lowers batch sizes if your computer runs out of memory).

### The "One-Click" Run
To execute the entire pipeline end-to-end (Load Data \u2192 NLP Sentiment \u2192 Extract Features \u2192 Fuse \u2192 Train AI \u2192 Evaluate):
```bash
python main.py run-all
```
*If you ever want to ignore the saved cache and force the AI to process everything from scratch, add the `--force` flag: `python main.py run-all --force`.*

### Step-by-Step Commands
If you want to run specific stages individually, you can use these commands:

1. **`python main.py sentiment`**
   *Loads your raw news data, detects the language, scores it via FinBERT/MuRIL, and outputs a daily directional sentiment score.*

2. **`python main.py features`**
   *Loads your OHLCV data, computes technical indicators, and normalizes them.*

3. **`python main.py fuse`**
   *Combines the Sentiment and Feature streams into dynamic time-series sliding windows (Tensors).*

4. **`python main.py train`**
   *Splits the data based on dates and trains the Dual-Stream LSTM model.*

5. **`python main.py eval --split test`**
   *Generates a massive 7-section JSON performance report detailing accuracy, F1 scores, missing values, and Expected Calibration Error.*

### Utility Commands
- **`python main.py generate-data`**: Generates synthetic financial data to help test the system when real data isn't available.
- **`python main.py finetune-muril`**: Begins the knowledge-distillation process, transferring English financial logic from FinBERT to MuRIL.
