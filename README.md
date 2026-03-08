# INLP Final Project - Dual-Stream Market Prediction Model

This repository implements a Transformer-LSTM model to predict Nifty50 and individual component movements by analyzing both numerical OHLCV components alongside FinBERT/MuRIL sentiment scores on financial news.

## Installation
Ensure you have Anaconda or Miniconda installed.
```bash
conda env create -f environment.yml
conda activate nifty-nlp
```

## Running the Pipeline
You can trigger the entire pipeline from raw news filtering down to Model Inference by just running the overarching script:

```bash
python run_pipeline.py --ticker NIFTY50 --epochs 50
```

## Structure
- `data/` contains all subsets of market price actions and raw input data.
- `src/` contains all sequentially ordered modularized python scripts.
- `models/` stores huggingface checkpoints.
