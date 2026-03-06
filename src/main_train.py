import argparse
import sys
from pathlib import Path

# Add src to Python Path
sys.path.append(str(Path(__file__).parent))

import torch
from data import get_train_val_dataloaders
from models.baseline_lstm import BaselineLSTM
from trainer import Trainer
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Nifty 50 Prediction Engine - Training Pipeline")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "hybrid"], help="Model type to train")
    parser.add_argument("--stocks", type=str, default="RELIANCE.NS,TCS.NS,HDFCBANK.NS", help="Comma-separated list of stock tickers to train on")
    parser.add_argument("--data", type=str, default="data1/20year_train.csv", help="Path to training CSV file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    
    args = parser.parse_args()
    
    tickers = [t.strip() for t in args.stocks.split(",")]
    print(f"--- Initialization ---")
    print(f"Targeting Tickers: {tickers}")
    print(f"Model Type: {args.model}")
    print(f"Data Source: {args.data}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device: {device}")
    
    # 1. Load Data
    print("\n--- Loading Data ---")
    
    # Let's peek at the training columns so we know input dimension
    df_sample = pd.read_csv(args.data, nrows=1)
    
    # The feature columns defined in data.py
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'dist_from_sma', 'vol_delta', 'rsi', 'bb_width',
        'Volatility_20D'
    ]
    
    # Calculate exactly how many features are present in this dataset
    available_features = [col for col in feature_cols if col in df_sample.columns]
    input_dim = len(available_features)
    print(f"Detected {input_dim} features for Stream B.")
    
    train_loader, val_loader = get_train_val_dataloaders(args.data, tickers, batch_size=args.batch_size)
    
    if len(train_loader) == 0:
        print("ERROR: Train loader is empty. Maybe Tickers were typed incorrectly?")
        return
        
    ticker_group_name = "pilot_mix" if len(tickers) > 1 else tickers[0]
    
    # 2. Initialize Model
    print(f"\n--- Constructing {args.model.upper()} ---")
    if args.model == "baseline":
        model = BaselineLSTM(input_dim=input_dim, hidden_dim=256, num_layers=2)
        stage = 'Stage0_Baseline'
    else:
        print("ERROR: Hybrid unimplemented in this script.")
        return
        
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 3. Train
    print("\n--- Starting Training Loop ---")
    trainer = Trainer(model, train_loader, val_loader, device, stage=stage)
    
    best_mae = trainer.train(epochs=args.epochs, ticker_group_name=ticker_group_name)
    
    print("\n==================================")
    print(f"✅ Training Complete. Best MAE: {best_mae:.4f}")
    print(f"Benchmark Stage 0 MAE for {args.stocks} is: {best_mae:.4f}")
    print("==================================")
    

if __name__ == "__main__":
    main()
