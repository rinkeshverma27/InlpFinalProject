import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import random
from pathlib import Path

# Ensure src can be imported
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.modeling.lstm_binary import BinaryLSTM
from src.modeling.dataloader import ProductionStockDataset

# ── 3. Training Script ───────────────────────────────────────────────────────
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(42)  # Ensure reproducibility across runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # A. Load Data
    train_path = "data/inputs/prod_train.csv"
    test_path = "data/inputs/prod_test.csv"
    
    os.makedirs("models", exist_ok=True) # Ensure directory exists
    
    print("Loading production datasets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # B. Label Engineering (Binary Classification)
    # Threshold: 0.5% move. We keep ONLY moves > 0.5% or < -0.5% for clean training signal.
    def prepare_binary_df(df, threshold=0.005):
        df = df.copy().sort_values(['ticker', 'Date'])
        # 1 = UP, 0 = DOWN
        df['target_binary'] = (df['target_label'] > 0).astype(int)
        # Filter neutral moves
        df_filtered = df[df['target_label'].abs() > threshold].copy()
        return df_filtered

    df_train_bin = prepare_binary_df(df_train)
    df_test_bin = prepare_binary_df(df_test)
    
    print(f"Train samples after filtering: {len(df_train_bin)}")
    print(f"Test samples after filtering: {len(df_test_bin)}")
    
    # C. Feature Selection
    feature_cols = [
        'Daily_Return', 'Volatility_20D', 'MA_50', 'MA_200', 
        'PE_Ratio', 'Forward_PE', 'Price_to_Book', 'Dividend_Yield', 
        'Beta', 'nifty_ret_proxy', 'rsi', 'macd_diff', 'bb_width', 
        'dist_from_sma', 'vol_delta', 'en_sentiment', 'hi_sentiment'
    ]
    
    # D. Scaling
    scaler = StandardScaler()
    df_train_bin[feature_cols] = scaler.fit_transform(df_train_bin[feature_cols].fillna(0))
    df_test_bin[feature_cols] = scaler.transform(df_test_bin[feature_cols].fillna(0))
    
    # E. Data Loaders
    window_size = 10
    train_loaders = []
    # We group by ticker to avoid overlapping sequences between stocks
    for ticker, group in df_train_bin.groupby('ticker'):
        if len(group) > window_size:
            ds = ProductionStockDataset(group, feature_cols, window_size=window_size)
            train_loaders.append(DataLoader(ds, batch_size=64, shuffle=True))
            
    test_loaders = []
    for ticker, group in df_test_bin.groupby('ticker'):
        if len(group) > window_size:
            ds = ProductionStockDataset(group, feature_cols, window_size=window_size)
            test_loaders.append(DataLoader(ds, batch_size=64, shuffle=False))

    # F. Model, Loss, Optimizer
    model = BinaryLSTM(len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # G. Super Training Run
    epochs = 100
    patience = 15
    best_acc = 0
    trigger_times = 0
    
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for loader in train_loaders:
            for x_seq, y_lab in loader:
                x_seq, y_lab = x_seq.to(device), y_lab.to(device)
                optimizer.zero_grad()
                logits = model(x_seq)
                loss = criterion(logits, y_lab)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Validation
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for loader in test_loaders:
                for x_seq, y_lab in loader:
                    x_seq = x_seq.to(device)
                    logits = model(x_seq)
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_true.extend(y_lab.numpy())
        
        acc = accuracy_score(all_true, all_preds)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loaders):.4f} | Val Acc: {acc:.2%}")
            
        if acc > best_acc:
            print(f"--> Improvement! Best Acc: {acc:.2%}. Saving model...")
            best_acc = acc
            trigger_times = 0
            torch.save(model.state_dict(), "models/prod_binary_lstm_best.pth")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    # H. Final Evaluation
    model.load_state_dict(torch.load("models/prod_binary_lstm_best.pth"))
    model.eval()
    all_preds = []
    all_probs = []
    all_true = []
    with torch.no_grad():
        for loader in test_loaders:
            for x_seq, y_lab in loader:
                x_seq = x_seq.to(device)
                logits = model(x_seq)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_true.extend(y_lab.numpy())
                
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    print(f"\n✅ Production Model Result:")
    print(f"Base Accuracy: {accuracy_score(all_true, all_preds):.2%}")
    
    # High Confidence Filter
    conf_mask = np.max(all_probs, axis=1) > 0.60
    if conf_mask.any():
        conf_acc = accuracy_score(all_true[conf_mask], all_preds[conf_mask])
        print(f"🔥 HIGH CONFIDENCE ACCURACY (p>0.6): {conf_acc:.2%}")
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/prod_scaler.joblib")
    print("Model and Scaler saved.")

if __name__ == "__main__":
    main()
