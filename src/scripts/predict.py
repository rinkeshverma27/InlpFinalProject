import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import sys
import joblib
from pathlib import Path

# Ensure src can be imported
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.modeling.lstm_binary import BinaryLSTM

def parse_args():
    parser = argparse.ArgumentParser(description="Run Binary LSTM prediction")
    parser.add_argument("--model-path", default="models/prod_binary_lstm_best.pth")
    parser.add_argument("--scaler-path", default="models/prod_scaler.joblib")
    parser.add_argument("--test-path", default="data/inputs/prod_test.csv")
    parser.add_argument("--output-path", default="data/predictions/production_predictions.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    model_path = args.model_path
    scaler_path = args.scaler_path
    test_data_path = args.test_path
    output_path = args.output_path
    
    if not os.path.exists(model_path):
        print("Error: Model not found.")
        raise SystemExit(1)
    if not os.path.exists(scaler_path):
        print("Error: Scaler not found.")
        raise SystemExit(1)
    if not os.path.exists(test_data_path):
        print("Error: Test data not found.")
        raise SystemExit(1)

    # Load Assets
    scaler = joblib.load(scaler_path)
    feature_cols = [
        'Daily_Return', 'Volatility_20D', 'MA_50', 'MA_200', 
        'PE_Ratio', 'Forward_PE', 'Price_to_Book', 'Dividend_Yield', 
        'Beta', 'nifty_ret_proxy', 'rsi', 'macd_diff', 'bb_width', 
        'dist_from_sma', 'vol_delta', 'en_sentiment', 'hi_sentiment'
    ]
    
    model = BinaryLSTM(len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # ── MC Dropout Requirement ──
    # Keep the model in train mode to leave Dropout layers ACTIVE during inference
    model.train()

    # Load Data
    df = pd.read_csv(test_data_path)
    df = df.sort_values(['ticker', 'Date'])
    
    results = []
    window_size = 10
    
    for ticker, group in df.groupby('ticker'):
        if len(group) < window_size: continue
        
        # Scale
        X_scaled = scaler.transform(group[feature_cols].fillna(0))
        # Take the last window for prediction
        last_window = X_scaled[-window_size:]
        
        with torch.no_grad():
            x_tensor = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)
            
            # ── 50-Pass MC Dropout Inference ──
            mc_passes = 50
            mc_probs = []
            
            for _ in range(mc_passes):
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                mc_probs.append(probs)
                
            # Calculate Mean and Variance
            mc_probs = np.array(mc_probs)
            mean_probs = mc_probs.mean(axis=0)       # Mean confidence
            std_probs = mc_probs.std(axis=0)         # Variance / Uncertainty Risk
            
            pred_class = np.argmax(mean_probs)
            confidence = mean_probs[pred_class]
            uncertainty = std_probs[pred_class]
            
            direction = "UP" if pred_class == 1 else "DOWN"
            
            # Blueprint: Significant signal = high confidence + low MC uncertainty
            # Aligns with production_predictions.csv schema (YES / NO column)
            is_significant = (uncertainty < 0.10 and confidence > 0.55)
            
            results.append({
                'Ticker': ticker,
                'Date': group['Date'].iloc[-1],
                'Prediction': direction,
                'Confidence': f"{confidence:.2%}",
                'Probability_UP': f"{mean_probs[1]:.2%}",
                'Probability_DOWN': f"{mean_probs[0]:.2%}",
                'Significant_Signal': "YES" if is_significant else "NO"
            })
            
    # Save Results
    final_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Production Predictions generated at {output_path}")
    print(final_df)

if __name__ == "__main__":
    main()
