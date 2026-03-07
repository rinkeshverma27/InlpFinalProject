import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import joblib
from pathlib import Path

# Ensure src can be imported
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.modeling.lstm_binary import BinaryLSTM

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    model_path = "models/prod_binary_lstm_best.pth"
    scaler_path = "models/prod_scaler.joblib"
    test_data_path = "data/inputs/prod_test.csv"
    
    if not os.path.exists(model_path):
        print("Error: Model not found.")
        return

    # Load Assets
    scaler = joblib.load(scaler_path)
    feature_cols = [
        'Daily_Return', 'Volatility_20D', 'MA_50', 'MA_200', 
        'PE_Ratio', 'Forward_PE', 'Price_to_Book', 'Dividend_Yield', 
        'Beta', 'nifty_ret_proxy', 'rsi', 'macd_diff', 'bb_width', 
        'dist_from_sma', 'vol_delta', 'en_sentiment', 'hi_sentiment'
    ]
    
    model = BinaryLSTM(len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

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
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            
            direction = "UP" if pred_class == 1 else "DOWN"
            
            results.append({
                'Ticker': ticker,
                'Date': group['Date'].iloc[-1],
                'Prediction': direction,
                'Confidence': f"{confidence:.2%}",
                'Probability_UP': f"{probs[1]:.2%}",
                'Probability_DOWN': f"{probs[0]:.2%}",
                'Significant_Signal': "YES" if confidence > 0.60 else "NO"
            })
            
    # Save Results
    final_df = pd.DataFrame(results)
    output_path = "data/predictions/production_predictions.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.sort_values('Confidence', ascending=False).to_csv(output_path, index=False)
    
    print(f"\n✅ Production Predictions generated at {output_path}")
    print(final_df)

if __name__ == "__main__":
    main()
