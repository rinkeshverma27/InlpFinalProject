import pandas as pd
import pathlib

def check_market_baseline(csv_path):
    # Load your Stage 3 output file
    if not pathlib.Path(csv_path).exists():
        return f"Error: File {csv_path} not found."
        
    df = pd.read_csv(csv_path)
    
    # Filter for your February 2025 test month
    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
        
    test_data = df[(df['date'].dt.year == 2025) & (df['date'].dt.month == 2)]
    month_name = "February 2025"
    
    if test_data.empty:
        # Fallback to latest available month
        max_date = df['date'].max()
        test_data = df[(df['date'].dt.year == max_date.year) & (df['date'].dt.month == max_date.month)]
        month_name = max_date.strftime("%B %Y")
        print(f"⚠️ Note: Feb 2025 empty. Falling back to {month_name}.")
    
    if test_data.empty:
        return "Error: No valid data found in the provided CSV."
    
    # Calculate Up vs Down distribution based on actual market moves
    total_days = len(test_data)
    up_days = (test_data['actual_pct'] > 0).sum()
    down_days = total_days - up_days
    
    baseline_up = (up_days / total_days) * 100
    baseline_down = (down_days / total_days) * 100
    
    # The "Lazy Model" baseline is simply guessing the majority class every day
    majority_class_baseline = max(baseline_up, baseline_down)
    
    print(f"File Analyzed: {csv_path}")
    print(f"Total Trading Records in {month_name}: {total_days}")
    print(f"Natural UP Bias: {baseline_up:.2f}%")
    print(f"Natural DOWN Bias: {baseline_down:.2f}%")
    print("-" * 30)
    
    # Compare against your top model accuracy (57.14%)
    if 57.14 > majority_class_baseline:
        print(f"✅ EDGE CONFIRMED: 57.14% beats the natural baseline of {majority_class_baseline:.2f}%.")
    else:
        print(f"🔴 NO EDGE: The model underperformed the natural market bias of {majority_class_baseline:.2f}%.")

if __name__ == "__main__":
    # Point to the most recent prediction CSV from the batch run
    prediction_csv = "data/predictions/predictions_2026-03-07.csv"
    check_market_baseline(prediction_csv)
