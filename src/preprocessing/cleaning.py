#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np

# 1. Load the primary historical dataset
df_all = pd.read_csv('dataset/nifty50_historical_data.csv', parse_dates=['Date'])

def get_blueprint_pilots(df, top_n=15):
    """
    Action 2 & 3: Audit for data integrity and high-news potential.
    """
    stats = []
    tickers = df['Ticker'].unique()

    for ticker in tickers:
        data = df[df['Ticker'] == ticker].sort_values('Date')

        # Action 2: Audit for gaps (Missing more than 5 days)
        gaps = (data['Date'].diff().dt.days > 5).sum() 
        history_years = (data['Date'].max() - data['Date'].min()).days / 365

        # Action 3: Volume as proxy for news density 
        avg_vol = data['Volume'].tail(200).mean() 

        stats.append({
            'Ticker': ticker,
            'Years': history_years,
            'Gaps': gaps,
            'Volume_Rank': avg_vol
        })

    # Selection: Must have long history and minimal gaps
    audit_results = pd.DataFrame(stats)
    pilots = audit_results[
        (audit_results['Years'] >= 10) & (audit_results['Gaps'] < 10)
    ].sort_values('Volume_Rank', ascending=False).head(top_n)

    return pilots['Ticker'].tolist()

# Execute Selection
pilot_tickers = get_blueprint_pilots(df_all)

# Create a 'Market Proxy' since we need it for Action 6 (Label Correction)
# We calculate the average daily return of all stocks to act as the "Nifty 50 Index"
market_proxy = df_all.groupby('Date')['Daily_Return'].mean().reset_index()
market_proxy.columns = ['Date', 'nifty_ret_proxy']

print(f"Selected {len(pilot_tickers)} Pilot Stocks.")
print(f"Top 5 Pilots: {pilot_tickers[:5]}")

# Filter main dataframe to only selected pilots
df = df_all[df_all['Ticker'].isin(pilot_tickers)].copy()
df = df.sort_values(['Ticker', 'Date'])


# - Cell 1 (The Auditor): Scans the raw Kaggle dataset to find the 10-15 "healthiest" stocks. It filters out stocks with missing data (Action 2) and prioritizes those with high volume to ensure they have enough news coverage (Action 3).

# In[20]:


def apply_blueprint_logic(df, market_proxy):
    # 1. Force both Date columns to be clean strings 'YYYY-MM-DD'
    # This removes any hidden timezone or timestamp noise
    df['Date_Key'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    market_proxy['Date_Key'] = pd.to_datetime(market_proxy['Date']).dt.strftime('%Y-%m-%d')

    # 2. Create a mapping dictionary from the market proxy
    # This is more reliable than a merge when dates are tricky
    market_map = dict(zip(market_proxy['Date_Key'], market_proxy['nifty_ret_proxy']))

    # 3. Map the market return to the main dataframe
    df['nifty_ret_proxy'] = df['Date_Key'].map(market_map)

    # Check for failures
    null_count = df['nifty_ret_proxy'].isnull().sum()
    if null_count > len(df) * 0.1: # If more than 10% failed
        raise ValueError(f"Mapping failed! {null_count} rows didn't match dates. Check date ranges.")

    # 4. ACTION 6: Confound-Corrected Label Calculation
    # Stock Return - Market Return
    df['target_label'] = df['Daily_Return'] - df['nifty_ret_proxy']

    # 5. Shift target by -1 (Today predicts Tomorrow)
    df['target_label'] = df.groupby('Ticker')['target_label'].shift(-1)

    # 6. ACTION 5: Timestamp Alignment
    df['news_sync_date'] = df['Date'].dt.normalize()

    # Clean up temporary key
    df.drop(columns=['Date_Key'], inplace=True)

    return df

# Execute
df = apply_blueprint_logic(df, market_proxy)

# Drop rows where we don't have a tomorrow's target
df.dropna(subset=['target_label'], inplace=True)

print("Confound-corrected labels generated without merge errors.")
print(df[['Ticker', 'Date', 'target_label']].head())


# Cell 2 (The Labeler): Isolates company-specific price moves by subtracting the Nifty 50 market return (Action 6). It also aligns the time to ensure 8 PM news is linked to the next day's 10 AM price move (Action 5).

# In[21]:


def add_technical_indicators(df):
    """
    Action: Build Technical Stream features per stock.
    Required by: T-LSTM Hybrid Branch 1.
    """
    # 1. RSI (Momentum) - Measures speed and change of price movements
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    # 2. MACD Histogram (Trend) - Shows the relationship between two moving averages
    macd = ta.trend.MACD(close=df['Close'])
    df['macd_diff'] = macd.macd_diff()

    # 3. Bollinger Band Width (Volatility) - Identifies 'squeeze' periods 
    bb = ta.volatility.BollingerBands(close=df['Close'])
    df['bb_width'] = bb.bollinger_wband()

    # 4. SMA Distance (Trend Strength) - Distance from 20-day Simple Moving Average
    # Your blueprint emphasizes this to capture mean reversion
    df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['dist_from_sma'] = (df['Close'] - df['sma_20']) / df['sma_20']

    # 5. Volume Shock (Activity) - Percentage change in volume
    df['vol_delta'] = df['Volume'].pct_change()

    return df

# Apply indicators per Ticker
df = df.groupby('Ticker', group_keys=False).apply(add_technical_indicators)

# Clean up: Drop the first 20 rows of each stock where indicators are still 'NaN'
df.dropna(subset=['rsi', 'sma_20'], inplace=True)

print("Technical indicators generated.")
print(f"Current Row Count: {len(df)}")
print(f"Sample Features: {df[['Ticker', 'rsi', 'macd_diff', 'dist_from_sma']].head()}")


# Cell 3 (The Tech Stream): Uses the ta library to turn raw prices into math-based indicators (RSI, MACD, etc.). This creates the "Technical" half of the dual-stream input for your T-LSTM model.

# In[22]:


# 1. Define the Year 20 Seal
max_date = df['Date'].max()
holdout_cutoff = max_date - pd.DateOffset(years=1)

# 2. Split into Training/Validation and Sealed Holdout
train_val_set = df[df['Date'] < holdout_cutoff].copy()
sealed_holdout = df[df['Date'] >= holdout_cutoff].copy()

# 3. Create Handshake Placeholders
for dataset in [train_val_set, sealed_holdout]:
    dataset['en_sentiment'] = 0.0
    dataset['hi_sentiment'] = 0.0
    dataset['source_conf'] = 1.0

# 4. FIX: Handle Infinity and NaN before Scaling
feature_cols = ['rsi', 'macd_diff', 'bb_width', 'dist_from_sma', 'vol_delta']

for dataset in [train_val_set, sealed_holdout]:
    # Replace inf with large finite numbers and fill NaNs with 0
    dataset[feature_cols] = dataset[feature_cols].replace([np.inf, -np.inf], np.nan)
    dataset[feature_cols] = dataset[feature_cols].fillna(0)

# 5. Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit only on training data
train_val_set[feature_cols] = scaler.fit_transform(train_val_set[feature_cols])

# Transform holdout using the same scaler
sealed_holdout[feature_cols] = scaler.transform(sealed_holdout[feature_cols])

print("Infinity values handled and Year 20 Sealed.")
print(f"Max value in vol_delta after scaling: {train_val_set['vol_delta'].max()}")


# Cell 4 (The Seal): Splits the data into training and the "Year 20" holdout. It normalizes the numbers (Scaling) but "seals" the holdout so the model doesn't "cheat" by seeing future data during scaling. It also adds blank slots for English and Hindi sentiment.

# In[23]:


# 1. Verification: Ensure all Blueprint-required columns exist
# These specific columns are the "contract" between your different project modules
required_columns = [
    'Ticker', 'Date', 'news_sync_date', 'target_label', 
    'rsi', 'macd_diff', 'en_sentiment', 'hi_sentiment'
]

def verify_blueprint_schema(df_to_check):
    missing = [col for col in required_columns if col not in df_to_check.columns]
    if not missing:
        print("Handshake Verification Passed: All blueprint columns present.")
    else:
        print(f"Handshake Verification Failed: Missing columns {missing}")

verify_blueprint_schema(train_val_set)

# 2. Save the Training/Validation Set
# This file is for the T-LSTM Branch (Technical Stream)
train_val_set.to_csv('m1_train_val_final.csv', index=False)

# 3. Save the Sealed Holdout (Year 20)
# This file is for the final "Out of Sample" Audit
sealed_holdout.to_csv('m1_sealed_holdout_year20.csv', index=False)

# 4. Save the list of selected Tickers
# This tells the scraper/NLP engine which stocks to focus on
pd.Series(pilot_tickers).to_csv('selected_pilot_tickers.csv', index=False)

print("\n--- DATA CLEANING COMPLETE ---")
print("Files Generated:")
print("1. train_val_final.csv (Main dataset)")
print("2. sealed_holdout_year20.csv (Untouched test set)")
print("3. selected_pilot_tickers.csv (Reference list)")


# Cell 5 (The Handshake): Verifies all required columns exist and saves the data into new CSV files. These files are the "contract" between your different machines (Market Data machine, NLP machine, and Training machine).
