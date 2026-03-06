#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

train_path = r'data/20year_train.csv'
news_path = r'data/Nifty50_news_data(2020Jan_2024April).csv'

top15_companies_output = r'data/top15_priority_news_companies.csv'
top15_news_output = r'data/top15_news.csv'
top15_stocks_output = r'data/top15_stocks.csv'

train_df = pd.read_csv(train_path)
news_df = pd.read_csv(news_path)

train_df['symbol_key'] = (
    train_df['Ticker']
    .astype(str)
    .str.replace('.NS', '', regex=False)
    .str.upper()
)

news_df['symbol_key'] = (
    news_df['symbol']
    .astype(str)
    .str.upper()
)

train_freq = (
    train_df.groupby(['symbol_key', 'Company_Name'])
    .size()
    .reset_index(name='train_frequency')
    .sort_values(['train_frequency', 'symbol_key'], ascending=[False, True])
)

news_freq = (
    news_df.groupby(['symbol_key', 'company'])
    .size()
    .reset_index(name='news_frequency')
    .sort_values(['news_frequency', 'symbol_key'], ascending=[False, True])
)

selected_companies = (
    news_freq.merge(
        train_freq[['symbol_key', 'Company_Name', 'train_frequency']],
        on='symbol_key',
        how='inner'
    )
    .rename(columns={'company': 'News_Company_Name'})
    .sort_values(
        ['news_frequency', 'train_frequency', 'symbol_key'],
        ascending=[False, False, True]
    )
    .head(15)
    .reset_index(drop=True)
)

selected_companies.insert(0, 'rank', selected_companies.index + 1)

selected_companies.to_csv(top15_companies_output, index=False)

selected_symbols = selected_companies['symbol_key'].tolist()

top15_news_df = news_df[news_df['symbol_key'].isin(selected_symbols)].copy()

top15_news_df.drop(columns=['symbol_key'], inplace=True)

top15_news_df.to_csv(top15_news_output, index=False)

top15_stocks_df = train_df[train_df['symbol_key'].isin(selected_symbols)].copy()

top15_stocks_df.drop(columns=['symbol_key'], inplace=True)

top15_stocks_df.to_csv(top15_stocks_output, index=False)

print("Top 15 companies saved to:", top15_companies_output)
print("Top 15 news dataset saved to:", top15_news_output)
print("Top 15 stock dataset saved to:", top15_stocks_output)

print("\nSelected Companies:")
print(selected_companies[['rank','symbol_key','News_Company_Name','train_frequency','news_frequency']])

print("\nTop15 news rows:", len(top15_news_df))
print("Top15 stock rows:", len(top15_stocks_df))

