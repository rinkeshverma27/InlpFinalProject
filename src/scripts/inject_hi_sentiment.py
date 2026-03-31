"""
inject_hi_sentiment.py
Injects MuRIL Hindi sentiment scores into prod_train.csv and prod_test.csv.

Reads the sentiment output from analyzer_hi.py, aggregates daily scores
per ticker, and fills the `hi_sentiment` column in the production datasets.

Usage:
    python src/scripts/inject_hi_sentiment.py \
        --sentiment data/hindi_news/hi_sentiment.csv \
        --train     data/inputs/prod_train.csv \
        --test      data/inputs/prod_test.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def load_daily_sentiment(sentiment_path: str) -> pd.DataFrame:
    """Load sentiment CSV and aggregate to daily mean per ticker."""
    df = pd.read_csv(sentiment_path)

    # Extract date from timestamp
    df['date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')

    # Confidence-weighted average: weight each score by model confidence
    df['weighted_score'] = df['sentiment_score'] * df['model_confidence']

    daily = df.groupby(['date', 'ticker']).agg(
        weighted_sum=('weighted_score', 'sum'),
        conf_sum=('model_confidence', 'sum'),
        raw_mean=('sentiment_score', 'mean'),
        article_count=('sentiment_score', 'count'),
    ).reset_index()

    # Confidence-weighted mean
    daily['hi_sentiment_new'] = (daily['weighted_sum'] / daily['conf_sum']).round(4)

    # Where confidence sum is 0 (shouldn't happen), fall back to raw mean
    zero_conf = daily['conf_sum'] == 0
    daily.loc[zero_conf, 'hi_sentiment_new'] = daily.loc[zero_conf, 'raw_mean'].round(4)

    print(f"Aggregated sentiment: {len(daily)} (date, ticker) pairs")
    print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"  Tickers: {sorted(daily['ticker'].unique())}")

    return daily[['date', 'ticker', 'hi_sentiment_new', 'article_count']]


def inject_into_dataset(dataset_path: str, daily_sentiment: pd.DataFrame, label: str):
    """Inject hi_sentiment scores into a production CSV."""
    path = Path(dataset_path)
    if not path.exists():
        print(f"WARNING: {path} not found, skipping.")
        return

    print(f"\nProcessing {label}: {path}")
    df = pd.read_csv(path)
    original_len = len(df)

    # Verify hi_sentiment column exists
    if 'hi_sentiment' not in df.columns:
        print(f"  ERROR: 'hi_sentiment' column not found in {path}")
        print(f"  Available columns: {list(df.columns)}")
        return

    # Extract date string for matching
    df['_date_key'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # Build a lookup dict for fast matching: (date, ticker) -> score
    lookup = {}
    for _, row in daily_sentiment.iterrows():
        lookup[(row['date'], row['ticker'])] = row['hi_sentiment_new']

    # Inject scores
    updated_count = 0
    for idx in df.index:
        key = (df.at[idx, '_date_key'], df.at[idx, 'ticker'])
        if key in lookup:
            df.at[idx, 'hi_sentiment'] = lookup[key]
            updated_count += 1

    # Remove temp column
    df.drop(columns=['_date_key'], inplace=True)

    # Verify structure hasn't changed
    assert len(df) == original_len, "Row count changed - something went wrong!"

    # Save back
    df.to_csv(path, index=False)

    # Stats
    total = len(df)
    nonzero = (df['hi_sentiment'] != 0.0).sum()
    print(f"  Total rows: {total}")
    print(f"  Updated with Hindi sentiment: {updated_count}")
    print(f"  Non-zero hi_sentiment values: {nonzero}")
    print(f"  hi_sentiment range: [{df['hi_sentiment'].min():.4f}, {df['hi_sentiment'].max():.4f}]")
    print(f"  Saved back to {path}")


def main():
    parser = argparse.ArgumentParser(description="Inject Hindi sentiment into production CSVs")
    parser.add_argument("--sentiment", type=str,
                        default="data/hindi_news/hi_sentiment.csv",
                        help="Path to Hindi sentiment output CSV")
    parser.add_argument("--train", type=str,
                        default="data/inputs/prod_train.csv",
                        help="Path to production training CSV")
    parser.add_argument("--test", type=str,
                        default="data/inputs/prod_test.csv",
                        help="Path to production test CSV")
    args = parser.parse_args()

    print("=" * 70)
    print("Hindi Sentiment Injection Pipeline")
    print("=" * 70)

    # Load and aggregate sentiment scores
    daily_sentiment = load_daily_sentiment(args.sentiment)

    # Inject into both datasets
    inject_into_dataset(args.train, daily_sentiment, "TRAIN")
    inject_into_dataset(args.test, daily_sentiment, "TEST")

    print("\n" + "=" * 70)
    print("Injection complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
