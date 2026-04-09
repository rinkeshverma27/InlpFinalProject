"""
Merge heterogeneous raw Nifty news CSVs into one normalized dataset.

Usage:
    ./venv/bin/python -m src.data.merge_news_csvs
    ./venv/bin/python -m src.data.merge_news_csvs --output data/raw/news/final_news_master.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.paths import RAW_DATA_DIR

DEFAULT_INPUTS = [
    RAW_DATA_DIR / "final_news_merged.csv",
    RAW_DATA_DIR / "preprocessed_nifty_context_part3.csv",
    RAW_DATA_DIR / "Nifty50_news_data(2020Jan_2024April).csv",
]
DEFAULT_OUTPUT = RAW_DATA_DIR / "final_news_master.csv"
OUTPUT_COLUMNS = [
    "datetime",
    "trade_date",
    "ticker",
    "headline",
    "body",
    "source",
    "source_type",
    "lang",
    "url",
    "company",
    "tags",
    "author",
    "source_file",
]


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.strip()
    )


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_convert(None)


def _normalize_final_news_merged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["datetime"] = _to_naive_datetime(df["datetime"])
    df["trade_date"] = df["datetime"].dt.normalize()
    df["body"] = _clean_text(df.get("body", pd.Series(index=df.index, dtype="object")))
    df["headline"] = _clean_text(df.get("headline", pd.Series(index=df.index, dtype="object")))
    df["ticker"] = df.get("ticker", "").fillna("").astype(str).str.upper().str.strip()
    df["source"] = _clean_text(df.get("source", pd.Series(index=df.index, dtype="object")))
    df["source_type"] = _clean_text(df.get("source_type", pd.Series(index=df.index, dtype="object")))
    df["lang"] = _clean_text(df.get("lang", pd.Series(index=df.index, dtype="object"))).replace("", "unknown")
    df["url"] = _clean_text(df.get("url", pd.Series(index=df.index, dtype="object")))
    df["company"] = ""
    df["tags"] = ""
    df["author"] = ""
    df["source_file"] = path.name
    return df


def _normalize_preprocessed_context(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["datetime"] = _to_naive_datetime(df["datetime"])
    df["trade_date"] = _to_naive_datetime(df["trade_date"]).fillna(df["datetime"].dt.normalize())
    df["body"] = _clean_text(df.get("body", pd.Series(index=df.index, dtype="object")))
    df["headline"] = _clean_text(df.get("headline", pd.Series(index=df.index, dtype="object")))
    df["ticker"] = df.get("ticker", "").fillna("").astype(str).str.upper().str.strip()
    df["source"] = _clean_text(df.get("source", pd.Series(index=df.index, dtype="object")))
    df["source_type"] = _clean_text(df.get("source_type", pd.Series(index=df.index, dtype="object")))
    df["lang"] = _clean_text(df.get("lang", pd.Series(index=df.index, dtype="object"))).replace("", "unknown")
    df["url"] = _clean_text(df.get("url", pd.Series(index=df.index, dtype="object")))
    df["company"] = _clean_text(df.get("company", pd.Series(index=df.index, dtype="object")))
    df["tags"] = _clean_text(df.get("tags", pd.Series(index=df.index, dtype="object")))
    df["author"] = ""
    df["source_file"] = _clean_text(df.get("source_file", pd.Series(index=df.index, dtype="object"))).replace("", path.name)
    return df


def _normalize_moneycontrol(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["datetime"] = _to_naive_datetime(df.get("datePublished"))
    df["trade_date"] = df["datetime"].dt.normalize()
    df["ticker"] = df.get("symbol", "").fillna("").astype(str).str.upper().str.strip()
    df["headline"] = _clean_text(df.get("headline", pd.Series(index=df.index, dtype="object")))
    df["body"] = _clean_text(df.get("articleBody", pd.Series(index=df.index, dtype="object")))
    df["source"] = "Moneycontrol"
    df["source_type"] = "archive"
    df["lang"] = "en"
    df["url"] = _clean_text(df.get("url", pd.Series(index=df.index, dtype="object")))
    df["company"] = _clean_text(df.get("company", pd.Series(index=df.index, dtype="object")))
    df["tags"] = _clean_text(df.get("tags", pd.Series(index=df.index, dtype="object")))
    df["author"] = _clean_text(df.get("author", pd.Series(index=df.index, dtype="object")))
    df["source_file"] = path.name
    return df


def _normalize(path: Path) -> pd.DataFrame:
    if path.name == "final_news_merged.csv":
        df = _normalize_final_news_merged(path)
    elif path.name == "preprocessed_nifty_context_part3.csv":
        df = _normalize_preprocessed_context(path)
    elif path.name == "Nifty50_news_data(2020Jan_2024April).csv":
        df = _normalize_moneycontrol(path)
    else:
        raise ValueError(f"Unsupported input file: {path}")

    df = df.reindex(columns=OUTPUT_COLUMNS)
    df = df.dropna(subset=["datetime", "headline"])
    df = df[df["headline"].astype(str).str.strip() != ""]
    return df


def merge_news_csvs(input_paths: list[Path], output_path: Path) -> pd.DataFrame:
    frames = []
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        frames.append(_normalize(path))

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["datetime", "ticker", "headline"], kind="stable").reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["url", "datetime", "ticker", "headline"], keep="first")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge normalized news CSVs into one file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_news_csvs(DEFAULT_INPUTS, args.output)
    print(f"Merged {len(merged):,} rows into {args.output}")


if __name__ == "__main__":
    main()
