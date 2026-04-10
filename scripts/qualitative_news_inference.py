#!/usr/bin/env python3
"""
Run qualitative news test cases through the trained stock-direction model.

What it does:
1. Takes a list of news test cases (built-in by default, or from a JSON file)
2. Detects language and scores headline sentiment using FinBERT / MuRIL
3. Builds the latest available price sequence for the ticker
4. Broadcasts the news sentiment across the sequence window
5. Uses the trained dual-stream model to predict:
   UP / DOWN / LOW-CONFIDENCE UP / LOW-CONFIDENCE DOWN / ABSTAIN

Usage:
  python3 scripts/qualitative_news_inference.py
  python3 scripts/qualitative_news_inference.py --input path/to/cases.json
  python3 scripts/qualitative_news_inference.py --output reports/qualitative_cases.csv

Expected JSON input format:
[
  {
    "ticker": "TCS",
    "news_input": "TCS begins earnings season amid mixed global cues...",
    "label": "optional free-text label"
  }
]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))

from src.data.ohlcv_loader import FEATURE_COLS, load_ohlcv
from src.features.window_sizer import get_window
from src.model.dual_stream_lstm import build_model
from src.model.mc_dropout import predict_single
from src.nlp.finbert_scorer import score_english
from src.nlp.lang_detector import detect_language
from src.nlp.muril_scorer import score_hindi
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import PRODUCTION_DIR

log = get_logger("qualitative_news_inference")



@dataclass
class NewsCase:
    ticker: str
    news_input: str
    label: str = ""
    expected_polarity: str = ""
    expected_model_output: str = ""
    explanation: str = ""


DEFAULT_CASES: List[NewsCase] = [
    NewsCase(
        ticker="BHARTIARTL",
        news_input="Bharti Airtel expands 5G footprint with 4,300 new sites in Eastern Uttar Pradesh, improving network coverage across urban and rural markets.",
        label="Strong positive infrastructure expansion",
        expected_polarity="Positive",
        expected_model_output="UP",
        explanation="Direct capex-driven network expansion signals revenue growth potential.",
    ),
    NewsCase(
        ticker="SBIN",
        news_input="Q4 results se pehle PSU banks par brokerages bullish hain; healthy business updates aur better credit momentum ki wajah se SBI par buying advice di gayi hai.",
        label="Positive pre-results brokerage support",
        expected_polarity="Positive",
        expected_model_output="UP",
        explanation="Bullish brokerage consensus on SBI ahead of Q4 results.",
    ),
    NewsCase(
        ticker="ICICIBANK",
        news_input="RBI ne policy rate hold ki, jiske baad banking shares mein rally dekhi gayi; ICICI Bank ko rate-sensitive recovery theme ka beneficiary bataya gaya.",
        label="Positive policy-linked bank rerating",
        expected_polarity="Positive",
        expected_model_output="UP",
        explanation="RBI rate hold spurs banking rally; ICICI Bank named as key beneficiary.",
    ),
    NewsCase(
        ticker="NTPC",
        news_input="Government announces ₹12,000 crore thermal capacity addition plan; NTPC likely to be the anchor developer for three new coal-based units.",
        label="Policy support with direct order visibility",
        expected_polarity="Positive",
        expected_model_output="UP",
        explanation="Direct order pipeline from government policy translates to future earnings growth.",
    ),
    NewsCase(
        ticker="BAJFINANCE",
        news_input="Goldman Sachs retains a Buy on Bajaj Finance, citing execution strength and AUM growth visibility despite mixed consumer lending trends.",
        label="Brokerage positive with caveats",
        expected_polarity="Positive",
        expected_model_output="LOW-CONFIDENCE UP",
        explanation="Buy rating is positive, but 'mixed trends' caveat introduces some uncertainty.",
    ),
    NewsCase(
        ticker="INFY",
        news_input="Zoho founder says the Indian IT sector is struggling not only because of AI and tariffs, but also because of weak product depth and service inefficiency.",
        label="Strong sector-level negative for IT",
        expected_polarity="Negative",
        expected_model_output="DOWN",
        explanation="Broad negative critique of the IT sector from a credible industry voice.",
    ),
    NewsCase(
        ticker="BHARTIARTL",
        news_input="Market-cap erosion among top Indian firms was led by Bharti Airtel, which emerged as the biggest laggard in a weak holiday-shortened week.",
        label="Market-led negative tone",
        expected_polarity="Negative",
        expected_model_output="DOWN",
        explanation="Stock named as the biggest laggard with market-cap erosion.",
    ),
    NewsCase(
        ticker="RELIANCE",
        news_input="CAG says BSNL failed to recover infrastructure-sharing dues from Reliance Jio for years, leading to a large loss to the government.",
        label="Governance and regulatory overhang",
        expected_polarity="Negative",
        expected_model_output="DOWN",
        explanation="Regulatory / governance concern involving a government audit finding.",
    ),
    NewsCase(
        ticker="TATASTEEL",
        news_input="Sensex mein 500+ points ki girawat ke beech metals par pressure bana, geopolitics aur oil spike ki wajah se Tata Steel jaise cyclical names weak mood mein rahe.",
        label="Macro risk-off with metals pressure",
        expected_polarity="Negative",
        expected_model_output="DOWN",
        explanation="Broad market selloff with metals under pressure; Tata Steel named as weak.",
    ),
    NewsCase(
        ticker="WIPRO",
        news_input="Wipro ke Q4 results mein revenue growth flat rahi, lekin management ne FY27 mein AI-led deals se recovery ka guidance diya hai.",
        label="Mixed results with forward guidance",
        expected_polarity="Mixed",
        expected_model_output="LOW-CONFIDENCE UP",
        explanation="Weak current results offset by optimistic forward guidance — opposing signals.",
    ),
    NewsCase(
        ticker="ONGC",
        news_input="Crude oil crashes nearly 16%; while lower oil eases inflation concerns for the economy, it clouds near-term realization outlook for upstream producers like ONGC.",
        label="Mixed macro, stock-specific negative",
        expected_polarity="Mixed",
        expected_model_output="LOW-CONFIDENCE DOWN",
        explanation="Macro-positive (lower inflation) but stock-specific negative (lower realizations for ONGC).",
    ),
    NewsCase(
        ticker="ITC",
        news_input="ITC chairman says India is better placed to manage US tariff disruption and may benefit from an early bilateral trade agreement.",
        label="Constructive macro but indirect",
        expected_polarity="Positive",
        expected_model_output="LOW-CONFIDENCE UP",
        explanation="Positive macro outlook from company leadership, but no direct earnings catalyst.",
    ),
    NewsCase(
        ticker="HDFCBANK",
        news_input="Top Technical Picks mein HDFC Bank ko weak market ke bawajood strong setup wala stock bataya gaya, even as broader indices stayed under pressure.",
        label="Weak positive technical signal",
        expected_polarity="Mixed",
        expected_model_output="LOW-CONFIDENCE UP",
        explanation="Technical pick despite weak market — positive for HDFC Bank but bearish broader context.",
    ),
    NewsCase(
        ticker="KOTAKBANK",
        news_input="Should investors buy Kotak Mahindra Bank ahead of Q4 results, or is the asset-quality risk still too large to ignore?",
        label="Ambiguous pre-results question",
        expected_polarity="Mixed",
        expected_model_output="ABSTAIN",
        explanation="Rhetorical question with no directional stance; maximum ambiguity.",
    ),
    NewsCase(
        ticker="POWERGRID",
        news_input="Power Grid Corporation employees may receive a performance-linked reward ahead of the festive season, though the direct earnings impact remains unclear.",
        label="Weak, ambiguous corporate update",
        expected_polarity="Mixed",
        expected_model_output="ABSTAIN",
        explanation="Minor corporate HR update with no material earnings impact.",
    ),
    NewsCase(
        ticker="BPCL",
        news_input="BPCL ki refining margins mein tezi aayi hai global supply cuts ke baad, lekin government ne fuel subsidy ka burden badhane ka ishara diya hai.",
        label="Positive fundamentals vs. policy risk",
        expected_polarity="Mixed",
        expected_model_output="LOW-CONFIDENCE DOWN",
        explanation="Positive refining margins offset by government subsidy burden signal.",
    ),
    NewsCase(
        ticker="SBIN",
        news_input="RBI imposes ₹1 crore penalty on State Bank of India for non-compliance with KYC norms and customer account regulations.",
        label="Regulatory penalty — negative but immaterial",
        expected_polarity="Negative",
        expected_model_output="LOW-CONFIDENCE DOWN",
        explanation="Regulatory penalty is negative in tone but financially immaterial for SBI.",
    ),
    NewsCase(
        ticker="RELIANCE",
        news_input="Reliance Retail reports record quarterly revenue of ₹78,000 crore, driven by strong store expansion and digital commerce growth across metros and Tier-2 cities.",
        label="Strong subsidiary earnings beat",
        expected_polarity="Positive",
        expected_model_output="UP",
        explanation="Record revenue from a key subsidiary with structural growth drivers.",
    ),
]


def load_cases(input_path: Optional[Path]) -> List[NewsCase]:
    if input_path is None:
        return DEFAULT_CASES

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Pass a valid JSON file with qualitative cases, or omit --input to use the built-in cases."
        )

    records = json.loads(input_path.read_text(encoding="utf-8"))
    return [
        NewsCase(
            ticker=str(item["ticker"]).upper(),
            news_input=str(item["news_input"]),
            label=str(item.get("label", "")),
            expected_polarity=str(item.get("expected_polarity", "")),
            expected_model_output=str(item.get("expected_model_output", "")),
            explanation=str(item.get("explanation", "")),
        )
        for item in records
    ]


def score_text_sentiment(text: str, cfg: dict, device: torch.device) -> tuple[str, dict]:
    lang = detect_language(text, cfg.get("nlp", {}).get("lang_confidence_threshold", 0.85))

    if lang in {"hi", "hinglish"}:
        scores = score_hindi([text], cfg, device=device)
    else:
        lang = "en" if lang == "unknown" else lang
        scores = score_english([text], cfg, device=device)

    row = scores.iloc[0]
    pos = float(row["pos"])
    neg = float(row["neg"])
    neu = float(row["neu"])
    net = pos - neg

    if net >= 0.15:
        polarity = "Positive"
    elif net <= -0.15:
        polarity = "Negative"
    else:
        polarity = "Mixed"

    return lang, {
        "polarity": polarity,
        "pos": round(pos, 4),
        "neg": round(neg, 4),
        "neu": round(neu, 4),
        "net_sentiment": round(net, 4),
    }


def build_sentiment_vector(sentiment_scores: dict) -> np.ndarray:
    pos = sentiment_scores["pos"]
    neg = sentiment_scores["neg"]
    neu = sentiment_scores["neu"]
    # For a single qualitative headline, broadcast the same sentiment across
    # the 1d / 3d / 7d slots as a lightweight proxy.
    return np.array([pos, neg, neu, pos, neg, neu, pos, neg, neu], dtype=np.float32)


def build_latest_price_sequence(ticker: str, cfg: dict) -> tuple[torch.Tensor, str, int]:
    price_df = load_ohlcv(ticker, cfg)
    feat_cfg = cfg.get("features", {})
    w_min = feat_cfg.get("window_min", 10)
    w_max = feat_cfg.get("window_max", 60)
    w_scale = feat_cfg.get("window_atr_scale", 50)

    if len(price_df) <= w_max:
        raise ValueError(f"[{ticker}] Not enough OHLCV rows ({len(price_df)}) for max_window={w_max}.")

    atr_series = price_df["atr_14_norm"].rank(pct=True)
    idx = len(price_df) - 1
    window = get_window(float(atr_series.iloc[idx]), w_min, w_max, w_scale)
    start = idx - window + 1

    price_arr = price_df[FEATURE_COLS].values.astype(np.float32)
    p_seq = price_arr[start:idx + 1]

    pad = w_max - len(p_seq)
    if pad > 0:
        p_seq = np.concatenate(
            [np.zeros((pad, p_seq.shape[1]), dtype=np.float32), p_seq],
            axis=0,
        )

    latest_date = price_df.index[idx].strftime("%Y-%m-%d")
    return torch.tensor(p_seq, dtype=torch.float32), latest_date, window


def available_ohlcv_tickers() -> list[str]:
    ohlcv_dir = ROOT / "data" / "raw" / "ohlcv"
    tickers = []
    for path in sorted(ohlcv_dir.glob("*.csv")):
        if path.stem == "20year_train":
            continue
        tickers.append(path.stem.upper())
    return tickers



def ensure_cases_have_ohlcv(cases: list[NewsCase]) -> list[NewsCase]:
    """Strictly filter to only cases whose ticker has OHLCV data available."""
    supported_tickers = set(available_ohlcv_tickers())
    valid_cases: list[NewsCase] = []
    dropped: list[str] = []

    for case in cases:
        if case.ticker.upper() in supported_tickers:
            valid_cases.append(case)
        else:
            dropped.append(case.ticker.upper())

    if dropped:
        log.warning(
            "Dropped %d case(s) — no OHLCV data for ticker(s): %s",
            len(dropped),
            ", ".join(sorted(set(dropped))),
        )

    log.info(
        "Retained %d / %d cases with OHLCV support.",
        len(valid_cases),
        len(cases),
    )
    return valid_cases


def map_model_output(raw_result: dict) -> str:
    direction = raw_result["direction"]
    if direction == "ABSTAIN":
        return "ABSTAIN"

    confidence = float(raw_result["confidence"])
    if confidence < 0.20:
        return f"LOW-CONFIDENCE {direction}"
    return direction


def run_cases(cases: Iterable[NewsCase], cfg: dict, device: torch.device) -> pd.DataFrame:
    model_path = PRODUCTION_DIR / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python main.py train` first."
        )

    model = build_model(cfg).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    supported_tickers = set(available_ohlcv_tickers())

    rows = []
    for case in cases:
        ticker = case.ticker.upper()
        if ticker not in supported_tickers:
            log.warning(f"[{ticker}] Skipping case: no OHLCV file found.")
            continue

        lang, sent = score_text_sentiment(case.news_input, cfg, device)
        price_seq, latest_date, effective_window = build_latest_price_sequence(ticker, cfg)
        sent_vec = build_sentiment_vector(sent)
        sent_seq = np.repeat(sent_vec[None, :], price_seq.shape[0], axis=0)
        sent_tensor = torch.tensor(sent_seq, dtype=torch.float32)

        raw_pred = predict_single(model, price_seq, sent_tensor, cfg, device)
        actual_output = map_model_output(raw_pred)
        matches_expected = (
            actual_output == case.expected_model_output
            if case.expected_model_output
            else None
        )

        rows.append({
            "ticker": ticker,
            "label": case.label,
            "language": lang,
            "news_input": case.news_input,
            "expected_polarity": case.expected_polarity,
            "sentiment_polarity": sent["polarity"],
            "pos": sent["pos"],
            "neg": sent["neg"],
            "neu": sent["neu"],
            "net_sentiment": sent["net_sentiment"],
            "expected_model_output": case.expected_model_output,
            "actual_model_output": actual_output,
            "match_expected": matches_expected,
            "raw_direction": raw_pred["direction"],
            "probability_up": raw_pred["probability"],
            "variance": raw_pred["variance"],
            "confidence": raw_pred["confidence"],
            "latest_price_date_used": latest_date,
            "effective_window": effective_window,
            "explanation": case.explanation,
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative news inference using the trained stock-direction model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", default=None, help="Optional JSON file with test cases")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--profile", default=None, choices=["4gb", "8gb", "full"], help="Optional profile override")
    args = parser.parse_args()

    cfg = load_config(Path(args.config), profile_override=args.profile)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        cases = load_cases(Path(args.input) if args.input else None)
    except FileNotFoundError as exc:
        print(exc)
        return

    cases = ensure_cases_have_ohlcv(cases)

    results = run_cases(cases, cfg, device)

    output_path = Path(args.output) if args.output else ROOT / "logs" / "qualitative_news_inference.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    display_cols = [
        "ticker",
        "language",
        "expected_polarity",
        "sentiment_polarity",
        "expected_model_output",
        "actual_model_output",
        "match_expected",
        "probability_up",
        "confidence",
        "label",
    ]
    print(results[display_cols].to_string(index=False))

    # Print summary
    if "match_expected" in results.columns:
        matched = results["match_expected"].sum()
        total = results["match_expected"].notna().sum()
        print(f"\n✓ Expected-output match rate: {matched}/{total} ({100*matched/total:.1f}%)" if total > 0 else "")
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
