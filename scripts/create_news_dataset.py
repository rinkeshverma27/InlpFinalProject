import csv
import os
from datetime import datetime
import re

# Comprehensive mapping for Nifty 50 companies
TICKER_MAP = {
    'ADANIENT': ['Adani Enterprises'],
    'ADANIPORTS': ['Adani Ports', 'Special Economic Zone', 'Adani Ports & SEZ'],
    'APOLLOHOSP': ['Apollo Hospitals'],
    'ASIANPAINT': ['Asian Paints'],
    'AXISBANK': ['Axis Bank'],
    'BAJAJ-AUTO': ['Bajaj Auto'],
    'BAJAJFINSV': ['Bajaj Finserv'],
    'BAJFINANCE': ['Bajaj Finance'],
    'BHARTIARTL': ['Bharti Airtel', 'Airtel'],
    'BPCL': ['Bharat Petroleum', 'BPCL'],
    'BRITANNIA': ['Britannia Industries', 'Britannia'],
    'CIPLA': ['Cipla'],
    'COALINDIA': ['Coal India'],
    'DIVISLAB': ["Divi's Laboratories", 'Divis Lab'],
    'DRREDDY': ["Dr. Reddy's Laboratories", 'Dr Reddys', 'Dr Reddy'],
    'EICHERMOT': ['Eicher Motors', 'Eicher'],
    'GRASIM': ['Grasim Industries', 'Grasim'],
    'HCLTECH': ['HCL Technologies', 'HCL Tech'],
    'HDFCBANK': ['HDFC Bank'],
    'HDFCLIFE': ['HDFC Life Insurance', 'HDFC Life'],
    'HEROMOTOCO': ['Hero MotoCorp', 'Hero Moto'],
    'HINDALCO': ['Hindalco Industries', 'Hindalco'],
    'HINDUNILVR': ['Hindustan Unilever', 'HUL'],
    'ICICIBANK': ['ICICI Bank'],
    'INDUSINDBK': ['IndusInd Bank'],
    'INFY': ['Infosys', 'INFY'],
    'ITC': ['ITC Ltd', 'ITC'],
    'JSWSTEEL': ['JSW Steel'],
    'KOTAKBANK': ['Kotak Mahindra Bank', 'Kotak Bank'],
    'LT': ['Larsen & Toubro', 'L&T', 'Larsen and Toubro'],
    'LTIM': ['LTIMindtree'],
    'M&M': ['Mahindra & Mahindra', 'M&M', 'Mahindra and Mahindra'],
    'MARUTI': ['Maruti Suzuki', 'Maruti'],
    'NESTLEIND': ['Nestle India', 'Nestle'],
    'NTPC': ['NTPC Ltd', 'NTPC'],
    'ONGC': ['Oil & Natural Gas', 'ONGC'],
    'POWERGRID': ['Power Grid Corporation', 'Power Grid'],
    'RELIANCE': ['Reliance Industries', 'RIL', 'Reliance'],
    'SBILIFE': ['SBI Life'],
    'SBIN': ['State Bank of India', 'SBI'],
    'SHRIRAMFIN': ['Shriram Finance'],
    'SUNPHARMA': ['Sun Pharmaceutical', 'Sun Pharma'],
    'TATACONSUM': ['Tata Consumer Products', 'Tata Consumer'],
    'TATAMOTORS': ['Tata Motors', 'Tata Motors Ltd'],
    'TATASTEEL': ['Tata Steel', 'Tata Steel Ltd'],
    'TCS': ['Tata Consultancy Services', 'TCS'],
    'TECHM': ['Tech Mahindra', 'TechM'],
    'TITAN': ['Titan Company', 'Titan'],
    'ULTRACEMCO': ['UltraTech Cement', 'UltraTech'],
    'WIPRO': ['Wipro']
}

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_date(date_str):
    if not date_str:
        return None
    formats = ['%Y-%m-%d %H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S.%f', '%d-%m-%Y', '%Y/%m/%d']
    if 'T' in date_str: date_str = date_str.replace('T', ' ')
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError: continue
    if isinstance(date_str, str) and len(date_str) >= 10 and date_str[4] == '-' and date_str[7] == '-': return date_str[:10]
    return None

def extract_tickers(text):
    """Scan text for any Nifty 50 companies and return a list of tickers."""
    if not text:
        return []
    text_lower = text.lower()
    found = set()
    for ticker, keywords in TICKER_MAP.items():
        for kw in keywords:
            # Word boundary matching to avoid accidental matches
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
                found.add(ticker)
                break
    return list(found)

def process_news():
    output_file = 'data/master_news_dataset.csv'
    count = 0
    seen = set()
    
    with open(output_file, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=['date', 'ticker', 'headline', 'content', 'language'])
        writer.writeheader()
        
        def write_row(date, ticker, headline, content, language):
            nonlocal count
            if not date or not headline: return
            
            headline_clean = clean_text(headline).lower()
            content_clean = clean_text(content).lower()
            
            # Key shifted to (ticker, headline, content) to remove multi-day repetitions
            key = (ticker, headline_clean, content_clean) 
            if key in seen: return
            
            seen.add(key)
            writer.writerow({'date': date, 'ticker': ticker, 'headline': headline, 'content': content, 'language': language})
            count += 1

        # File configurations
        # We process 'Part 1/2' and 'Part 3' and others
        files_to_process = [
            # Labeled Stocks (priority)
            {'path': 'Nifty50_news_data(2020Jan_2024April).csv', 'date_col': 'datePublished', 'ticker_col': 'symbol', 'headline_col': 'headline', 'content_col': 'articleBody', 'content_alt': 'description', 'lang': 'en'},
            {'path': 'hindi_articles_all.csv', 'date_col': 'trading_date', 'date_alt': 'datePublished', 'ticker_col': 'symbol', 'headline_col': 'headline', 'content_col': 'articleBody', 'content_alt': 'description', 'lang_col': 'language', 'lang_default': 'hi', 'encoding': 'utf-8-sig'},
            {'path': 'nifty50_news_2020_2026.csv', 'date_col': 'timestamp', 'ticker_col': None, 'headline_col': 'headline', 'content_col': 'summary', 'lang': 'en'},
            
            # Context Files (need extraction & summarization)
            {'path': 'nifty_context_part1.csv', 'date_col': 'timestamp', 'ticker_col': None, 'headline_col': 'headline', 'content_col': 'summary', 'lang': 'en'},
            {'path': 'nifty_context_part3.csv', 'date_col': 'timestamp', 'ticker_col': None, 'headline_col': 'headline', 'content_col': 'summary', 'lang': 'en'},
            {'path': 'cleaned_nifty_part1.csv', 'date_col': 'datetime', 'ticker_col': None, 'headline_col': 'headline', 'content_col': 'body', 'lang': 'en'},
            
            # Additional Financial News
            {'path': 'english_financial.csv', 'date_col': None, 'ticker_col': None, 'headline_col': 'text', 'content_col': 'text', 'lang': 'en'},
        ]

        for cfg in files_to_process:
            file_path = cfg['path']
            if not os.path.exists(file_path):
                print(f"Skipping {file_path} (not found)")
                continue
            
            print(f"Processing {file_path}...")
            encoding = cfg.get('encoding', 'utf-8')
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Date handling: for files without date, skip or use a fallback if absolutely necessary 
                    # but backtesting requires dates. Plan says "7 source files", so we include it.
                    # If it has no date, parse_date returns None and we skip as per logic.
                    date = parse_date(row.get(cfg['date_col']) or row.get(cfg.get('date_alt')) if cfg.get('date_col') or cfg.get('date_alt') else None)
                    if not date:
                         # Special case: english_financial might have a date in another field or just skip
                         continue
                    
                    headline = clean_text(row.get(cfg['headline_col']))
                    # For content, always prefer summary if available, then articlesBody
                    content = clean_text(row.get(cfg['content_col']) or row.get(cfg.get('content_alt')))
                    lang = row.get(cfg.get('lang_col')) if 'lang_col' in cfg else cfg.get('lang', 'en')
                    
                    # Extraction logic
                    tickers_found = set()
                    
                    # Logic 1: Explicit Ticker
                    explicit_ticker = row.get(cfg['ticker_col']) if cfg.get('ticker_col') else None
                    if explicit_ticker:
                        # Clean explicit ticker (sometimes includes .NS or extra whitespace)
                        clean_t = explicit_ticker.replace('.NS', '').strip().upper()
                        if clean_t in TICKER_MAP:
                            tickers_found.add(clean_t)
                    
                    # Logic 2: Inferred Tickers (from headline + content)
                    inferred = extract_tickers(headline + " " + content)
                    for t in inferred:
                        tickers_found.add(t)
                    
                    for t in tickers_found:
                        write_row(date, t, headline, content, lang)

    print(f"Finished! Total items in Master News Dataset: {count}")

if __name__ == "__main__":
    process_news()
