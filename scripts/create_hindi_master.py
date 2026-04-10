import csv
import os
from datetime import datetime
import re
from glob import glob
import sys

# Comprehensive mapping for Nifty 50 companies with Hindi support
TICKER_MAP = {
    'ADANIENT': ['Adani Enterprises', 'अडानी एंटरप्राइजेज', 'अडाणी एंटरप्राइजेज'],
    'ADANIPORTS': ['Adani Ports', 'Special Economic Zone', 'Adani Ports & SEZ', 'अडानी पोर्ट्स', 'अडाणी पोर्ट्स'],
    'APOLLOHOSP': ['Apollo Hospitals', 'अपोलो हॉस्पिटल्स'],
    'ASIANPAINT': ['Asian Paints', 'एशियन पेंट्स'],
    'AXISBANK': ['Axis Bank', 'एक्सिस बैंक'],
    'BAJAJ-AUTO': ['Bajaj Auto', 'बजाज ऑटो'],
    'BAJAJFINSV': ['Bajaj Finserv', 'बजाज फिनसर्व'],
    'BAJFINANCE': ['Bajaj Finance', 'बजाज फाइनेंस'],
    'BHARTIARTL': ['Bharti Airtel', 'Airtel', 'भारती एयरटेल', 'एयरटेल'],
    'BPCL': ['Bharat Petroleum', 'BPCL', 'बीपीसीएल', 'भारत पेट्रोलियम'],
    'BRITANNIA': ['Britannia Industries', 'Britannia', 'ब्रिटानिया'],
    'CIPLA': ['Cipla', 'सिप्ला'],
    'COALINDIA': ['Coal India', 'कोल इंडिया'],
    'DIVISLAB': ["Divi's Laboratories", 'Divis Lab', 'डिविस लैब'],
    'DRREDDY': ["Dr. Reddy's Laboratories", 'Dr Reddys', 'Dr Reddy', 'डॉ रेड्डी'],
    'EICHERMOT': ['Eicher Motors', 'Eicher', 'आयशर मोटर्स', 'आयशर'],
    'GRASIM': ['Grasim Industries', 'Grasim', 'ग्रासिम'],
    'HCLTECH': ['HCL Technologies', 'HCL Tech', 'एचसीएल टेक', 'एचसीएल'],
    'HDFCBANK': ['HDFC Bank', 'एचडीएफसी बैंक'],
    'HDFCLIFE': ['HDFC Life Insurance', 'HDFC Life', 'एचडीएफसी लाइफ'],
    'HEROMOTOCO': ['Hero MotoCorp', 'Hero Moto', 'हीरो मोटोकॉर्प', 'हीरो मोटो'],
    'HINDALCO': ['Hindalco Industries', 'Hindalco', 'हिंडाल्को'],
    'HINDUNILVR': ['Hindustan Unilever', 'HUL', 'हिंदुस्तान यूनिलीवर'],
    'ICICIBANK': ['ICICI Bank', 'आईसीआईसीआई बैंक'],
    'INDUSINDBK': ['IndusInd Bank', 'इंडसइंड बैंक'],
    'INFY': ['Infosys', 'INFY', 'इंफोसिस'],
    'ITC': ['ITC Ltd', 'ITC', 'आईटीसी'],
    'JSWSTEEL': ['JSW Steel', 'जेएसडब्ल्यू स्टील'],
    'KOTAKBANK': ['Kotak Mahindra Bank', 'Kotak Bank', 'कोटक महिंद्रा बैंक', 'कोटक बैंक'],
    'LT': ['Larsen & Toubro', 'L&T', 'Larsen and Toubro', 'लार्सन एंड टुब्रो', 'एलएंडटी'],
    'LTIM': ['LTIMindtree'],
    'M&M': ['Mahindra & Mahindra', 'M&M', 'Mahindra and Mahindra', 'महिंद्रा एंड महिंद्रा', 'महिंद्रा'],
    'MARUTI': ['Maruti Suzuki', 'Maruti', 'मारुति सुजुकी', 'मारुति'],
    'NESTLEIND': ['Nestle India', 'Nestle', 'नेस्ले इंडिया', 'नेस्ले'],
    'NTPC': ['NTPC Ltd', 'NTPC', 'एनटीपीसी'],
    'ONGC': ['Oil & Natural Gas', 'ONGC', 'ओएनजीसी', 'ऑयल एंड नेचुरल गैस'],
    'POWERGRID': ['Power Grid Corporation', 'Power Grid', 'पावर ग्रिड'],
    'RELIANCE': ['Reliance Industries', 'RIL', 'Reliance', 'रिलायंस इंडस्ट्रीज', 'रिलायंस'],
    'SBILIFE': ['SBI Life', 'एसबीआई लाइफ'],
    'SBIN': ['State Bank of India', 'SBI', 'एसबीआई', 'स्टेट बैंक ऑफ इंडिया'],
    'SHRIRAMFIN': ['Shriram Finance', 'श्रीराम फाइनेंस'],
    'SUNPHARMA': ['Sun Pharmaceutical', 'Sun Pharma', 'सन फार्मा'],
    'TATACONSUM': ['Tata Consumer Products', 'Tata Consumer', 'टाटा कंज्यूमर'],
    'TATAMOTORS': ['Tata Motors', 'Tata Motors Ltd', 'टाटा मोटर्स'],
    'TATASTEEL': ['Tata Steel', 'Tata Steel Ltd', 'टाटा स्टील'],
    'TCS': ['Tata Consultancy Services', 'TCS', 'टीसीएस', 'टाटा कंसल्टेंसी सर्विसेज'],
    'TECHM': ['Tech Mahindra', 'TechM', 'टेक महिंद्रा'],
    'TITAN': ['Titan Company', 'Titan', 'टाइटन'],
    'ULTRACEMCO': ['UltraTech Cement', 'UltraTech', 'अल्ट्राटेक'],
    'WIPRO': ['Wipro', 'विप्रो']
}

# Pre-compile regex for performance
KEYWORD_TO_TICKER = {}
for ticker, keywords in TICKER_MAP.items():
    for kw in keywords:
        KEYWORD_TO_TICKER[kw.lower()] = ticker

ALL_KEYWORDS = sorted(KEYWORD_TO_TICKER.keys(), key=len, reverse=True)
TICKER_REGEX = re.compile(r'\b(' + '|'.join(map(re.escape, ALL_KEYWORDS)) + r')\b', re.IGNORECASE)

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_date(date_str):
    if not date_str:
        return None
    if not isinstance(date_str, str): return None
    date_str = date_str.strip()
    if not date_str: return None
    
    formats = ['%Y-%m-%d %H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S.%f', '%d-%m-%Y', '%Y/%m/%d']
    if 'T' in date_str: date_str = date_str.replace('T', ' ')
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError: continue
    
    if len(date_str) >= 10 and date_str[4] == '-' and date_str[7] == '-': return date_str[:10]
    return None

def extract_tickers_fast(text):
    if not text:
        return []
    matches = TICKER_REGEX.findall(text)
    found = set()
    for m in matches:
        ticker = KEYWORD_TO_TICKER.get(m.lower())
        if ticker:
            found.add(ticker)
    return list(found)

def process_hindi_news():
    hindi_dir = 'Hindi/'
    output_file = 'data/hindi_master.csv'
    total_count = 0
    seen = set()
    
    os.makedirs('data', exist_ok=True)
    csv_files = sorted(glob(os.path.join(hindi_dir, "*.csv")))
    
    if not csv_files:
        print(f"No CSV files found in {hindi_dir}")
        return

    print(f"Found {len(csv_files)} files to process.")

    with open(output_file, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=['date', 'ticker', 'headline', 'content', 'language'])
        writer.writeheader()
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            file_matches = 0
            try:
                # Use utf-8-sig to handle optional BOM
                with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Normalize keys to handle potential BOM or variations in header naming
                        clean_row = {k.lstrip('\ufeff'): v for k, v in row.items() if k is not None}
                        
                        date_raw = clean_row.get('datePublished') or clean_row.get('timestamp') or clean_row.get('trading_date')
                        date = parse_date(date_raw)
                        
                        headline = clean_row.get('headline', '')
                        description = clean_row.get('description', '')
                        article_body = clean_row.get('articleBody', '')
                        explicit_symbol = clean_row.get('symbol', '').strip().upper()
                        explicit_company = clean_row.get('company', '').strip()
                        lang = clean_row.get('language', 'hi')

                        full_text = f"{headline} {description} {article_body}"
                        
                        tickers_found = set()
                        
                        if explicit_symbol and explicit_symbol in TICKER_MAP:
                            tickers_found.add(explicit_symbol)
                        elif explicit_company and explicit_company.upper() != 'MACRO':
                             found_from_company = extract_tickers_fast(explicit_company)
                             for t in found_from_company:
                                 tickers_found.add(t)

                        inferred = extract_tickers_fast(full_text)
                        for t in inferred:
                            tickers_found.add(t)
                        
                        if not tickers_found:
                            continue

                        headline_clean = clean_text(headline)
                        content_clean = clean_text(article_body or description)
                        
                        for t in tickers_found:
                            if not date or not headline_clean:
                                continue
                                
                            key = (t, headline_clean.lower(), content_clean.lower()[:500])
                            if key in seen:
                                continue
                            
                            seen.add(key)
                            writer.writerow({
                                'date': date,
                                'ticker': t,
                                'headline': headline_clean,
                                'content': content_clean,
                                'language': lang
                            })
                            file_matches += 1
                
                fout.flush()
                print(f"Processed {file_name}: {file_matches} matches found.")
                total_count += file_matches
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Finished! Total unique items in Hindi Master Dataset: {total_count}")

if __name__ == "__main__":
    process_hindi_news()
