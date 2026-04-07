import pandas as pd
from newspaper import Article
import time
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import argparse
from curl_cffi import requests

# ==========================================
# CONFIGURATION: MASTER EDITION (V8)
# ==========================================
# SAFE MODE: Industrial-Grade Safeguards to Avoid Blacklisting
MAX_WORKERS = 1         # DO NOT INCREASE (Sequential is safest for Akamai)
SAVE_INTERVAL = 10      # Save every 10 articles to protect progress
INITIAL_BACKOFF = 30    # Seconds to wait after first 403 error
MAX_RETRIES = 3         # Retries per URL
JITTER_RANGE = (5, 15)  # Random noise added to backoff
csv_lock = threading.Lock()

# Rotating User-Agents (Mimic different browser profiles)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
]

class MasterScraper:
    def __init__(self):
        self.session = requests.Session()
        self.consecutive_403_count = 0
        self.warmup_done = False

    def warmup(self):
        """Perform a Home Page hit to seed cookies and simulate a real user session."""
        try:
            print("--- SESSION WARMUP STARTED ---")
            ua = random.choice(USER_AGENTS)
            self.session.get("https://economictimes.indiatimes.com/", impersonate="chrome110", headers={"User-Agent": ua}, timeout=20)
            self.warmup_done = True
            print("--- WARMUP COMPLETE (Cookies Seeded) ---")
        except Exception as e:
            print(f"Warning: Warmup failed ({e}). Proceeding carefully.")

    def fetch_with_impersonation(self, url):
        """Fetch article HTML with TLS/JA3 impersonation and exponential backoff."""
        if not self.warmup_done:
            self.warmup()

        for attempt in range(MAX_RETRIES):
            try:
                ua = random.choice(USER_AGENTS)
                # Mimic Chrome 110 TLS Fingerprint
                response = self.session.get(
                    url, 
                    impersonate="chrome110", 
                    headers={"User-Agent": ua, "Referer": "https://www.google.com/"},
                    timeout=25
                )

                if response.status_code == 200:
                    self.consecutive_403_count = 0 # Reset block counter
                    return response.text
                
                elif response.status_code == 403:
                    self.consecutive_403_count += 1
                    if self.consecutive_403_count >= 2:
                        print(f"\n[CIRCUIT BREAKER] Consecutive 403 errors on {url}. EXITING to protect IP.")
                        os._exit(1) # Emergency stop

                    # Exponential Backoff + Jitter
                    backoff = (INITIAL_BACKOFF * (2 ** attempt)) + random.uniform(*JITTER_RANGE)
                    print(f"\n[BLOCKED] 403 on {url}. Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                
                else:
                    time.sleep(random.uniform(5, 10))

            except Exception as e:
                time.sleep(random.uniform(5, 10))
        
        return None

    def fetch_content_v8(self, url):
        html = self.fetch_with_impersonation(url)
        if not html: return ""
        try:
            article = Article(url)
            article.html = html
            article.is_downloaded = True
            article.parse()
            return article.text.strip()
        except Exception: return ""

def process_distributed_v8(input_source, system_number):
    if not os.path.exists(input_source):
        print(f"Error: Source file {input_source} not found.")
        return

    # Load targets
    df_source = pd.read_csv(input_source)
    total_total = len(df_source)
    chunk_size = total_total // 3
    ranges = {1: (0, chunk_size), 2: (chunk_size, 2*chunk_size), 3: (2*chunk_size, total_total)}
    
    start_idx, end_idx = ranges.get(system_number)
    output_file = f"nifty_context_part{system_number}.csv"
    
    df_system = df_source.iloc[start_idx:end_idx].copy()
    print(f"--- SYSTEM {system_number} MASTER EDITION ONLINE ---")
    print(f"Assigning {len(df_system)} articles to this machine.")

    # Check progress
    done_data = {}
    if os.path.exists(output_file):
        try:
            df_done = pd.read_csv(output_file)
            df_valid = df_done[df_done['summary'].notna() & (df_done['summary'].str.len() > 30)]
            done_data = dict(zip(df_valid['url'], df_valid['summary']))
            print(f"Resuming: {len(done_data)} articles already finished.")
        except Exception: pass

    rows_to_process = [row.to_dict() for _, row in df_system.iterrows() if row['url'] not in done_data]

    if not rows_to_process:
        print("All assigned articles are already complete!")
        return

    scraper_engine = MasterScraper()
    print(f"🚀 Starting Industrial-Grade Extraction (V8) for {len(rows_to_process)} articles...")
    
    if not os.path.exists(output_file):
        pd.DataFrame(columns=df_source.columns).to_csv(output_file, index=False)

    batch = []
    pbar = tqdm(total=len(rows_to_process), desc=f"System-{system_number}")
    
    for row in rows_to_process:
        row['summary'] = scraper_engine.fetch_content_v8(row['url'])
        if row['summary'] and len(row['summary']) > 30:
            batch.append(row)
        
        pbar.update(1)
        
        # Human-like delay 15-30s
        time.sleep(random.uniform(15.0, 30.0))
        
        if len(batch) >= SAVE_INTERVAL:
            with csv_lock:
                pd.DataFrame(batch).to_csv(output_file, mode='a', index=False, header=False)
            batch = []
    
    pbar.close()
    if batch:
        with csv_lock:
            pd.DataFrame(batch).to_csv(output_file, mode='a', index=False, header=False)

    print(f"\n✅ System {system_number} Master Run Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()
    process_distributed_v8("nifty50_news_2020_2026.csv", args.system)
