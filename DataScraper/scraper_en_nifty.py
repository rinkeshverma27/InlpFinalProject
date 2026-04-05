"""
scraper_en_nifty.py  v3
─────────────────────────────────────────────────────────────────────────────
English financial news — all 50 Nifty-50 stocks
Same fixes as Hindi v7:
  • Wayback windows only marked done on confirmed CDX response
  • ET sitemap recursive parsing (index-of-indexes)
  • Auto-resume loop (--loop N --gap SECONDS)
  • reset_failed_wayback_windows() on every startup

Sources:
  A. ET RSS (5 feeds) + NDTV + MC + BS — aiohttp
  B. ET sitemap (recursive) — curl_cffi
  C. Wayback CDX (MC + BS + ET + FE) — aiohttp

HOW TO RUN
──────────
pip install curl_cffi aiohttp beautifulsoup4 lxml pandas tqdm
python scraper_en_nifty.py --diagnose
python scraper_en_nifty.py --loop 20 --gap 120
"""

import sys
if sys.platform == "win32":
    import asyncio as _aio_fix
    _aio_fix.set_event_loop_policy(_aio_fix.WindowsSelectorEventLoopPolicy())

import argparse, asyncio, csv, json, logging, os, re, time, warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] pip install tqdm"); sys.exit(1)
try:
    from curl_cffi.requests import AsyncSession as CurlSession
except ImportError:
    print("[ERROR] pip install curl_cffi"); sys.exit(1)
try:
    import aiohttp
except ImportError:
    print("[ERROR] pip install aiohttp"); sys.exit(1)
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("[ERROR] pip install beautifulsoup4 lxml"); sys.exit(1)

log = logging.getLogger("en_scraper")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

IST            = ZoneInfo("Asia/Kolkata")
STOP_DATE      = date(2021, 1, 1)
END_DATE       = date.today()
OUT_DIR        = Path("dataset/raw_dataset")
OUT_CSV        = OUT_DIR / "english_news_nifty50.csv"
CKPT_FILE      = OUT_DIR / "en_scraper_checkpoint.json"
BATCH_MONTHS   = 8
CONCURRENCY    = 4
PARSE_WORKERS  = 4
MAX_RETRIES    = 3
RETRY_DELAY    = 2.0
MARKET_CLOSE_H = 15
MARKET_CLOSE_M = 30
WAYBACK_LIMIT  = 500
WAYBACK_TIMEOUT = 90

def _make_windows(start, end, months):
    windows, cur = [], start
    while cur <= end:
        m_end = cur.month + months - 1
        y_end = cur.year + (m_end - 1) // 12
        m_end = (m_end - 1) % 12 + 1
        w_end = min(date(y_end, m_end, 1) + timedelta(days=31),
                    end + timedelta(days=1)) - timedelta(days=1)
        windows.append((cur, min(w_end, end)))
        cur = min(w_end, end) + timedelta(days=1)
    return list(reversed(windows))

ALL_WINDOWS = _make_windows(STOP_DATE, END_DATE, BATCH_MONTHS)

NIFTY50 = {
    "RELIANCE":   ["reliance industries", "ril", "mukesh ambani"],
    "HDFCBANK":   ["hdfc bank"],
    "ICICIBANK":  ["icici bank"],
    "INFY":       ["infosys"],
    "TCS":        ["tata consultancy", "tcs"],
    "KOTAKBANK":  ["kotak mahindra", "kotak bank"],
    "LT":         ["larsen toubro", "l&t"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "AXISBANK":   ["axis bank"],
    "BAJFINANCE": ["bajaj finance"],
    "SBIN":       ["state bank of india", "sbi"],
    "MARUTI":     ["maruti suzuki"],
    "WIPRO":      ["wipro"],
    "ULTRACEMCO": ["ultratech cement"],
    "TATAMOTORS": ["tata motors"],
    "HCLTECH":    ["hcl technologies", "hcl tech"],
    "SUNPHARMA":  ["sun pharma", "sun pharmaceutical"],
    "TITAN":      ["titan company"],
    "NESTLEIND":  ["nestle india"],
    "TECHM":      ["tech mahindra"],
    "NTPC":       ["ntpc"],
    "POWERGRID":  ["power grid"],
    "ONGC":       ["ongc", "oil natural gas"],
    "BPCL":       ["bharat petroleum", "bpcl"],
    "TATASTEEL":  ["tata steel"],
    "ITC":        ["itc limited", "itc ltd"],
    "ADANIENT":   ["adani enterprises"],
    "ADANIPORTS": ["adani ports"],
    "COALINDIA":  ["coal india"],
    "JSWSTEEL":   ["jsw steel"],
    "M&M":        ["mahindra & mahindra", "mahindra mahindra"],
    "EICHERMOT":  ["eicher motors", "royal enfield"],
    "DRREDDY":    ["dr reddys"],
    "HINDALCO":   ["hindalco"],
    "CIPLA":      ["cipla"],
    "BAJAJFINSV": ["bajaj finserv"],
    "GRASIM":     ["grasim"],
    "HEROMOTOCO": ["hero motocorp"],
    "APOLLOHOSP": ["apollo hospitals"],
    "DIVISLAB":   ["divis laboratories"],
    "TATACONSUM": ["tata consumer"],
    "INDUSINDBK": ["indusind bank"],
    "HDFCLIFE":   ["hdfc life"],
    "SBILIFE":    ["sbi life"],
    "BRITANNIA":  ["britannia"],
    "SHREECEM":   ["shree cement"],
    "UPL":        ["upl limited"],
    "BAJAJ-AUTO": ["bajaj auto"],
    "HDFC":       ["hdfc ltd", "housing development finance"],
    "ASIANPAINT": ["asian paints"],
}
MACRO_WORDS = [
    "rbi","repo rate","budget","nifty","sensex","crude oil","opec",
    "federal reserve","fed rate","inflation","gdp","rupee","fii","sebi",
    "lockdown","covid","war","recession","ipo","interest rate","election",
]

DATE_FMTS = [
    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
    "%B %d, %Y %I:%M %p IST", "%b %d, %Y %I:%M %p IST",
    "%d %B %Y", "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S GMT", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
]

def _parse_dt(raw):
    if not raw: return None
    for fmt in DATE_FMTS:
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=IST)
        except ValueError: continue
    return None

def _in_window(lm, w0, w1):
    if not lm: return True
    dt = _parse_dt(lm)
    return dt is None or w0 <= dt.date() <= w1

def _detect_tickers(text):
    tl = text.lower()
    return [t for t, kws in NIFTY50.items() if any(k in tl for k in kws)]

def _is_macro(text):
    tl = text.lower()
    return any(w in tl for w in MACRO_WORDS)

def _align_trading_day(dt):
    if dt.tzinfo is None: dt = dt.replace(tzinfo=IST)
    ist = dt.astimezone(IST)
    after = ist.hour > MARKET_CLOSE_H or (
        ist.hour == MARKET_CLOSE_H and ist.minute >= MARKET_CLOSE_M)
    d = ist.date() + timedelta(days=1 if after else 0)
    while d.weekday() >= 5: d += timedelta(days=1)
    return d

def _market_session(dt):
    ist = dt.astimezone(IST)
    h, m = ist.hour, ist.minute
    if h < 9 or (h == 9 and m < 15):   return "pre_open"
    if h < 15 or (h == 15 and m < 30): return "market_hours"
    return "post_close"

def load_checkpoint():
    if CKPT_FILE.exists():
        try: return json.loads(CKPT_FILE.read_text(encoding="utf-8"))
        except Exception: pass
    return {}

def save_checkpoint(ckpt):
    tmp = CKPT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt), encoding="utf-8")
    tmp.replace(CKPT_FILE)

def reset_failed_wayback_windows(ckpt):
    to_remove = []
    for key in list(ckpt.keys()):
        if key.startswith("wb_win_") and ckpt[key] is True:
            count_key = key + "_rows"
            if ckpt.get(count_key, 0) == 0:
                to_remove.append(key)
                to_remove.append(count_key)
    for k in to_remove: ckpt.pop(k, None)
    if to_remove:
        log.info(f"Reset {len(to_remove)//2} failed Wayback windows for retry")
    return ckpt

def load_seen_urls():
    seen = set()
    if OUT_CSV.exists():
        try:
            df = pd.read_csv(OUT_CSV, usecols=["url"])
            seen = set(df["url"].dropna())
        except Exception: pass
    log.info(f"Loaded {len(seen):,} existing URLs")
    return seen

FIELDNAMES = ["date","ticker","title","news","url","source",
              "published_at","market_session","is_macro","en_sentiment"]

def append_rows(rows):
    if not rows: return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header: w.writeheader()
        w.writerows(rows)
        f.flush(); os.fsync(f.fileno())

def parse_article_html(args):
    url, html, source_label = args
    try:
        soup = BeautifulSoup(html, "lxml")
        title = body = pub_date_str = None
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(tag.string or "")
                if isinstance(data, dict) and "@graph" in data:
                    data = data["@graph"]
                for item in (data if isinstance(data, list) else [data]):
                    t = item.get("@type","")
                    if any(x in t for x in ("NewsArticle","Article","WebPage")):
                        title        = title or item.get("headline")
                        body         = body  or item.get("articleBody")
                        pub_date_str = pub_date_str or item.get("datePublished")
            except Exception: continue
        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else None
        if not title:
            og = soup.find("meta", property="og:title")
            title = og.get("content","").strip() if og else None
        if not title:
            title = soup.title.string.strip() if soup.title else ""
        if not pub_date_str:
            for attr in ("article:published_time","datePublished"):
                m = (soup.find("meta", property=attr) or
                     soup.find("meta", itemprop=attr))
                if m and m.get("content"):
                    pub_date_str = m["content"]; break
        if not pub_date_str:
            t = soup.find("time", datetime=True)
            pub_date_str = t["datetime"] if t else None
        if not body:
            for sel in [".article-body p",".content-body p",".artText p",
                        ".article-txt p",".story-content p","article p"]:
                paras = soup.select(sel)
                if paras:
                    body = " ".join(p.get_text(strip=True) for p in paras); break
        if not body:
            paras = [p.get_text(strip=True) for p in soup.find_all("p")
                     if len(p.get_text(strip=True)) > 60]
            body = " ".join(paras[:25])
        if not body:
            og = soup.find("meta", property="og:description")
            body = og.get("content","").strip() if og else ""
        if not title or not body: return None
        published_at = _parse_dt(pub_date_str)
        if not published_at: return None
        return {"title": title.strip(), "news": body.strip()[:3000],
                "published_at": published_at, "source": source_label, "url": url}
    except Exception:
        return None

def parse_rss_item(args):
    title, description, pub_str, url, source_label = args
    published_at = _parse_dt(pub_str)
    if not published_at or not title: return None
    return {"title": title.strip(), "news": (description or title).strip()[:3000],
            "published_at": published_at, "source": source_label, "url": url}

def _make_rows(parsed_list, seen_urls):
    rows = []
    for p in parsed_list:
        if p is None or p["url"] in seen_urls: continue
        txt = p["title"] + " " + p["news"]
        tickers = _detect_tickers(txt); is_mac = _is_macro(txt)
        if not tickers and not is_mac: continue
        if not tickers: tickers = ["_MACRO_"]
        td = _align_trading_day(p["published_at"])
        for ticker in tickers:
            rows.append({
                "date": td.isoformat(), "ticker": ticker,
                "title": p["title"],   "news": p["news"],
                "url": p["url"],       "source": p["source"],
                "published_at": p["published_at"].isoformat(),
                "market_session": _market_session(p["published_at"]),
                "is_macro": is_mac,    "en_sentiment": 0.0,
            })
        seen_urls.add(p["url"])
    return rows

async def aio_get(session, url, timeout=20):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(
                url, ssl=False, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as r:
                if r.status == 200:
                    return await r.text(encoding="utf-8", errors="replace")
                if r.status in (429, 503):
                    await asyncio.sleep(RETRY_DELAY * attempt * 2)
                else: return ""
        except Exception as e:
            if attempt == MAX_RETRIES: return ""
            await asyncio.sleep(RETRY_DELAY * attempt)
    return ""

async def fetch_batch_aio(session, urls, label, sem):
    if not urls: return []
    pbar = tqdm(total=len(urls), desc=f"  fetch {label[:28]}",
                unit="url", leave=False, ncols=80)
    async def _one(u):
        async with sem:
            html = await aio_get(session, u)
            pbar.update(1); return (u, html)
    results = await asyncio.gather(*[_one(u) for u in urls])
    pbar.close(); return list(results)

def _parse_rss(text):
    soup = BeautifulSoup(text, "xml"); res = []
    for item in soup.find_all("item"):
        link = item.find("link")
        url  = link.get_text(strip=True) if link else ""
        if not url and link:
            sib = link.next_sibling; url = str(sib).strip() if sib else ""
        if not url:
            guid = item.find("guid"); url = guid.get_text(strip=True) if guid else ""
        pub  = item.find("pubDate") or item.find("dc:date")
        t    = item.find("title")
        desc = item.find("description") or item.find("summary")
        if url:
            raw_d = desc.get_text(strip=True) if desc else ""
            res.append((url, pub.get_text(strip=True) if pub else "",
                        t.get_text(strip=True) if t else "",
                        BeautifulSoup(raw_d, "lxml").get_text()))
    for entry in soup.find_all("entry"):
        link = entry.find("link", rel="alternate") or entry.find("link")
        url  = (link.get("href","") if link else "") or \
               (link.get_text(strip=True) if link else "")
        pub  = entry.find("published") or entry.find("updated")
        t    = entry.find("title")
        desc = entry.find("summary") or entry.find("content")
        if url:
            raw_d = desc.get_text(strip=True) if desc else ""
            res.append((url, pub.get_text(strip=True) if pub else "",
                        t.get_text(strip=True) if t else "",
                        BeautifulSoup(raw_d, "lxml").get_text()))
    return res

RSS_FEEDS = [
    {"name": "ET Markets",
     "url":  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"},
    {"name": "ET Stocks",
     "url":  "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"},
    {"name": "ET Economy",
     "url":  "https://economictimes.indiatimes.com/economy/rssfeeds/1373380680.cms"},
    {"name": "ET IPO",
     "url":  "https://economictimes.indiatimes.com/markets/ipos/fpos/rssfeeds/7771250.cms"},
    {"name": "NDTV Profit",
     "url":  "https://feeds.feedburner.com/ndtvprofit-latest"},
]

async def scrape_rss_feeds(seen_urls, ckpt, executor):
    all_rows = []
    conn = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0"}
    async with aiohttp.ClientSession(connector=conn, headers=hdrs) as session:
        sem  = asyncio.Semaphore(CONCURRENCY)
        pbar = tqdm(RSS_FEEDS, desc="RSS feeds", unit="feed", ncols=80)
        for feed in pbar:
            pbar.set_postfix_str(feed["name"][:22])
            try:
                async with session.get(feed["url"], ssl=False,
                    timeout=aiohttp.ClientTimeout(total=20)) as r:
                    text = await r.text(encoding="utf-8", errors="replace")
                    status = r.status
            except Exception as e:
                log.warning(f"RSS {feed['name']}: {e}"); continue
            entries = _parse_rss(text)
            log.info(f"RSS {feed['name']}: HTTP {status}, {len(entries)} entries")
            to_process = [(u, p, t, d) for u, p, t, d in entries
                          if u not in seen_urls and
                          (not _parse_dt(p) or
                           STOP_DATE <= _parse_dt(p).date() <= END_DATE)]
            if not to_process: continue
            results  = await fetch_batch_aio(
                session, [u for u,*_ in to_process], feed["name"], sem)
            html_map = {u: h for u, h in results}
            parsed = []
            for url, pub_str, title_fb, desc_fb in to_process:
                html = html_map.get(url,"")
                if html:
                    p = parse_article_html((url, html, feed["name"]))
                    if p: parsed.append(p); continue
                p = parse_rss_item((title_fb, desc_fb, pub_str, url, feed["name"]))
                if p: parsed.append(p)
            feed_rows = _make_rows(parsed, seen_urls)
            append_rows(feed_rows); all_rows.extend(feed_rows)
            log.info(f"RSS {feed['name']}: saved {len(feed_rows)} rows")
        pbar.close()
    return all_rows

ET_SM_INDEX = ("https://economictimes.indiatimes.com"
               "/etstatic/sitemaps/et/news/sitemap-index.xml")
ET_TARGETS  = ["/markets/stocks/news/", "/markets/", "/economy/"]

async def _curl_get(session, url, timeout=25):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = await session.get(url, timeout=timeout)
            if r.status_code == 200: return r.text
            if r.status_code in (429, 503):
                await asyncio.sleep(RETRY_DELAY * attempt * 2)
            else: return ""
        except (asyncio.CancelledError, Exception) as e:
            if attempt == MAX_RETRIES: return ""
            await asyncio.sleep(RETRY_DELAY * attempt)
    return ""

async def _et_get_all_entries(session):
    async def _fetch_sm(url, depth=0):
        if depth > 2: return []
        html = await _curl_get(session, url, timeout=30)
        if not html: return []
        soup = BeautifulSoup(html, "xml")
        sub_locs = [l.get_text(strip=True) for l in soup.find_all("loc")
                    if soup.find("sitemap")]
        if sub_locs and not soup.find("url"):
            entries = []
            pbar_sub = tqdm(sub_locs, desc=f"  ET sub-sitemaps d={depth}",
                            unit="sm", leave=False, ncols=80)
            for sub in pbar_sub:
                entries.extend(await _fetch_sm(sub, depth+1))
            pbar_sub.close()
            return entries
        entries = []
        for entry in soup.find_all("url"):
            loc = entry.find("loc"); lm = entry.find("lastmod")
            if not loc: continue
            url = loc.get_text(strip=True)
            if any(p in url for p in ET_TARGETS):
                entries.append((url, lm.get_text(strip=True) if lm else ""))
        return entries
    log.info("ET sitemap: fetching index...")
    entries = await _fetch_sm(ET_SM_INDEX)
    log.info(f"ET sitemap: {len(entries)} total article entries")
    return entries

async def scrape_et_sitemap(seen_urls, ckpt, executor):
    all_rows = []
    done_sm  = set(ckpt.get("et_sm_done", []))
    sem      = asyncio.Semaphore(CONCURRENCY)
    async with CurlSession(impersonate="chrome124") as session:
        all_entries = await _et_get_all_entries(session)
        if not all_entries:
            log.warning("ET sitemap: 0 entries — skipping"); return []
        valid = [(u, lm) for u, lm in all_entries
                 if u not in seen_urls and _in_window(lm, STOP_DATE, END_DATE)
                 and u not in done_sm]
        log.info(f"ET sitemap: {len(valid)} new articles")
        pbar_win = tqdm(ALL_WINDOWS, desc="ET date windows", unit="win", ncols=80)
        for w0, w1 in pbar_win:
            wk = f"et_win_{w0}_{w1}"
            if ckpt.get(wk): continue
            pbar_win.set_postfix_str(f"{w0}→{w1}")
            to_fetch = [u for u, lm in valid
                        if u not in seen_urls and _in_window(lm, w0, w1)]
            if not to_fetch:
                ckpt[wk] = True; save_checkpoint(ckpt); continue
            results = await fetch_batch_aio(session, to_fetch,
                                            f"ET {w0}→{w1}", sem)
            pargs   = [(u, h, "Economic Times") for u, h in results if h]
            parsed  = list(tqdm(executor.map(parse_article_html, pargs),
                                total=len(pargs), desc="  parse ET",
                                unit="art", leave=False, ncols=80))
            rows = _make_rows(parsed, seen_urls)
            append_rows(rows); all_rows.extend(rows)
            for u, _ in [r for r in results if r[1]]: done_sm.add(u)
            ckpt["et_sm_done"] = list(done_sm)
            ckpt[wk] = True; save_checkpoint(ckpt)
            log.info(f"ET {w0}→{w1}: fetched {len(to_fetch)}, "
                     f"saved {len(rows)} (total: {len(all_rows)})")
        pbar_win.close()
    return all_rows

WAYBACK_TARGETS = [
    {"domain": "moneycontrol.com",   "path": "*/news/business/stocks/*"},
    {"domain": "moneycontrol.com",   "path": "*/news/business/*"},
    {"domain": "business-standard.com", "path": "*/markets/news/*"},
    {"domain": "economictimes.indiatimes.com", "path": "*/markets/stocks/news/*"},
    {"domain": "financialexpress.com", "path": "*/market/*"},
]

async def _wayback_cdx_query(session, domain, path, w0, w1):
    params = {"url": f"{domain}/{path}", "matchType": "prefix",
              "output": "json", "fl": "timestamp,original",
              "filter": "statuscode:200",
              "from": w0.strftime("%Y%m%d"), "to": w1.strftime("%Y%m%d"),
              "limit": WAYBACK_LIMIT, "collapse": "urlkey"}
    try:
        async with session.get(
            "http://web.archive.org/cdx/search/cdx",
            params=params, ssl=False,
            timeout=aiohttp.ClientTimeout(total=WAYBACK_TIMEOUT)
        ) as r:
            if r.status != 200: return [], True
            data = await r.json(content_type=None)
    except asyncio.TimeoutError:
        return [], False
    except Exception as e:
        log.debug(f"Wayback CDX {domain}: {e}"); return [], False
    if not data or len(data) < 2: return [], True
    results = []
    for row in data[1:]:
        try:
            ts, url = row[0], row[1]
            ts_str = (f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
                      f"T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}")
            results.append((url, ts_str))
        except Exception: continue
    return results, True

async def scrape_wayback(seen_urls, ckpt, executor):
    all_rows = []
    conn = aiohttp.TCPConnector(limit=2, ssl=False)
    hdrs = {"User-Agent": "Mozilla/5.0 (academic research; NLP project)"}
    async with aiohttp.ClientSession(connector=conn, headers=hdrs) as session:
        sem  = asyncio.Semaphore(CONCURRENCY)
        pbar = tqdm(ALL_WINDOWS, desc="Wayback windows", unit="win", ncols=80)
        for w0, w1 in pbar:
            wk = f"wb_win_{w0}_{w1}"
            if ckpt.get(wk) and ckpt.get(f"{wk}_rows", -1) >= 0: continue
            pbar.set_postfix_str(f"{w0}→{w1}")
            all_entries = []; all_success = True
            for target in WAYBACK_TARGETS:
                entries, ok = await _wayback_cdx_query(
                    session, target["domain"], target["path"], w0, w1)
                all_entries.extend(entries)
                if not ok: all_success = False
                await asyncio.sleep(1.0)
            seen_batch = set()
            to_fetch = []
            for url, ts_str in all_entries:
                if url in seen_urls or url in seen_batch: continue
                seen_batch.add(url); to_fetch.append((url, ts_str))
            log.info(f"Wayback {w0}→{w1}: {len(all_entries)} CDX, "
                     f"{len(to_fetch)} unique new")
            if not to_fetch:
                if all_success:
                    ckpt[wk] = True; ckpt[f"{wk}_rows"] = 0
                    save_checkpoint(ckpt)
                continue
            results  = await fetch_batch_aio(
                session, [u for u,_ in to_fetch], f"WB {w0}→{w1}", sem)
            html_map = {u: h for u, h in results}
            parsed = []
            for url, ts_str in to_fetch:
                html = html_map.get(url,"")
                if html:
                    p = parse_article_html((url, html, "ET/MC/BS"))
                    if p: parsed.append(p); continue
                slug = re.sub(r"[-_]", " ",
                              url.rstrip("/").split("/")[-1].split(".")[0])
                slug = re.sub(r"\d{5,}", "", slug).strip()
                if len(slug) > 10:
                    dt = _parse_dt(ts_str)
                    if dt:
                        parsed.append({"title": slug, "news": slug,
                                       "published_at": dt,
                                       "source": "Wayback", "url": url})
            rows = _make_rows(parsed, seen_urls)
            append_rows(rows); all_rows.extend(rows)
            if all_success:
                ckpt[wk] = True; ckpt[f"{wk}_rows"] = len(rows)
                save_checkpoint(ckpt)
            log.info(f"Wayback {w0}→{w1}: saved {len(rows)} (total: {len(all_rows)})")
            await asyncio.sleep(2.0)
        pbar.close()
    return all_rows

async def diagnose():
    print("\n" + "="*60)
    print("DIAGNOSE — checking sources reachable from your machine")
    print("="*60 + "\n")
    tests = [
        ("ET Markets RSS", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", True),
        ("ET Stocks RSS",  "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", True),
        ("NDTV RSS",       "https://feeds.feedburner.com/ndtvprofit-latest", True),
        ("ET sitemap",     "https://economictimes.indiatimes.com/etstatic/sitemaps/et/news/sitemap-index.xml", False),
        ("Wayback CDX MC", "http://web.archive.org/cdx/search/cdx?url=moneycontrol.com/news/business/stocks/*&output=json&limit=3&fl=timestamp,original&from=20230101&to=20230110", False),
        ("Wayback CDX ET", "http://web.archive.org/cdx/search/cdx?url=economictimes.indiatimes.com/markets/stocks/news/*&output=json&limit=3&fl=timestamp,original&from=20230101&to=20230110", False),
        ("Wayback CDX BS", "http://web.archive.org/cdx/search/cdx?url=business-standard.com/markets/news/*&output=json&limit=3&fl=timestamp,original&from=20230101&to=20230110", False),
    ]
    conn = aiohttp.TCPConnector(ssl=False)
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0) Chrome/122.0"}
    async with aiohttp.ClientSession(connector=conn, headers=hdrs) as s:
        for name, url, is_rss in tests:
            try:
                async with s.get(url, ssl=False,
                                 timeout=aiohttp.ClientTimeout(total=20)) as r:
                    text    = await r.text(encoding="utf-8", errors="replace")
                    entries = _parse_rss(text) if is_rss else []
                    print(f"  {'✓' if r.status==200 else '✗'}  "
                          f"HTTP {r.status}  {len(text):>9,}B  "
                          f"{len(entries):3}e  {name}")
            except asyncio.TimeoutError:
                print(f"  ✗  TIMEOUT  {name}")
            except Exception as e:
                print(f"  ✗  ERR  {name}: {str(e)[:55]}")
    print()

async def run_once() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt      = load_checkpoint()
    ckpt      = reset_failed_wayback_windows(ckpt)
    seen_urls = load_seen_urls()
    log.info(f"English scraper v3 | {STOP_DATE} → {END_DATE} "
             f"| {len(ALL_WINDOWS)} × {BATCH_MONTHS}-month windows")
    log.info(f"Platform: {sys.platform} | Seen URLs: {len(seen_urls):,}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARSE_WORKERS) as executor:
        rss_rows = await scrape_rss_feeds(seen_urls, ckpt, executor)
        et_rows  = await scrape_et_sitemap(seen_urls, ckpt, executor)
        wb_rows  = await scrape_wayback(seen_urls, ckpt, executor)
    total   = len(rss_rows) + len(et_rows) + len(wb_rows)
    elapsed = time.time() - t0
    log.info(f"Run complete in {elapsed/60:.1f} min | New rows: {total:,}")
    log.info(f"  RSS    : {len(rss_rows):,}")
    log.info(f"  ET SM  : {len(et_rows):,}")
    log.info(f"  Wayback: {len(wb_rows):,}")
    return total

async def main(diagnose_only=False, loop_n=1, gap_s=60):
    if diagnose_only:
        await diagnose(); return
    grand_total = 0
    for i in range(1, loop_n + 1):
        log.info(f"\n{'━'*55}")
        log.info(f"ITERATION {i}/{loop_n}")
        log.info(f"{'━'*55}")
        try:
            new_rows = await run_once()
            grand_total += new_rows
        except KeyboardInterrupt:
            log.info("Interrupted — checkpoint saved, safe to resume"); break
        except Exception as e:
            log.error(f"Iteration {i} failed: {e}")
        if OUT_CSV.exists():
            df = pd.read_csv(OUT_CSV)
            log.info(f"\nCSV: {df.shape[0]:,} rows | "
                     f"{df['ticker'].nunique()} tickers | "
                     f"{df['date'].min()} → {df['date'].max()}")
            log.info(f"Sources:\n{df['source'].value_counts().to_string()}")
        if i < loop_n and new_rows == 0:
            log.info("No new rows — sources exhausted. Stopping."); break
        if i < loop_n:
            log.info(f"Waiting {gap_s}s… (Ctrl+C to stop)")
            try: await asyncio.sleep(gap_s)
            except KeyboardInterrupt: break
    log.info(f"\n{'='*55}")
    log.info(f"Grand total new rows: {grand_total:,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--loop", type=int, default=1, metavar="N")
    parser.add_argument("--gap",  type=int, default=60, metavar="SECONDS")
    args = parser.parse_args()
    asyncio.run(main(diagnose_only=args.diagnose,
                     loop_n=args.loop, gap_s=args.gap))
