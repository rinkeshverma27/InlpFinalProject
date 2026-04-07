# 🚀 Nifty 50 News Context Scraper (Distributed Master Edition)

This project is a high-security, distributed pipeline designed to scrape full article context for 16,000+ Nifty 50 news articles while bypassing advanced anti-bot protections (Akamai/Cloudflare).

## 🛡️ Industrial-Grade Safeguards
- **TLS/JA3 Impersonation**: Uses `curl_cffi` to mimic a real Chrome 110 browser fingerprint.
- **Exponential Backoff**: Automatically waits longer if a "403 Forbidden" is detected.
- **Circuit Breaker**: Automatically stops the script if it detects a hard block (2 consecutive 403s).
- **User-Agent Rotation**: Randomly switches identities to avoid fingerprinting.
- **Session Warmup**: Seeds legitimate cookies before starting the scrape.

---

## 📋 Setup Instructions (Run on all 3 Systems)

1. **Install Dependencies**:
   ```bash
   pip install curl_cffi newspaper4k pandas tqdm
   ```

2. **Copy Files**: Ensure the following files are in the same folder on each machine:
   - `content_scraper.py`
   - `nifty50_news_2020_2026.csv` (The source database)

---

## 🕹️ Distributed Execution Guide

Run the corresponding command on each machine using a **unique internet connection** (e.g., Home WiFi, Mobile Hotspot A, Mobile Hotspot B).

| System | Assigned Command | Output File |
| :--- | :--- | :--- |
| **System 1** | `python3 content_scraper.py --system 1` | `nifty_context_part1.csv` |
| **System 2** | `python3 content_scraper.py --system 2` | `nifty_context_part2.csv` |
| **System 3** | `python3 content_scraper.py --system 3` | `nifty_context_part3.csv` |

> [!CAUTION]
> **Don't Rush**: This script is intentionally slow (15-30s delay). This is the only way to avoid a permanent IP ban. Each machine will take approximately **30 hours** to finish its portion.

---

## 🔗 Merging the Data

Once all 3 machines have finished, move the three `nifty_context_partX.csv` files to your main machine and run:
```bash
python3 merge_distributed.py
```

---

## 🛠️ Recovery & Status
- **If blocked (403)**: The script will wait and retry. If it hits 2 consecutive blocks, it will exit.
- **To Resume**: Simply run the same command again. The script will automatically detect what you've already finished and skip it.
- **Tracking**: Check `recovery_status.md` for historical block details.
