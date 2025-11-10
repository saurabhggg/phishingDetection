#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Crawler (CSE lookalikes + non-resemblance phishing)
-----------------------------------------------------------

Finds candidate domains from:
  1) Certificate Transparency logs (crt.sh) – for CSE keywords & tunneling namespaces
  2) Search queries (DDG HTML or Bing API – optional) – for tunneling/free-hosting phishing
  3) Public feeds (URLHaus / PhishTank – optional) – disabled by default

Writes:
  fetched_domains.csv  (append-safe; deduped)
  crawler/unified_cache.json  (resumable)
  crawler/unified_log.txt     (progress log)

Columns:
  Domain, Source, First_Seen_ISO, Notes

Usage examples:
  python crawler_unified.py
  python crawler_unified.py --search ddg
  python crawler_unified.py --search bing --bing-key YOUR_KEY
  python crawler_unified.py --enable-urlhaus --enable-phishtank

Notes:
- DDG HTML scraping is best-effort and may break if layout changes. Bing API is robust but requires a key.
- If you enable feeds, remember to declare them in your submission declarations.
"""

import asyncio, aiohttp, json, re, time, csv, sys, argparse
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse

import pandas as pd

# ---- local utils (adjust import path if needed)
try:
    from utils.logger_config import setup_logger
    from utils.cache_handler import load_json, save_json
except Exception:
    # Minimal fallback logger if utils not available
    def setup_logger(path, rotation="10 MB"):
        class _L:
            def info(self, m):  print(m, flush=True)
            def warning(self, m): print(m, flush=True)
            def success(self, m): print(m, flush=True)
            def error(self, m): print(m, flush=True)
        return _L()
    def load_json(path, default): 
        p=Path(path); 
        return json.loads(p.read_text()) if p.exists() else default
    def save_json(path, data): Path(path).write_text(json.dumps(data, indent=2))

# -------------------------------
# Paths & logging
# -------------------------------
OUT_FILE  = Path("fetched_domains.csv")
CACHE_FILE= Path("crawler/unified_cache.json")
LOG_FILE  = Path("crawler/unified_log.txt")
LOG       = setup_logger(LOG_FILE)

# -------------------------------
# HTTP defaults
# -------------------------------
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept": "*/*"}
TIMEOUT = aiohttp.ClientTimeout(total=15, connect=6, sock_read=12)

# -------------------------------
# CSE definitions (keywords/official domains)
# -------------------------------
CSE_DEFS = {
    "SBI": {
        "official_domains": ["onlinesbi.sbi", "sbi.co.in", "sbicard.com", "sbilife.co.in", "sbiepay.sbi"],
        "keywords": ["sbi", "sbicard", "yonobusiness", "onlinesbi", "sbiepay", "sbilife"],
    },
    "ICICI": {
        "official_domains": ["icicibank.com", "icicidirect.com", "iciciprulife.com", "icicilombard.com"],
        "keywords": ["icici", "icicidirect", "iciciprulife", "icicibank", "icicilombard", "icicicareers"],
    },
    "HDFC": {
        "official_domains": ["hdfcbank.com", "hdfclife.com", "hdfcergo.com"],
        "keywords": ["hdfc", "hdfcbank", "hdfclife", "hdfcergo"],
    },
    "PNB": {
        "official_domains": ["pnbindia.in", "netpnb.com"],
        "keywords": ["pnb", "pnbindia", "netpnb"],
    },
    "BoB": {
        "official_domains": ["bankofbaroda.in", "bankofbaroda.com", "bobibanking.com"],
        "keywords": ["bankofbaroda", "bobibanking"],
    },
    "NIC": {
        "official_domains": ["nic.in", "gov.in", "kavach.gov.in"],
        "keywords": ["nic", "gov.in", "mgovcloud", "kavach"],
    },
    "IRCTC": {
        "official_domains": ["irctc.co.in"],
        "keywords": ["irctc"],
    },
    "Airtel": {
        "official_domains": ["airtel.in"],
        "keywords": ["airtel"],
    },
    "IOCL": {
        "official_domains": ["iocl.com", "indianoil.in"],
        "keywords": ["iocl", "indianoil"],
    },
}

# -------------------------------
# Namespaces / hosts that often carry non-resemblance phishing
# -------------------------------
TUNNEL_NAMESPACES = [
    "ngrok.io","vercel.app","netlify.app","herokuapp.com","glitch.me","repl.co",
    "firebaseapp.com","web.app","pages.dev","trycloudflare.com","surge.sh","render.com",
    "github.io","cloudfront.net","000webhostapp.com","weebly.com","wixsite.com",
    "webflow.io","appspot.com","azurewebsites.net"
]

# Search phrases designed to catch phishing-y landing pages on these hosts
SEARCH_PHRASES = [
    "login", "secure", "verify", "update kyc", "account verification",
    "bank login", "document upload", "otp verification"
]

# -------------------------------
# crt.sh helpers
# -------------------------------
CRT_URL = "https://crt.sh/?q={q}&output=json"

async def crt_fetch(session, query: str) -> set[str]:
    """
    Fetch domains from crt.sh for a query. Query may include wildcards (e.g. %.ngrok.io).
    """
    try:
        async with session.get(CRT_URL.format(q=query), headers=HEADERS, timeout=TIMEOUT) as r:
            if r.status != 200:
                return set()
            data = await r.json(content_type=None)
    except Exception:
        return set()
    out = set()
    for row in data:
        name = str(row.get("name_value","")).strip().lower()
        if not name:
            continue
        for p in name.split("\n"):
            p = p.strip()
            if p and not p.startswith("*"):
                # keep host part only
                host = p.split("/")[0]
                out.add(host)
    return out

# -------------------------------
# Search providers
# -------------------------------
async def ddg_search_domains(session, query: str, max_results=20) -> set[str]:
    """
    Very light HTML scrape from DuckDuckGo. Best-effort only.
    """
    base = "https://duckduckgo.com/html/"
    params = {"q": query, "kl": "in-en"}
    try:
        async with session.post(base, headers=HEADERS, data=params, timeout=TIMEOUT) as r:
            if r.status != 200:
                return set()
            html = await r.text()
    except Exception:
        return set()

    # extract links
    doms = set()
    for m in re.finditer(r'href="(https?://[^"]+)"', html):
        try:
            url = m.group(1)
            host = urlparse(url).netloc.lower()
            if host:
                doms.add(host)
        except Exception:
            pass
    return set(list(doms)[:max_results])

async def bing_search_domains(session, query: str, bing_key: str, max_results=20) -> set[str]:
    """
    Bing Web Search API (requires key). More reliable than HTML scraping.
    """
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    params = {"q": query, "count": max_results}
    headers = dict(HEADERS)
    headers["Ocp-Apim-Subscription-Key"] = bing_key
    try:
        async with session.get(endpoint, headers=headers, params=params, timeout=TIMEOUT) as r:
            if r.status != 200:
                return set()
            data = await r.json()
    except Exception:
        return set()

    doms = set()
    for section in ("webPages","news","images","videos"):
        items = (data.get(section) or {}).get("value") or []
        for it in items:
            try:
                url = it.get("url") or it.get("contentUrl") or ""
                host = urlparse(url).netloc.lower()
                if host:
                    doms.add(host)
            except Exception:
                pass
    return doms

# -------------------------------
# Public feeds (optional)
# -------------------------------
async def urlhaus_fetch(session, limit=10000) -> set[str]:
    """
    URLHaus online payloads (public CSV). Disabled by default.
    """
    url = "https://urlhaus.abuse.ch/downloads/csv_recent/"
    try:
        async with session.get(url, headers=HEADERS, timeout=TIMEOUT) as r:
            if r.status != 200:
                return set()
            text = await r.text()
    except Exception:
        return set()

    doms = set()
    for line in text.splitlines():
        if line.startswith("#") or "," not in line:
            continue
        parts = line.split(",")
        if parts:
            # URL column is usually the first after timestamp; be defensive
            for p in parts:
                if p.startswith("http"):
                    try:
                        host = urlparse(p).netloc.lower()
                        if host:
                            doms.add(host)
                    except Exception:
                        pass
                    break
    return set(list(doms)[:limit])

async def phishtank_fetch(session, limit=10000) -> set[str]:
    """
    PhishTank online-valid (JSON). Disabled by default.
    """
    url = "https://data.phishtank.com/data/online-valid.json"
    try:
        async with session.get(url, headers=HEADERS, timeout=TIMEOUT) as r:
            if r.status != 200:
                return set()
            data = await r.json()
    except Exception:
        return set()

    doms = set()
    for row in data:
        try:
            url = row.get("url","")
            host = urlparse(url).netloc.lower()
            if host:
                doms.add(host)
        except Exception:
            pass
        if len(doms) >= limit:
            break
    return doms

# -------------------------------
# Helpers
# -------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def resembles_tunnel(host: str) -> bool:
    return any(host.endswith("." + ns) or host == ns for ns in TUNNEL_NAMESPACES)

def annotate(host: str) -> str:
    notes = []
    if resembles_tunnel(host):
        notes.append("tunnel/free-host")
    # quick path-keyword signal (best-effort; we only have host here)
    return ";".join(notes)

def ensure_out_header(path: Path):
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Domain","Source","First_Seen_ISO","Notes"])

async def main(args):
    LOG.info("▶ Unified crawler starting...")
    ensure_out_header(OUT_FILE)

    cache = load_json(CACHE_FILE, {})
    seen = set(cache.get("seen_hosts", []))

    # load existing CSV to avoid duplicates across runs
    if OUT_FILE.exists():
        try:
            df_prev = pd.read_csv(OUT_FILE)
            seen.update(df_prev["Domain"].astype(str).str.lower().tolist())
            LOG.info(f"⏩ Resuming with {len(seen)} pre-seen domains.")
        except Exception:
            pass

    connector = aiohttp.TCPConnector(limit=args.concurrency, ssl=False, ttl_dns_cache=300)
    async with aiohttp.ClientSession(headers=HEADERS, timeout=TIMEOUT, connector=connector) as session:
        new_rows = []
        total_new = 0

        # 1) crt.sh for CSE keywords (lookalikes) ------------------------------
        cse_queries = []
        for cse, meta in CSE_DEFS.items():
            cse_queries.extend(meta["keywords"])
            cse_queries.extend(meta["official_domains"])
        cse_queries = sorted(set(q.lower() for q in cse_queries))

        LOG.info(f"🔎 crt.sh CSE queries: {len(cse_queries)}")
        for q in cse_queries:
            doms = await crt_fetch(session, q)
            added = 0
            for d in doms:
                host = d.lower()
                if host in seen: 
                    continue
                seen.add(host)
                new_rows.append([host, f"crt.sh:{q}", now_iso(), annotate(host)])
                total_new += 1
                added += 1
            LOG.info(f"  +{added:5d}  (q='{q}')")
            if total_new and total_new % args.save_interval == 0:
                pd.DataFrame(new_rows, columns=["Domain","Source","First_Seen_ISO","Notes"]).to_csv(
                    OUT_FILE, mode="a", header=False, index=False
                )
                new_rows.clear()
                cache["seen_hosts"] = sorted(list(seen))
                save_json(CACHE_FILE, cache)
                LOG.info(f"💾 checkpoint: total_new={total_new}, seen={len(seen)}")

        # 2) crt.sh for tunneling namespaces (non-resemblance) ----------------
        tunnel_queries = [f"%.{ns}" for ns in TUNNEL_NAMESPACES]
        LOG.info(f"🔎 crt.sh tunnel queries: {len(tunnel_queries)}")
        for q in tunnel_queries:
            doms = await crt_fetch(session, q)
            added = 0
            for d in doms:
                host = d.lower()
                if host in seen: 
                    continue
                seen.add(host)
                new_rows.append([host, f"crt.sh:{q}", now_iso(), annotate(host)])
                total_new += 1
                added += 1
            LOG.info(f"  +{added:5d}  (q='{q}')")
            if total_new and total_new % args.save_interval == 0:
                pd.DataFrame(new_rows, columns=["Domain","Source","First_Seen_ISO","Notes"]).to_csv(
                    OUT_FILE, mode="a", header=False, index=False
                )
                new_rows.clear()
                cache["seen_hosts"] = sorted(list(seen))
                save_json(CACHE_FILE, cache)
                LOG.info(f"💾 checkpoint: total_new={total_new}, seen={len(seen)}")

        # 3) Search queries for suspicious pages on tunnels --------------------
        if args.search in ("ddg", "bing"):
            LOG.info(f"🔎 search provider: {args.search}")
            for ns in TUNNEL_NAMESPACES:
                for phrase in SEARCH_PHRASES:
                    q = f'site:{ns} "{phrase}"'
                    if args.search == "ddg":
                        doms = await ddg_search_domains(session, q, max_results=args.search_count)
                    else:
                        doms = await bing_search_domains(session, q, args.bing_key, max_results=args.search_count)
                    added = 0
                    for host in doms:
                        if host in seen:
                            continue
                        seen.add(host)
                        new_rows.append([host, f"{args.search}:{q}", now_iso(), annotate(host)])
                        total_new += 1
                        added += 1
                    LOG.info(f"  +{added:3d}  (q='{q}')")
                    if total_new and total_new % args.save_interval == 0:
                        pd.DataFrame(new_rows, columns=["Domain","Source","First_Seen_ISO","Notes"]).to_csv(
                            OUT_FILE, mode="a", header=False, index=False
                        )
                        new_rows.clear()
                        cache["seen_hosts"] = sorted(list(seen))
                        save_json(CACHE_FILE, cache)
                        LOG.info(f"💾 checkpoint: total_new={total_new}, seen={len(seen)}")

        # 4) Public feeds (optional; disabled by default) ----------------------
        if args.enable_urlhaus:
            LOG.info("🔎 URLHaus (recent CSV)")
            doms = await urlhaus_fetch(session)
            added = 0
            for host in doms:
                if host in seen:
                    continue
                seen.add(host)
                new_rows.append([host, "feed:urlhaus", now_iso(), annotate(host)])
                total_new += 1
                added += 1
            LOG.info(f"  +{added:5d} from URLHaus")

        if args.enable_phishtank:
            LOG.info("🔎 PhishTank (online-valid)")
            doms = await phishtank_fetch(session)
            added = 0
            for host in doms:
                if host in seen:
                    continue
                seen.add(host)
                new_rows.append([host, "feed:phishtank", now_iso(), annotate(host)])
                total_new += 1
                added += 1
            LOG.info(f"  +{added:5d} from PhishTank")

        # final flush
        if new_rows:
            pd.DataFrame(new_rows, columns=["Domain","Source","First_Seen_ISO","Notes"]).to_csv(
                OUT_FILE, mode="a", header=False, index=False
            )
        cache["seen_hosts"] = sorted(list(seen))
        save_json(CACHE_FILE, cache)
        LOG.success(f"✅ done: total_new={total_new}, total_seen={len(seen)}, wrote → {OUT_FILE}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Unified suspicious domain crawler")
    ap.add_argument("--concurrency", type=int, default=40, help="HTTP concurrency (default 40)")
    ap.add_argument("--save-interval", type=int, default=1000, help="Rows per checkpoint (default 1000)")
    ap.add_argument("--search", choices=["none","ddg","bing"], default="none",
                    help="Enable search provider (default: none)")
    ap.add_argument("--search-count", type=int, default=20, help="Max results per query (default 20)")
    ap.add_argument("--bing-key", type=str, default="", help="Bing Web Search API key (if --search=bing)")
    ap.add_argument("--enable-urlhaus", action="store_true", help="Include URLHaus feed (optional)")
    ap.add_argument("--enable-phishtank", action="store_true", help="Include PhishTank feed (optional)")
    args = ap.parse_args()

    if args.search == "bing" and not args.bing_key:
        print("ERROR: --bing-key is required when --search=bing", file=sys.stderr)
        sys.exit(2)

    asyncio.run(main(args))
