#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cse_relevance_heavy_v2.py – Enhanced Heavy Relevance Classifier
================================================================

Performs deep analysis on domains shortlisted from the prefilter/light classifier.

🔹 Input:
    - CSV with `Domain` column (e.g. prefilter_filtered.csv)
    - Or prefilter output with `CSE_Score` column (use `--prefilter`)

🔹 Output:
    - CSE_Relevance_Output_v2.csv (detailed columns: semantic, favicon, RDAP, etc.)

🔹 Features:
    - HTML parsing & phishing keyword counts
    - Semantic similarity to official CSEs
    - Favicon perceptual hash matching
    - RDAP org/registrar info
    - IP resolution
    - Weighted CSE relevance scoring

Usage:
    python cse_relevance_heavy_v2.py --input prefilter_filtered.csv --prefilter --min-score 0.45 --concurrency 80

Dependencies:
    pip install aiohttp aiohttp-socks pandas sentence-transformers pillow imagehash tldextract requests beautifulsoup4
"""

import argparse, asyncio, json, os, re, io, socket
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import pandas as pd
import aiohttp
from bs4 import BeautifulSoup
import tldextract
from PIL import Image
import imagehash
import requests

# Optional heavy model usage (if you want to run trained phishing detector on shortlisted)
try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
OUT_CSV = "CSE_Relevance_Output_v2.csv"
CACHE_JSON = Path("crawler/cse_relevance_cache_v2.json")
LOG_INTERVAL = 200
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=12, connect=6, sock_read=8)

CSE_DEFS = {
    "SBI": {"official_domains": ["onlinesbi.sbi", "sbi.co.in", "sbicard.com","sbilife.co.in","sbiepay.sbi"],
            "keywords": ["sbi","sbicard","onlinesbi","sbilife"]},
    "ICICI": {"official_domains": ["icicibank.com","icicidirect.com","iciciprulife.com","icicilombard.com"],
              "keywords": ["icici","icicibank","iciciprulife"]},
    "HDFC": {"official_domains": ["hdfcbank.com","hdfclife.com","hdfcergo.com"],
             "keywords": ["hdfc","hdfcbank"]},
    "PNB": {"official_domains": ["pnbindia.in","netpnb.com"], "keywords": ["pnb","pnbindia"]},
    "BoB": {"official_domains": ["bankofbaroda.in","bankofbaroda.com"], "keywords": ["bankofbaroda"]},
    "NIC": {"official_domains": ["nic.in","gov.in"], "keywords": ["nic","gov.in"]},
    "IRCTC": {"official_domains": ["irctc.co.in"], "keywords": ["irctc"]},
    "Airtel": {"official_domains": ["airtel.in"], "keywords": ["airtel"]},
    "IOCL": {"official_domains": ["iocl.com","indianoil.in"], "keywords": ["iocl","indianoil"]}
}

# ---------------------------------------------------------------------
# Semantic model
# ---------------------------------------------------------------------
SEM_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
_sem_model = None
_cse_names = list(CSE_DEFS.keys())
_cse_embeds = None

def ensure_semantic():
    global _sem_model, _cse_embeds
    if _sem_model is None:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim
        _sem_model = SentenceTransformer(SEM_MODEL_NAME)
        _cse_embeds = _sem_model.encode(_cse_names, convert_to_tensor=True, normalize_embeddings=True)

def semantic_match(text):
    try:
        from sentence_transformers import util
        ensure_semantic()
        emb = _sem_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(emb, _cse_embeds)[0]
        best_idx = int(sims.argmax())
        return _cse_names[best_idx], float(sims[best_idx])
    except Exception:
        return None, 0.0


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def safe_host(h):
    if not h:
        return ""
    s = str(h).strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = s.split("/")[0]
    return s

def load_cache():
    if CACHE_JSON.exists():
        try:
            return json.loads(CACHE_JSON.read_text())
        except Exception:
            return {}
    return {}

def save_cache(d):
    try:
        CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
        CACHE_JSON.write_text(json.dumps(d, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------
# Async Fetchers
# ---------------------------------------------------------------------
async def fetch_html(session, domain):
    for scheme in ("https","http"):
        url = f"{scheme}://{domain}"
        try:
            async with session.get(url, timeout=DEFAULT_TIMEOUT, allow_redirects=True) as r:
                status = r.status
                ctype = r.headers.get("Content-Type","").lower()
                txt = await r.text(errors="ignore") if "text/html" in ctype else ""
                return status, str(r.url), txt
        except Exception:
            continue
    return 0, "", ""

async def fetch_bytes(session, url):
    try:
        async with session.get(url, timeout=DEFAULT_TIMEOUT) as r:
            if r.status == 200:
                return await r.read()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------
# RDAP Lookup
# ---------------------------------------------------------------------
def rdap_fetch(domain):
    out = {"creation_date": "", "registrar": "", "org": "", "nameservers":[]}
    try:
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=8)
        if r.status_code != 200:
            return out
        data = r.json()
        for e in data.get("events", []):
            if e.get("eventAction") == "registration":
                out["creation_date"] = e.get("eventDate")
                break
        out["nameservers"] = [ns.get("ldhName") for ns in data.get("nameservers", []) if isinstance(ns, dict)]
        for ent in data.get("entities", []):
            roles = ent.get("roles", [])
            if "registrar" in roles and not out["registrar"]:
                out["registrar"] = ent.get("handle") or ""
            if any(r in roles for r in ["registrant","administrative","technical"]) and not out["org"]:
                vca = ent.get("vcardArray", [])
                if isinstance(vca, list) and len(vca)==2:
                    for row in vca[1]:
                        if row[0] == "org":
                            out["org"] = row[3]
                            break
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------
# Favicon Hash Library Builder
# ---------------------------------------------------------------------
async def build_official_fav_lib(session):
    lib = {}
    for cse, meta in CSE_DEFS.items():
        lib[cse] = []
        for off in meta["official_domains"]:
            for scheme in ("https","http"):
                try:
                    url = f"{scheme}://{off}/favicon.ico"
                    async with session.get(url, timeout=DEFAULT_TIMEOUT) as r:
                        if r.status==200:
                            b = await r.read()
                            ph = str(imagehash.phash(Image.open(io.BytesIO(b)).convert("RGB")))
                            if ph not in lib[cse]:
                                lib[cse].append(ph)
                except Exception:
                    continue
    return lib


# ---------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------
async def analyze_one(session, domain, fav_lib):
    d = safe_host(domain)
    if not d:
        return None

    # Fetch HTML
    status, final, html = await fetch_html(session, d)
    if status >= 400 or not html or len(html) < 400:
        return {"Domain": d, "Status": "inactive", "HTTP_Status": status}

    try:
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else ""
    except Exception:
        title = ""

    # URL-level keyword match
    low = d.lower()
    url_cse = None
    for cse, meta in CSE_DEFS.items():
        if any(off in low for off in meta["official_domains"]):
            url_cse = cse
        if any(kw in low for kw in meta["keywords"]) and not url_cse:
            url_cse = cse

    # Semantic similarity
    sem_cse, sem_score = semantic_match(low + " " + title)

    # Content keyword analysis
    text = " ".join([t.get_text(" ", strip=True) for t in soup.find_all(["h1","h2","p"])])
    content_text = (title + " " + text).lower()

    cse_kw_count = sum(kw in content_text for cse in CSE_DEFS.values() for kw in cse["keywords"])
    phish_kw_count = sum(kw in content_text for kw in ["login","verify","otp","password","netbanking","cvv","account","card","pin","update"])

    # Favicon
    fav_ph = None
    fav_url = ""
    try:
        link = soup.find("link", rel=lambda v: v and "icon" in v.lower())
        if link and link.get("href"):
            fav_url = urljoin(final or f"https://{d}", link["href"])
        else:
            fav_url = f"https://{d}/favicon.ico"
        fav_bytes = await fetch_bytes(session, fav_url)
        if fav_bytes:
            fav_ph = str(imagehash.phash(Image.open(io.BytesIO(fav_bytes)).convert("RGB")))
    except Exception:
        fav_ph = None

    # Favicon similarity
    fav_hit = (None, 999)
    if fav_ph:
        best_dist, best_cse = 999, None
        for cse, hashes in fav_lib.items():
            for h in hashes:
                try:
                    hd = imagehash.hex_to_hash(fav_ph) - imagehash.hex_to_hash(h)
                    if hd < best_dist:
                        best_dist, best_cse = hd, cse
                except Exception:
                    continue
        fav_hit = (best_cse, best_dist)

    # RDAP + IPs
    rd = rdap_fetch(d)
    try:
        _,_,ips = socket.gethostbyname_ex(d)
        host_ips = ",".join(ips)
    except Exception:
        host_ips = ""

    # Weight combination
    weights = {}
    def add_w(c, w):
        if c:
            weights[c] = weights.get(c, 0.0) + w

    if url_cse: add_w(url_cse, 1.0)
    if sem_cse and sem_score >= 0.65: add_w(sem_cse, sem_score)
    if cse_kw_count: add_w(sem_cse or url_cse, 0.5)
    if fav_hit[0] and fav_hit[1] <= 6: add_w(fav_hit[0], 1.2)
    if rd.get("org") and any(kw in rd["org"].lower() for meta in CSE_DEFS.values() for kw in meta["keywords"]):
        add_w(sem_cse or url_cse, 1.0)

    if not weights:
        return {"Domain": d, "Status": "no_relation", "HTTP_Status": status}

    best_cse = max(weights, key=weights.get)
    score = float(weights[best_cse])

    return {
        "Domain": d,
        "Related_CSE": best_cse,
        "Relation_Score": round(score, 3),
        "Match_Types": ",".join([k for k in ["URL","SEM","CONTENT","FAV","RDAP"]]),
        "HTTP_Status": status,
        "Title": title,
        "Favicon_URL": fav_url,
        "Favicon_HD": fav_hit[1] if isinstance(fav_hit[1], int) else 999,
        "RDAP_Org": rd.get("org",""),
        "RDAP_Registrar": rd.get("registrar",""),
        "Domain_Creation_Date": rd.get("creation_date",""),
        "Hosting_IPs": host_ips,
        "Content_CSE_Keyword_Count": cse_kw_count,
        "Content_Phish_Keyword_Count": phish_kw_count,
        "Semantic_Score": sem_score,
        "Status": "related"
    }


# ---------------------------------------------------------------------
# Main Async Runner
# ---------------------------------------------------------------------
async def run(input_csv, min_score, concurrency, use_prefilter):
    df = pd.read_csv(input_csv)

    if use_prefilter:
        # Detect correct score column
        score_col = next((c for c in ["CSE_Score", "score", "Score"] if c in df.columns), None)
        if not score_col:
            raise ValueError("❌ Prefilter file must contain 'CSE_Score' or 'score' column.")
        before = len(df)
        df = df[df[score_col] >= min_score]
        print(f"✅ Prefilter applied: kept {len(df)} of {before} domains (≥ {min_score})")

    # Extract domain list
    if "Domain" not in df.columns:
        domain_col = next((c for c in df.columns if "domain" in c.lower() or "url" in c.lower()), None)
        if not domain_col:
            raise ValueError("❌ Input missing Domain column.")
        df["Domain"] = df[domain_col]

    domains = df["Domain"].astype(str).str.lower().dropna().unique().tolist()
    print(f"🚀 Starting deep analysis on {len(domains):,} domains from {input_csv}")

    cache = load_cache()
    done = set()
    if Path(OUT_CSV).exists():
        try:
            prev = pd.read_csv(OUT_CSV)
            done = set(prev["Domain"].astype(str).str.lower())
            print(f"Resuming: {len(done)} already processed.")
        except Exception:
            done = set()

    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False)
    async with aiohttp.ClientSession(connector=connector, timeout=DEFAULT_TIMEOUT) as session:
        fav_lib = await build_official_fav_lib(session)
        print(f"🎨 Loaded official favicons: { {k: len(v) for k, v in fav_lib.items()} }")

        sem = asyncio.Semaphore(concurrency)
        out_rows, processed = [], 0

        async def worker(dom):
            async with sem:
                if dom in done:
                    return None
                return await analyze_one(session, dom, fav_lib)

        batch = []
        for d in domains:
            batch.append(asyncio.create_task(worker(d)))
            if len(batch) >= concurrency * 2:
                for t in asyncio.as_completed(batch):
                    r = await t
                    processed += 1
                    if r:
                        out_rows.append(r)
                    if processed % LOG_INTERVAL == 0:
                        pd.DataFrame(out_rows).to_csv(OUT_CSV, mode="a", header=not Path(OUT_CSV).exists(), index=False)
                        save_cache(cache)
                        print(f"💾 checkpoint: processed {processed}/{len(domains)}")
                        out_rows = []
                batch = []

        # Drain remaining
        for t in asyncio.as_completed(batch):
            r = await t
            if r:
                out_rows.append(r)

        if out_rows:
            pd.DataFrame(out_rows).to_csv(OUT_CSV, mode="a", header=not Path(OUT_CSV).exists(), index=False)
        save_cache(cache)

    print(f"✅ Done. Results written to {OUT_CSV}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV (prefilter output or domain list)")
    ap.add_argument("--min-score", type=float, default=0.45, help="Minimum score if prefilter used")
    ap.add_argument("--concurrency", type=int, default=80, help="HTTP concurrency")
    ap.add_argument("--prefilter", action="store_true", help="Input has CSE_Score/score column")
    args = ap.parse_args()
    asyncio.run(run(args.input, args.min_score, args.concurrency, args.prefilter))
