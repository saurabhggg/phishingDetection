#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Extractor – EXTENDED + Caching + Resume
===============================================
Extracts advanced lexical, domain, HTML content, and favicon features from
both whitelisted and phishing URL datasets (.xlsx). 

✔ Lexical + Domain-level (your full list)
✔ HTML content analysis
✔ Favicon similarity to official CSE favicons
✔ CSE keyword presence
✔ Caching (resumes from last checkpoint)
"""

import os, re, math, io, json, hashlib, requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from urllib.parse import urlparse
from collections import Counter
from bs4 import BeautifulSoup
from tldextract import extract
from PIL import Image
import imagehash

# ------------------------------
# CONFIG
# ------------------------------
CACHE_FILE = "feature_cache.json"
CSE_KEYWORDS = [
    "sbi","sbicard","sbilife","hdfc","hdfclife","hdfcergo",
    "icici","icicibank","icicidirect","pnb","bankofbaroda",
    "nic","gov","irctc","airtel","iocl","indianoil"
]

CSE_FAVICONS = {
    "SBI": ["https://sbi.co.in/favicon.ico"],
    "HDFC": ["https://www.hdfcbank.com/favicon.ico"],
    "ICICI": ["https://www.icicibank.com/favicon.ico"],
    "PNB": ["https://www.pnbindia.in/favicon.ico"],
    "BoB": ["https://www.bankofbaroda.in/favicon.ico"],
    "NIC": ["https://www.nic.in/favicon.ico"],
    "IRCTC": ["https://www.irctc.co.in/favicon.ico"],
    "Airtel": ["https://www.airtel.in/favicon.ico"],
    "IOCL": ["https://www.iocl.com/favicon.ico"]
}

# ------------------------------
# UTILS
# ------------------------------
def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    prob = [freq / len(s) for freq in Counter(s).values()]
    return -sum(p * math.log2(p) for p in prob)

def has_ip_address(host: str) -> bool:
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))

def fetch_page_html(url: str, timeout: int = 10) -> str:
    try:
        if not url.startswith("http"):
            url = "http://" + url
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if "text/html" in r.headers.get("Content-Type", ""):
            return r.text
    except Exception:
        return ""
    return ""

def extract_html_features(html: str) -> dict:
    feats = {
        "html_length": 0, "num_links": 0, "external_links_ratio": 0,
        "num_forms": 0, "has_login_form": 0, "num_scripts": 0,
        "num_iframes": 0, "title_length": 0, "meta_keyword_match": 0
    }
    if not html:
        return feats
    feats["html_length"] = len(html)
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        feats["title_length"] = len(soup.title.string)
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    feats["num_links"] = len(links)
    external_links = [l for l in links if l and not l.startswith("#") and not l.startswith("/")]
    feats["external_links_ratio"] = len(external_links) / (len(links) + 1e-5)
    forms = soup.find_all("form")
    feats["num_forms"] = len(forms)
    feats["has_login_form"] = 1 if any("password" in str(f).lower() for f in forms) else 0
    feats["num_scripts"] = len(soup.find_all("script"))
    feats["num_iframes"] = len(soup.find_all("iframe"))
    for m in soup.find_all("meta"):
        cont = str(m.get("content") or "").lower()
        if any(kw in cont for kw in CSE_KEYWORDS):
            feats["meta_keyword_match"] = 1
            break
    return feats

def get_favicon_hash(url: str) -> str | None:
    try:
        if not url.startswith("http"):
            url = "http://" + url
        domain = extract(urlparse(url).netloc)
        base = f"https://{domain.domain}.{domain.suffix}"
        fav_urls = [base + "/favicon.ico", base + "/favicon.png"]
        for f in fav_urls:
            try:
                r = requests.get(f, timeout=5)
                if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                    img = Image.open(io.BytesIO(r.content))
                    return str(imagehash.phash(img.convert("RGB")))
            except Exception:
                continue
    except Exception:
        return None
    return None

def compare_with_cse_favicons(fav_hash: str, cse_lib: dict) -> tuple[str, int]:
    if not fav_hash:
        return "", 999
    best_cse, best_dist = "", 999
    for cse, urls in cse_lib.items():
        for u in urls:
            try:
                r = requests.get(u, timeout=5)
                img = Image.open(io.BytesIO(r.content))
                official_hash = str(imagehash.phash(img.convert("RGB")))
                dist = imagehash.hex_to_hash(fav_hash) - imagehash.hex_to_hash(official_hash)
                if dist < best_dist:
                    best_dist, best_cse = dist, cse
            except Exception:
                continue
    return best_cse, best_dist

# ------------------------------
# FEATURE EXTRACTION
# ------------------------------
def extract_features(url: str) -> dict:
    u = str(url).strip()
    parsed = urlparse(u if "://" in u else f"http://{u}")
    domain_info = extract(parsed.netloc)
    domain = f"{domain_info.domain}.{domain_info.suffix}" if domain_info.suffix else domain_info.domain
    subdomain = domain_info.subdomain or ""
    host = parsed.netloc.lower()
    url_lower = u.lower()

    feats = {}

    # --- FULL URL LEXICAL FEATURES ---
    feats["url_length"] = len(u)
    feats["num_dots"] = u.count(".")
    feats["num_hyphens"] = u.count("-")
    feats["num_underscores"] = u.count("_")
    feats["num_slashes"] = u.count("/")
    feats["num_digits"] = sum(c.isdigit() for c in u)
    feats["digit_ratio"] = feats["num_digits"] / (len(u) + 1e-5)
    feats["num_special"] = sum(u.count(ch) for ch in "?=.$!#%")
    feats["num_question"] = u.count("?")
    feats["num_equal"] = u.count("=")
    feats["num_dollar"] = u.count("$")
    feats["num_exclamation"] = u.count("!")
    feats["num_hashtag"] = u.count("#")
    feats["num_percent"] = u.count("%")
    feats["repeated_digits_url"] = 1 if re.search(r"\d{3,}", u) else 0

    # --- DOMAIN FEATURES ---
    feats["domain_length"] = len(domain)
    feats["num_hyphens_domain"] = domain.count("-")
    feats["num_special_domain"] = sum(domain.count(ch) for ch in "$#%_")
    feats["has_special_domain"] = 1 if feats["num_special_domain"] > 0 else 0
    feats["subdomain_count"] = len(subdomain.split(".")) if subdomain else 0
    feats["avg_subdomain_len"] = np.mean([len(s) for s in subdomain.split(".")]) if subdomain else 0
    feats["subdomain_hyphen"] = 1 if "-" in subdomain else 0
    feats["subdomain_repeated_digits"] = 1 if re.search(r"\d{2,}", subdomain) else 0
    feats["entropy_url"] = shannon_entropy(u)
    feats["entropy_domain"] = shannon_entropy(domain)
    feats["https"] = 1 if parsed.scheme.lower() == "https" else 0
    feats["has_ip_host"] = 1 if has_ip_address(host) else 0
    feats["is_related_to_cse"] = 1 if any(kw in u.lower() for kw in CSE_KEYWORDS) else 0

    # --- CONTENT FEATURES ---
    html = fetch_page_html(u)
    feats.update(extract_html_features(html))

    # --- FAVICON FEATURES ---
    fav_hash = get_favicon_hash(u)
    cse, dist = compare_with_cse_favicons(fav_hash, CSE_FAVICONS)
    feats["favicon_match_cse"] = cse
    feats["favicon_distance"] = dist

    return feats

# ------------------------------
# MAIN
# ------------------------------
def build_dataset(whitelist_xlsx, phishing_dir, output_csv):
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            cache = json.load(open(CACHE_FILE, "r"))
        except Exception:
            pass

    rows = []
    whitelist_df = pd.read_excel(whitelist_xlsx)
    url_col = [c for c in whitelist_df.columns if "url" in c.lower() or "domain" in c.lower()][0]
    wl_urls = whitelist_df[url_col].dropna().astype(str).tolist()

    for u in tqdm(wl_urls, desc="Whitelisted"):
        if u in cache:
            feats = cache[u]
        else:
            feats = extract_features(u)
            cache[u] = feats
        feats["label"] = 0
        feats["source"] = "whitelist"
        rows.append(feats)

    phishing_files = [f for f in os.listdir(phishing_dir) if f.endswith(".xlsx")]
    for f in phishing_files:
        df = pd.read_excel(os.path.join(phishing_dir, f))
        url_col = [c for c in df.columns if "url" in c.lower() or "domain" in c.lower()][0]
        urls = df[url_col].dropna().astype(str).tolist()
        for u in tqdm(urls, desc=f"Phishing - {f}"):
            if u in cache:
                feats = cache[u]
            else:
                feats = extract_features(u)
                cache[u] = feats
            feats["label"] = 1
            feats["source"] = f
            rows.append(feats)

            # save cache checkpoint every 100
            if len(rows) % 100 == 0:
                with open(CACHE_FILE, "w") as fp:
                    json.dump(cache, fp)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Dataset saved: {output_csv} | Samples: {len(df_out)}")
    with open(CACHE_FILE, "w") as fp:
        json.dump(cache, fp)

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extended Feature Extractor with Cache")
    ap.add_argument("--whitelist", required=True, help="Path to whitelisted.xlsx")
    ap.add_argument("--phishing_dir", required=True, help="Directory containing phishing .xlsx files")
    ap.add_argument("--output", default="features_extended.csv", help="Output CSV file path")
    args = ap.parse_args()
    build_dataset(args.whitelist, args.phishing_dir, args.output)
