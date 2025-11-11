#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS-02 Submission Builder – Model Inference + Evidences + Excel
--------------------------------------------------------------

Inputs:
- Relevance CSV: rows with CSE relation (e.g., CSE_Relevance_Output.csv)
- Trained model + scaler in ./outputs/
  - auto-picks the newest *_balanced.pkl (or falls back to any .pkl) + scaler.pkl

Outputs (per Annexure-B):
PS-02_<APPID>_Submission/
├── PS-02_<APPID>_Submission_Set.xlsx
├── PS-02_<APPID>_Evidences/
│   └── <CSE>_<two-level-domain>_<serial>.pdf
└── PS-02_<APPID>_Documentation_folder/
    └── (placeholder for your report)

Notes:
- Uses open/public sources only: rdap.org, DNS, optional ip-api.com (free)
- Screenshot via Selenium (Chrome/Chromium); if not available, writes a blank PDF.
- Resumable: if a PDF already exists for that row, it won’t re-screenshot.
"""

import os
import io
import re
import sys
import json
import time
import math
import glob
import socket
import argparse
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import requests
import pandas as pd
from PIL import Image
from io import BytesIO

# Optional deps that we try to use if present
try:
    import dns.resolver  # dnspython
except Exception:
    dns = None

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
except Exception:
    webdriver = None

# Try your feature extractor first (fast/extended)
EXTRACTOR = None
try:
    from model.feature_extractor import extract_features as _fx
    EXTRACTOR = _fx
except Exception:
    try:
        from model.feature_extractor import extract_features as _fx2
        EXTRACTOR = _fx2
    except Exception:
        EXTRACTOR = None

import joblib

# ----------------------------
# Config (can be overridden by CLI)
# ----------------------------
DEFAULT_APP_ID = "AIGR-S74001"
DEFAULT_RELEVANCE_CSV = "CSE_Relevance_Output.csv"
PROBLEM_NUM = "PS-02"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

PARKED_KEYWORDS = [
    "coming soon","under construction","domain for sale","buy this domain",
    "parkingcrew","afternic","sedo","bodis","godaddy",
    "this site can’t be reached","this site can't be reached",
    "apache2 debian default page","welcome to nginx","account suspended","site suspended"
]

# CSE mapping for official domain list (for Excel "Corresponding CSE Domain Name")
CSE_DEFS = {
    "SBI":   {"official_domains": ["onlinesbi.sbi","sbi.co.in","sbicard.com","sbilife.co.in","sbiepay.sbi"]},
    "ICICI": {"official_domains": ["icicibank.com","icicidirect.com","iciciprulife.com","icicilombard.com"]},
    "HDFC":  {"official_domains": ["hdfcbank.com","hdfclife.com","hdfcergo.com"]},
    "PNB":   {"official_domains": ["pnbindia.in","netpnb.com"]},
    "BoB":   {"official_domains": ["bankofbaroda.in","bankofbaroda.com","bobibanking.com"]},
    "NIC":   {"official_domains": ["nic.in","gov.in","kavach.gov.in"]},
    "IRCTC": {"official_domains": ["irctc.co.in"]},
    "Airtel":{"official_domains": ["airtel.in"]},
    "IOCL":  {"official_domains": ["iocl.com","indianoil.in"]},
}

# ----------------------------
# Logging
# ----------------------------
def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("submission")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs(root: Path, app_id: str):
    base = Path(f"{PROBLEM_NUM}_{app_id}_Submission")
    evid = base / f"{PROBLEM_NUM}_{app_id}_Evidences"
    docs = base / f"{PROBLEM_NUM}_{app_id}_Documentation_folder"
    base.mkdir(parents=True, exist_ok=True)
    evid.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    return base, evid, docs

def load_best_model(outputs_dir: Path, logger) -> tuple[object, object]:
    """Pick newest *_balanced.pkl (else any .pkl). Load scaler.pkl if available."""
    pkl_list = sorted(outputs_dir.glob("*_balanced.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pkl_list:
        pkl_list = sorted(outputs_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pkl_list:
        raise FileNotFoundError("No model .pkl found in ./outputs/")
    model_path = pkl_list[0]
    logger.info(f"🔹 Using model: {model_path.name}")
    model = joblib.load(model_path)

    scaler = None
    scaler_path = outputs_dir / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info("🔹 Loaded scaler.pkl")
    else:
        logger.info("ℹ️ No scaler.pkl found; will feed raw features to model.")
    return model, scaler

def normalize_domain(d: str) -> str:
    d = (d or "").strip().lower()
    d = re.sub(r"^https?://", "", d, flags=re.I)
    return d.split("/")[0]

def safe_two_level_name(domain: str) -> str:
    """
    Convert 'aa.bb.cc.tld' -> 'bb.cc.tld' (max 1 subdomain level)
    """
    parts = domain.split(".")
    if len(parts) < 2:  # no suffix?
        return domain
    # Keep last 2+1 parts if available
    if len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])

def is_parked_or_empty(domain: str, timeout=8) -> bool:
    try:
        for scheme in ("https", "http"):
            url = f"{scheme}://{domain}"
            r = requests.get(url, timeout=timeout, headers=HEADERS)
            html = (r.text or "").lower()
            if r.status_code != 200:
                continue
            if len(html) < 600:
                return True
            if any(k in html for k in PARKED_KEYWORDS):
                return True
            # If got some decent HTML, treat as not parked
            return False
        return True
    except Exception:
        return True

def screenshot_to_pdf(domain: str, out_pdf: Path, logger, width=1280, height=720):
    """
    Try Selenium → PNG → single-page PDF. If Selenium unavailable/fails, write blank PDF.
    """
    try:
        if webdriver is None:
            raise RuntimeError("Selenium not available")

        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--window-size={width},{height}")

        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(12)

        url = f"http://{domain}"
        driver.get(url)
        png = driver.get_screenshot_as_png()
        driver.quit()

        img = Image.open(BytesIO(png))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out_pdf, "PDF", resolution=180.0)
        return True
    except Exception as e:
        logger.info(f"📄 Screenshot fallback for {domain}: {e}")
        # blank placeholder
        img_blank = Image.new("RGB", (width, height), (255, 255, 255))
        img_blank.save(out_pdf, "PDF", resolution=180.0)
        return False

def rdap_fetch(domain: str, logger) -> dict:
    """
    WHOIS/RDAP fields needed: creation date, registrar, registrant org/country, nameservers.
    """
    out = {
        "creation_date": "",
        "registrar": "",
        "registrant_org": "",
        "registrant_name": "",
        "registrant_country": "",
        "nameservers": []
    }
    try:
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=10, headers=HEADERS)
        if r.status_code != 200:
            return out
        data = r.json()

        # created
        for e in data.get("events", []):
            if e.get("eventAction") == "registration":
                out["creation_date"] = e.get("eventDate", "")
                break

        # entities
        for ent in data.get("entities", []):
            roles = ent.get("roles", [])
            vcard = ent.get("vcardArray", [])
            if "registrar" in roles and not out["registrar"]:
                out["registrar"] = ent.get("handle", "") or ""
                # Sometimes registrar name is inside vcard
                if vcard and len(vcard) == 2:
                    for row in vcard[1]:
                        if row[0] in ("org", "fn"):
                            out["registrar"] = row[3]
                            break
            if any(r in roles for r in ["registrant", "administrative", "technical"]):
                if vcard and len(vcard) == 2:
                    for row in vcard[1]:
                        if row[0] == "org":
                            out["registrant_org"] = row[3]
                        if row[0] == "fn":
                            out["registrant_name"] = row[3]
                        if row[0] == "adr":
                            adr = row[3]
                            if isinstance(adr, list) and len(adr) >= 7:
                                out["registrant_country"] = adr[-1]
        # nameservers
        out["nameservers"] = [ns.get("ldhName") for ns in data.get("nameservers", []) if ns.get("ldhName")]
    except Exception as e:
        logger.debug(f"RDAP error for {domain}: {e}")
    return out

def dns_enrich(domain: str, logger) -> tuple[str, list, dict]:
    """
    Return (hosting_ip, nameservers, dns_records) using socket + dnspython if available.
    """
    hosting_ip = ""
    nameservers = []
    records = {}

    # A/AAAA
    try:
        host, aliases, ips = socket.gethostbyname_ex(domain)
        if ips:
            hosting_ip = ips[0]
        records["A"] = ips
    except Exception:
        pass

    if dns:
        try:
            # NS
            ns_answers = dns.resolver.resolve(domain, "NS")
            ns = sorted({str(r.target).rstrip(".") for r in ns_answers})
            nameservers = ns
            records["NS"] = ns
        except Exception:
            pass
        try:
            # MX
            mx_ans = dns.resolver.resolve(domain, "MX")
            mx = sorted({str(r.exchange).rstrip(".") for r in mx_ans})
            records["MX"] = mx
        except Exception:
            pass
        try:
            # TXT (limit)
            txt_ans = dns.resolver.resolve(domain, "TXT")
            txt = []
            for r in txt_ans:
                try:
                    s = b"".join(r.strings).decode("utf-8", "ignore")
                    txt.append(s)
                except Exception:
                    pass
            if txt:
                records["TXT"] = txt[:5]
        except Exception:
            pass

    # If no NS via dnspython, use RDAP result later
    return hosting_ip, nameservers, records

def geoip(hosting_ip: str, logger) -> tuple[str, str]:
    """
    Optional ISP/Country via ip-api.com (free). Graceful failure returns ("","").
    """
    if not hosting_ip:
        return "", ""
    try:
        r = requests.get(f"http://ip-api.com/json/{hosting_ip}?fields=isp,country", timeout=6, headers=HEADERS)
        if r.status_code == 200:
            data = r.json()
            return data.get("isp", "") or "", data.get("country", "") or ""
    except Exception as e:
        logger.debug(f"geoip failed for {hosting_ip}: {e}")
    return "", ""

def load_features_for_model(domain: str, logger) -> dict:
    """
    Use your extractor if available. As a fallback, a minimal lexical set.
    """
    if EXTRACTOR:
        try:
            feats = EXTRACTOR(domain)
            # Ensure plain Python types (no numpy types) for model/scaler
            clean = {}
            for k, v in feats.items():
                if hasattr(v, "item"):
                    clean[k] = v.item()
                else:
                    clean[k] = v
            return clean
        except Exception as e:
            logger.debug(f"extract_features() failed for {domain}: {e}")

    # Fallback minimal features (keep names stable with your training if possible)
    d = domain
    url = f"http://{domain}"
    return {
        "url_length": len(url),
        "num_dots": d.count("."),
        "num_hyphens": d.count("-"),
        "num_underscores": d.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in d),
        "digit_ratio": (sum(c.isdigit() for c in d) / (len(d) + 1e-5)),
        "entropy_domain": (lambda s: -sum((s.count(ch)/len(s))*math.log2((s.count(ch)/len(s)))
                                          for ch in set(s))) (d) if d else 0.0,
        "https": 0,
        "has_ip_host": 1 if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", d) else 0,
    }

def predict_label(model, scaler, feats_df_row) -> tuple[int, float]:
    """
    Return (pred, prob1). If scaler is present, apply to feature vector.
    """
    X = feats_df_row.values.reshape(1, -1)
    if scaler is not None:
        try:
            Xs = scaler.transform(X)
        except Exception:
            Xs = X
    else:
        Xs = X

    pred = int(model.predict(Xs)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(model.predict_proba(Xs)[0][1])
        except Exception:
            prob = None
    return pred, (prob if prob is not None else 0.0)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="PS-02 Submission Builder (Model + Evidences + Excel)")
    ap.add_argument("--app-id", default=DEFAULT_APP_ID, help="Application ID (folder naming)")
    ap.add_argument("--relevance", default=DEFAULT_RELEVANCE_CSV, help="Path to CSE relevance CSV")
    ap.add_argument("--outputs-dir", default="outputs", help="Directory containing model/scaler")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap for debugging")
    ap.add_argument("--log", default="submission_build.log", help="Log file path")
    args = ap.parse_args()

    logger = setup_logger(Path(args.log))
    logger.info("🚀 Starting PS-02 submission builder")
    logger.info(f"App ID: {args.app_id}")

    base_dir, evid_dir, docs_dir = ensure_dirs(Path("."), args.app_id)
    excel_out = base_dir / f"{PROBLEM_NUM}_{args.app_id}_Submission_Set.xlsx"

    # Load model + scaler
    model, scaler = load_best_model(Path(args.outputs_dir), logger)

    # Read relevance CSV
    if not Path(args.relevance).exists():
        logger.error(f"Missing relevance CSV: {args.relevance}")
        sys.exit(1)

    rel = pd.read_csv(args.relevance)
    # Expect columns: Domain, Related_CSE (non-empty), etc.
    if "Domain" not in rel.columns:
        logger.error("Relevance CSV must have a 'Domain' column.")
        sys.exit(1)

    # Keep only related rows if present; else use all
    if "Related_CSE" in rel.columns:
        rel = rel[(rel["Related_CSE"].astype(str).str.len() > 0) & (rel["Status"].astype(str) == "related")] if "Status" in rel.columns else rel[rel["Related_CSE"].astype(str).str.len() > 0]

    # Apply limit for debugging
    if args.limit and args.limit > 0:
        rel = rel.head(args.limit).copy()

    rel["Domain"] = rel["Domain"].astype(str).str.lower().str.strip()
    rel = rel.drop_duplicates(subset=["Domain"]).reset_index(drop=True)

    # Build feature frame progressively
    rows_excel = []
    serial_no = 1
    processed = 0
    t0 = time.time()

    # Determine model feature names by trying the first feature mapping
    # We'll build a consistent column order after first item
    feature_columns = None

    for idx, row in rel.iterrows():
        domain = normalize_domain(row["Domain"])
        cse = (row.get("Related_CSE") or "").strip()
        if not domain:
            continue

        # Model features
        feats = load_features_for_model(domain, logger)
        if feature_columns is None:
            feature_columns = list(feats.keys())
        # Ensure consistent order and fill missing
        feats_ordered = {k: feats.get(k, 0) for k in feature_columns}
        feats_df_row = pd.Series(feats_ordered, index=feature_columns, dtype="float64")

        # Predict
        pred, prob1 = predict_label(model, scaler, feats_df_row)

        # Suspected heuristic
        label = None
        if pred == 1:
            parked = is_parked_or_empty(domain)
            label = "Suspected" if parked else "Phishing"
        else:
            # skip benign; PS-02 wants phishing/suspected
            continue

        # RDAP + DNS + GeoIP
        who = rdap_fetch(domain, logger)
        hosting_ip, ns_dns, dns_records = dns_enrich(domain, logger)
        if not ns_dns:  # fallback to RDAP NS if DNS missing
            ns_dns = who.get("nameservers") or []
        isp, country_host = geoip(hosting_ip, logger)

        # Evidence PDF
        pdf_name = f"{cse or 'CSE'}_{safe_two_level_name(domain)}_{serial_no}.pdf"
        pdf_path = evid_dir / pdf_name
        if not pdf_path.exists():  # resumable: don't redo screenshots
            screenshot_to_pdf(domain, pdf_path, logger)

        # Detection time (now)
        now = datetime.now()
        detect_date = now.strftime("%d-%m-%Y")
        detect_time = now.strftime("%H-%M-%S")

        # Annexure-B row
        row_out = {
            "Application_ID": args.app_id,
            "Source of detection": "CSE Relevance Classifier + Phishing Model",
            "Identified Phishing/Suspected Domain Name": domain,
            "Corresponding CSE Domain Name": ", ".join(CSE_DEFS.get(cse, {}).get("official_domains", [])),
            "Critical Sector Entity Name": cse,
            "Phishing/Suspected Domains (i.e. Class Label)": label,
            "Domain Registration Date": who.get("creation_date", ""),
            "Registrar Name": who.get("registrar", ""),
            "Registrant Name or Registrant Organisation": (who.get("registrant_name") or who.get("registrant_org") or ""),
            "Registrant Country": who.get("registrant_country", ""),
            "Name Servers": ", ".join(ns_dns),
            "Hosting IP": hosting_ip,
            "Hosting ISP": isp,
            "Hosting Country": country_host,
            "DNS Records (if any)": json.dumps(dns_records, ensure_ascii=False),
            "Evidence file name": pdf_name,
            "Date of detection (DD-MM-YYYY)": detect_date,
            "Time of detection (HH-MM-SS)": detect_time,
            "Date of Post (If detection is from Source: social media)": "",
            "Remarks (If any)": f"ModelProb={prob1:.4f}"
        }
        rows_excel.append(row_out)

        serial_no += 1
        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - t0
            logger.info(f"… progress {processed}/{len(rel)} rows | elapsed {elapsed/60:.1f} min")

    if not rows_excel:
        logger.warning("No phishing/suspected rows produced. Nothing to write.")
        sys.exit(0)

    # Write Excel
    df_out = pd.DataFrame(rows_excel)
    df_out.to_excel(excel_out, index=False)
    logger.info(f"✅ Wrote Excel: {excel_out}")

    logger.info(f"📁 Evidences in: {evid_dir}")
    logger.info(f"🧾 Docs folder: {docs_dir} (place your report here)")
    logger.info("✅ Submission pack ready. Zip the folder for upload.")

if __name__ == "__main__":
    main()
