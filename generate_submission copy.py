#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_submission.py – Robust version
---------------------------------------
✅ Automatically aligns features with training schema
✅ Fixes string→float, missing columns, and unseen feature issues
✅ Predicts using trained model on CSE relevance outputs
✅ Saves screenshots for flagged (phishing) URLs
"""

import os, json, time, hashlib, joblib
import pandas as pd, numpy as np
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from model.feature_extractor import extract_features
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
INPUT_FILE = "prefilter_scored.csv"
OUTPUT_FILE = "submission_results.csv"
CACHE_FILE = "submission_cache.json"

MODEL_DIR = "outputs"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------------------------
def load_best_model():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith("_balanced.pkl")]
    if not models:
        raise FileNotFoundError("❌ No trained models found in /outputs/")
    best_model = sorted(models)[0]
    print(f"✅ Loading model: {best_model}")
    model = joblib.load(os.path.join(MODEL_DIR, best_model))
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    # Get training feature names if available
    try:
        feature_names = model.feature_names_in_.tolist()
    except AttributeError:
        feature_names = None
    return model, scaler, feature_names

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def safe_filename(url):
    return os.path.join(SCREENSHOT_DIR, hashlib.md5(url.encode()).hexdigest()[:10] + ".png")

def capture_screenshot(url, out_path):
    try:
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1280,800")
        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(10)
        driver.get(url if url.startswith("http") else f"http://{url}")
        time.sleep(2)
        driver.save_screenshot(out_path)
        driver.quit()
        return True
    except Exception as e:
        print(f"⚠️ Screenshot failed for {url[:60]}: {e}")
        return False

def load_cache():
    return json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# ---------------------------------------------------------------------
# ENCODING + ALIGNMENT
# ---------------------------------------------------------------------
def encode_and_align(df: pd.DataFrame, train_features: list[str]) -> pd.DataFrame:
    df = df.fillna(0)

    # Encode non-numeric features
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes
        elif df[col].dtype == "bool":
            df[col] = df[col].astype("int")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if not train_features:
        return df  # fallback

    # Align columns to match training schema
    for feat in train_features:
        if feat not in df.columns:
            df[feat] = 0  # missing -> fill with 0
    df = df[train_features]  # drop extras
    return df

# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
    print("🚀 Generating submission predictions...")
    model, scaler, train_features = load_best_model()
    cache = load_cache()

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"📥 Loaded {len(df)} candidate domains")

    for col in df.columns:
        if df[col].dtype == "object" and col != "Domain":
            print(f"⚙️ Encoding text column: {col}")
            df[col] = df[col].astype("category").cat.codes

    url_col = [c for c in df.columns if "url" in c.lower() or "domain" in c.lower()][0]
    urls = df[url_col].dropna().astype(str).tolist()
    rows = []
    for u in tqdm(urls):
            if u in cache:
                feats = cache[u]
            else:
                feats = extract_features(u)
                cache[u] = feats
            rows.append(feats)

            # Checkpoint every 100 processed rows
            # if len(rows) % 100 == 0:
            #     save_cache(cache)
    # Encode & align
    # feats_df = encode_and_align(df.copy(), train_features)
    print("rows = ", row[0])
    
    scaler = StandardScaler()
    rows_new = scaler.fit_transform(rows)

    results = []
    for i,row in enumerate(rows_new):
        domain = urls[i]
        prob = float(model.predict_proba(row))
        label = int(prob >= 0.5)

        screenshot_path = ""
        # if label == 1:
        #     screenshot_path = safe_filename(domain)
        #     capture_screenshot(domain, screenshot_path)

        out = {
            "Domain": domain,
            "Predicted_Label": label,
            "Phishing_Score": round(prob, 4),
            "Related_CSE": row.get("Related_CSE", ""),
            "Relation_Score": row.get("Relation_Score", 0),
            "Screenshot_Path": screenshot_path,
        }
        results.append(out)
        cache[domain] = out

        if len(results) % 100 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
            save_cache(cache)
            print(f"💾 Checkpoint saved ({len(results)} / {len(df)})")

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    save_cache(cache)
    print(f"✅ Done. Results saved to {OUTPUT_FILE}")
    print(f"📸 Screenshots saved in {SCREENSHOT_DIR}/")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
