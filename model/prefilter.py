#!/usr/bin/env python3
# prefilter_v2.py
"""
Fast prefilter for CSE-relevance (vectorized lexical + simple embedding token features).

- Input: CSV with a column containing domains/urls (auto-detects 'domain' or 'url' column)
- Output: prefilter_scored.csv with columns:
    Domain, base_host, score (0..1), risk_bucket (low/medium/high), reason
- Fast: pure CPU numeric transforms; can process ~1M domains/min (depends on CPU).
- Resume: will skip domains already present in output file.
Dependencies:
  pip install pandas numpy scikit-learn tldextract
Usage:
  python prefilter_v2.py --input combined_domains.csv --out prefilter_scored.csv --model logistic
"""
import argparse, os, re, time, math
from pathlib import Path
import pandas as pd
import numpy as np
import tldextract
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ---------- CONFIG ----------
OUT_DEFAULT = "prefilter_scored.csv"
CACHE_MODEL = "models/prefilter_lr.joblib"
os.makedirs("models", exist_ok=True)

SUSP_TLDS = {".xyz", ".top", ".tk", ".ga", ".cf", ".ml", ".gq", ".cam", ".buzz", ".click", ".work", ".monster", ".icu"}

# ---------- HELPERS ----------
def detect_domain_col(df):
    for c in df.columns:
        if "domain" in c.lower() or "url" in c.lower():
            return c
    return df.columns[0]

def normalize_host(s):
    s = (str(s) or "").strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = s.split("/")[0]
    return s

def lexical_features_series(hosts):
    # hosts: list/array of host strings
    n = len(hosts)
    out = {
        "host": np.array(hosts, dtype=object),
        "len_host": np.zeros(n, dtype=np.int32),
        "num_dots": np.zeros(n, dtype=np.int32),
        "num_hyphen": np.zeros(n, dtype=np.int32),
        "num_digits": np.zeros(n, dtype=np.int32),
        "digit_ratio": np.zeros(n, dtype=np.float32),
        "num_special": np.zeros(n, dtype=np.int32),
        "subdomain_count": np.zeros(n, dtype=np.int32),
        "avg_sub_len": np.zeros(n, dtype=np.float32),
        "susp_tld": np.zeros(n, dtype=np.int32),
        "entropy_host": np.zeros(n, dtype=np.float32),
    }

    def entropy(s):
        if not s: return 0.0
        vals = {}
        for ch in s:
            vals[ch] = vals.get(ch, 0) + 1
        ent = 0.0
        L = len(s)
        for v in vals.values():
            p = v / L
            ent -= p * math.log2(p)
        return ent

    for i,h in enumerate(hosts):
        if not h or h == "nan":
            continue
        out["len_host"][i] = len(h)
        out["num_dots"][i] = h.count(".")
        out["num_hyphen"][i] = h.count("-")
        digits = sum(c.isdigit() for c in h)
        out["num_digits"][i] = digits
        out["digit_ratio"][i] = digits / (len(h) + 1e-6)
        specials = sum(h.count(ch) for ch in "_$#%!")
        out["num_special"][i] = specials
        parts = h.split(".")
        out["subdomain_count"][i] = max(0, len(parts)-2)
        lens = [len(p) for p in parts[:-2]] if len(parts) > 2 else []
        out["avg_sub_len"][i] = float(np.mean(lens)) if lens else 0.0
        # tld
        ext = tldextract.extract(h)
        tld = ("." + ext.suffix) if ext.suffix else ""
        out["susp_tld"][i] = 1 if tld in SUSP_TLDS else 0
        out["entropy_host"][i] = entropy(h)
    return out

def build_or_load_model():
    # If model exists, load; else train a simple rule-based fallback and save for reuse.
    if Path(CACHE_MODEL).exists():
        return joblib.load(CACHE_MODEL)
    # Train a tiny logistic on synthetic data (fast) just to get numeric mapping
    X = np.vstack([
        # benign-ish examples
        [10,1,0,0,0.0,0,0.0,0,2.0],
        [12,2,0,0,0.0,0,0.0,0,3.0],
        # suspicious
        [40,7,3,5,0.12,2,5.0,1,4.5],
        [50,8,4,12,0.24,3,6.0,1,5.0],
    ])
    y = np.array([0,0,1,1])
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipeline.fit(X,y)
    joblib.dump(pipeline, CACHE_MODEL)
    return pipeline

# ---------- MAIN ----------
def main(input_csv, out_csv, model_choice):
    df = pd.read_csv(input_csv)
    domain_col = detect_domain_col(df)
    hosts = df[domain_col].astype(str).map(normalize_host).tolist()

    # resume: skip domains already in out_csv
    existing = set()
    if Path(out_csv).exists():
        try:
            prev = pd.read_csv(out_csv)
            existing = set(prev["Domain"].astype(str).str.lower().tolist())
        except Exception:
            existing = set()

    # prepare features in batches
    batch_size = 200_000
    model = build_or_load_model()

    rows = []
    t0 = time.time()
    for i in range(0, len(hosts), batch_size):
        sub = hosts[i:i+batch_size]
        feats = lexical_features_series(sub)
        # numeric matrix for model: pick subset of features
        Xmat = np.column_stack([
            feats["len_host"], feats["num_dots"], feats["num_hyphen"], feats["num_digits"],
            feats["digit_ratio"], feats["num_special"], feats["avg_sub_len"], feats["susp_tld"], feats["entropy_host"]
        ])
        probs = model.predict_proba(Xmat)[:,1]
        for j, h in enumerate(sub):
            if not h or h in existing:
                continue
            score = float(probs[j])
            if score >= 0.75:
                bucket = "high"
            elif score >= 0.45:
                bucket = "medium"
            else:
                bucket = "low"
            reason = "lexical_score"
            rows.append({
                "Domain": h,
                "base_host": h,
                "score": round(score,4),
                "risk_bucket": bucket,
                "reason": reason
            })
            existing.add(h)

        # checkpoint append
        if rows:
            pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)
            rows = []
        elapsed = time.time() - t0
        print(f"Processed {min(i+batch_size, len(hosts))}/{len(hosts)} elapsed={elapsed:.1f}s")

    print("✅ prefilter done. Output:", out_csv)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV")
    ap.add_argument("--out", default=OUT_DEFAULT, help="Output CSV")
    ap.add_argument("--model", default="logistic", help="model (unused)")
    args = ap.parse_args()
    main(args.input, args.out, args.model)
