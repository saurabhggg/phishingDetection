#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSE Prefilter Classifier – Fast Approximation
---------------------------------------------
Scores all domains quickly for CSE relevance using lightweight lexical,
semantic, and keyword-based features.

Input  : combined_output.csv (with 'Domain' column)
Output : prefilter_scored.csv (Domain + CSE_Score + Best_CSE + signals)

Use the output to decide what to feed into the heavy classifier.
"""

import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urlparse

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
INPUT_FILE  = "combined_output.csv"
OUTPUT_FILE = "prefilter_scored.csv"

CSE_KEYWORDS = {
    "SBI":   ["sbi","sbicard","sbilife","yonobusiness","onlinesbi","sbiepay"],
    "ICICI": ["icici","icicidirect","iciciprulife","icicibank","icicilombard"],
    "HDFC":  ["hdfc","hdfcbank","hdfclife","hdfcergo"],
    "PNB":   ["pnb","pnbindia","netpnb"],
    "BoB":   ["bankofbaroda","bobibanking"],
    "NIC":   ["nic","gov.in","kavach"],
    "IRCTC": ["irctc"],
    "Airtel":["airtel"],
    "IOCL":  ["iocl","indianoil"]
}

SUSPICIOUS_TLDS = set([".xyz",".top",".tk",".ga",".cf",".ml",".gq",".cam",".buzz",".click",".work",".monster",".icu"])

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_host(u):
    try:
        u = u.strip().lower()
        if not re.match(r"^https?://", u):
            u = "http://" + u
        parsed = urlparse(u)
        return parsed.netloc or u
    except Exception:
        return u

def lexical_features(host):
    tld = "." + host.split(".")[-1] if "." in host else ""
    return dict(
        len_url=len(host),
        dots=host.count("."),
        hyphens=host.count("-"),
        digits=sum(c.isdigit() for c in host),
        susp_tld=int(any(tld.endswith(st) for st in SUSPICIOUS_TLDS))
    )

def keyword_match(host):
    low = host.lower()
    matches = []
    for cse, kws in CSE_KEYWORDS.items():
        if any(kw in low for kw in kws):
            matches.append(cse)
    return matches

# ---------------------------------------------------------------------
# Model Init (small & fast)
# ---------------------------------------------------------------------
print("🔹 Loading semantic model (MiniLM-L6-v2)...")
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

CSE_NAMES = list(CSE_KEYWORDS.keys())
CSE_EMBEDS = model.encode(CSE_NAMES, convert_to_tensor=True, normalize_embeddings=True)

def semantic_best_match(text):
    from sentence_transformers import util
    tokens = re.sub(r"[^a-zA-Z0-9]+"," ", text)
    emb = model.encode(tokens, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(emb, CSE_EMBEDS)[0]
    best_idx = int(sims.argmax())
    return CSE_NAMES[best_idx], float(sims[best_idx])

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
print(f"📂 Loading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)
if "Domain" not in df.columns:
    raise ValueError("Input file must contain a 'Domain' column.")

domains = df["Domain"].dropna().astype(str).unique().tolist()

rows = []
for d in tqdm(domains, desc="Scoring domains"):
    host = normalize_host(d)
    if not host:
        continue

    lex = lexical_features(host)
    kw_hits = keyword_match(host)
    sem_cse, sem_score = semantic_best_match(host)

    # Simple heuristic scoring
    base_score = sem_score
    if kw_hits:
        base_score += 0.2 * len(kw_hits)
    if lex["susp_tld"]:
        base_score -= 0.1
    if lex["hyphens"] > 2 or lex["digits"] > 5:
        base_score -= 0.05

    rows.append({
        "Domain": host,
        "Best_CSE": sem_cse,
        "CSE_Score": round(base_score, 3),
        "Keyword_Hits": ",".join(kw_hits),
        **lex
    })

out_df = pd.DataFrame(rows)
out_df.sort_values("CSE_Score", ascending=False, inplace=True)
out_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Done: scored {len(out_df)} domains → {OUTPUT_FILE}")
print(out_df["CSE_Score"].describe())
