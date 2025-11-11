#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSE Relevance Classifier – FAST (Batched Semantics + Conditional RDAP/Favicon)
-----------------------------------------------------------------------------
• Input:  --input CSV with a 'Domain' column (default: combined_output.csv)
• Output: CSE_Relevance_Output.csv (only Status='related' rows written)
• Cache:  crawler/cse_relevance_cache_plus.json  (resumable)
• Log:    crawler/cse_relevance_log.txt

Speed-ups vs FULL+:
  - Batched SentenceTransformer embeddings (vectorize once)
  - RDAP + favicon only when URL/semantic signals suggest relevance
  - Higher concurrency, better connector pooling, fewer checkpoints/logging
  - Shorter timeouts, smarter HTML fetch

Expected: ~5–10× faster depending on network.

Usage:
  python cse_relevance_classifier_fast.py \
      --input combined_output.csv --concurrency 100 --save-interval 2000
"""

import asyncio, aiohttp, json, re, os, io, time, signal, sys, logging
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
import imagehash

# ---------------- configuration ----------------
IN_FILE    = Path("prefilter_for_heavy.csv")
OUT_FILE   = Path("CSE_Relevance_Output.csv")
CACHE_FILE = Path("crawler/cse_relevance_cache_plus.json")
LOG_FILE   = Path("crawler/cse_relevance_log.txt")

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HEADERS    = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
TIMEOUT    = aiohttp.ClientTimeout(total=10, connect=4, sock_read=8)

DEFAULT_CONCURRENCY   = 100
DEFAULT_SAVE_INTERVAL = 2000
SEM_THRESHOLD_PRIMARY = 0.60   # semantic threshold to trigger heavier checks
FAV_HD_MAX            = 6

PARKED_KEYWORDS = [
    "coming soon","under construction","domain for sale","buy this domain",
    "parkingcrew","afternic","sedo","bodis","godaddy",
    "this site can’t be reached","this site can't be reached",
    "apache2 debian default page","welcome to nginx","account suspended","site suspended"
]

SUSPICIOUS_TLDS = set([".xyz",".top",".tk",".ga",".cf",".ml",".gq",".cam",".buzz",".click",".work",".monster",".icu"])

PHISH_HINTS = [
    "verify","secure","login","update kyc","otp","enter otp","bank verification","netbanking",
    "password","cvv","card","pin","account verification","document upload","refund","prize"
]

CSE_DEFS = {
    "SBI":   {"official_domains": ["onlinesbi.sbi","sbi.co.in","sbicard.com","sbilife.co.in","sbiepay.sbi"], "keywords": ["sbi","sbicard","yonobusiness","onlinesbi","sbiepay","sbilife"]},
    "ICICI": {"official_domains": ["icicibank.com","icicidirect.com","iciciprulife.com","icicilombard.com"], "keywords": ["icici","icicidirect","iciciprulife","icicibank","icicilombard","icicicareers"]},
    "HDFC":  {"official_domains": ["hdfcbank.com","hdfclife.com","hdfcergo.com"], "keywords": ["hdfc","hdfcbank","hdfclife","hdfcergo"]},
    "PNB":   {"official_domains": ["pnbindia.in","netpnb.com"], "keywords": ["pnb","pnbindia","netpnb"]},
    "BoB":   {"official_domains": ["bankofbaroda.in","bankofbaroda.com","bobibanking.com"], "keywords": ["bankofbaroda","bobibanking"]},
    "NIC":   {"official_domains": ["nic.in","gov.in","kavach.gov.in"], "keywords": ["nic","gov.in","mgovcloud","kavach"]},
    "IRCTC": {"official_domains": ["irctc.co.in"], "keywords": ["irctc"]},
    "Airtel":{"official_domains": ["airtel.in"], "keywords": ["airtel"]},
    "IOCL":  {"official_domains": ["iocl.com","indianoil.in"], "keywords": ["iocl","indianoil"]},
}

# ---------------- logging (standalone) ----------------
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cse_fast")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logger(LOG_FILE)

# ---------------- tiny cache I/O ----------------
def load_json(path: Path, default):
    try:
        if path.exists():
            return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        pass
    return default

def save_json(path: Path, obj):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")

# ---------------- helpers: lexical / tld / parked / content ----------------
def normalize_host(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^https?://", "", s, flags=re.I)
    return s.split("/")[0]

def url_lexical_features(host: str) -> dict:
    has_digit = int(bool(re.search(r"\d", host)))
    has_hyphen = int("-" in host)
    dot_count = host.count(".")
    tld = "." + host.split(".")[-1] if "." in host else ""
    susp_tld = int(any(tld.endswith(st) for st in SUSPICIOUS_TLDS))
    return {
        "Lex_URL_Len": len(host),
        "Lex_NumDots": dot_count,
        "Lex_HasDigit": has_digit,
        "Lex_HasHyphen": has_hyphen,
        "Lex_SuspiciousTLD": susp_tld,
    }

def count_keywords(text: str, vocab: list[str]) -> int:
    if not text: return 0
    low = text.lower()
    return sum(1 for w in vocab if w in low)

def html_title_meta_htext(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        buf=[]
        if soup.title: buf.append(str(soup.title.string or ""))
        for tag in soup.find_all(["h1","h2","meta"]):
            if tag.name=="meta" and tag.get("content"): buf.append(str(tag["content"]))
            elif tag.string: buf.append(str(tag.string))
        return " ".join(buf)
    except Exception:
        return ""

def is_parked_like(html: str, status: int) -> bool:
    if status >= 400: return True
    if not html or len(html) < 300: return True
    low = html.lower()
    return any(k in low for k in PARKED_KEYWORDS)

# ---------------- semantic (batched) ----------------
_sem_model = None
_cse_names = list(CSE_DEFS.keys())
_cse_embeds = None

def ensure_sem():
    global _sem_model, _cse_embeds
    if _sem_model is None:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import util as _util  # noqa
        logger.info("🔹 Loading sentence-transformers/paraphrase-MiniLM-L6-v2 (batched)…")
        _sem_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
        _cse_embeds = _sem_model.encode(_cse_names, convert_to_tensor=True, normalize_embeddings=True)

def batch_semantic(domains: list[str], batch_size: int = 256) -> dict[str, tuple[str, float]]:
    """
    Returns {domain: (best_cse, score)} using one batched forward pass.
    """
    from sentence_transformers import util as _util
    ensure_sem()
    # clean tokens
    tokens = [re.sub(r"[^a-zA-Z0-9]+", " ", d) for d in domains]
    embs = _sem_model.encode(tokens, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
    sims = _util.cos_sim(embs, _cse_embeds)  # shape: [N, C]
    out = {}
    for i, d in enumerate(domains):
        row = sims[i]
        best_idx = int(row.argmax())
        best = float(row[best_idx])
        out[d] = (_cse_names[best_idx], best)
    return out

# ---------------- aiohttp helpers ----------------
async def fetch_text(session, url: str):
    for _ in range(2):
        try:
            async with session.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True) as r:
                status=r.status; final=str(r.url)
                ctype=(r.headers.get("Content-Type") or "").lower()
                text = await r.text(errors="ignore") if "text/html" in ctype else ""
                return status, final, text
        except asyncio.TimeoutError:
            await asyncio.sleep(0.7)
        except Exception:
            return 0, url, ""
    return 0, url, ""

async def fetch_bytes(session, url: str):
    for _ in range(2):
        try:
            async with session.get(url, headers=HEADERS, timeout=TIMEOUT) as r:
                if r.status==200:
                    return await r.read()
        except asyncio.TimeoutError:
            await asyncio.sleep(0.6)
        except Exception:
            return None
    return None

def extract_favicon_url(base_url: str, html: str) -> str:
    try:
        soup=BeautifulSoup(html,"html.parser")
        link=soup.find("link",rel=lambda v: v and "icon" in v.lower())
        if link and link.get("href"): return urljoin(base_url, link["href"].strip())
    except Exception: pass
    try:
        parsed=urlparse(base_url); return f"{parsed.scheme}://{parsed.netloc}/favicon.ico"
    except Exception:
        return ""

def phash_from_bytes(b: bytes):
    try:
        img=Image.open(io.BytesIO(b)).convert("RGBA")
        bg=Image.new("RGB", img.size, (255,255,255))
        bg.paste(img, mask=img.split()[-1] if img.mode=="RGBA" else None)
        return str(imagehash.phash(bg))
    except Exception:
        return None

def phash_hd(h1: str, h2: str) -> int:
    try:
        return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)
    except Exception:
        return 999

# ---------------- RDAP ----------------
async def rdap_fetch_age_org_reg(session, domain: str):
    url=f"https://rdap.org/domain/{domain}"
    out={"Domain_Age_Days":None,"RDAP_Org":"","RDAP_Registrar":""}
    try:
        async with session.get(url, headers=HEADERS, timeout=TIMEOUT) as r:
            if r.status!=200: return out
            data=await r.json(content_type=None)
    except Exception:
        return out

    # creation date → age
    try:
        events=data.get("events",[])
        created=None
        for e in events:
            if e.get("eventAction")=="registration":
                created=e.get("eventDate"); break
        if created:
            dt=datetime.fromisoformat(created.replace("Z","+00:00"))
            out["Domain_Age_Days"]=max(0, (datetime.now(timezone.utc)-dt).days)
    except Exception:
        pass

    # entities → org / registrar
    try:
        for ent in data.get("entities",[]):
            roles=ent.get("roles",[])
            if "registrar" in roles and not out["RDAP_Registrar"]:
                out["RDAP_Registrar"]=ent.get("handle") or ""
            if any(r in roles for r in ["registrant","administrative","technical"]) and not out["RDAP_Org"]:
                vca=ent.get("vcardArray",[])
                if len(vca)==2:
                    for row in vca[1]:
                        if row[0]=="org": out["RDAP_Org"]=row[3]; break
                        if row[0]=="fn" and not out["RDAP_Org"]: out["RDAP_Org"]=row[3]
    except Exception:
        pass
    return out

# ---------------- favicon library for official CSE sites ----------------
async def build_official_favicon_library(session, cache: dict):
    key="__official_favicons__"
    if key in cache: return cache[key]
    lib={cse:[] for cse in CSE_DEFS}
    for cse, meta in CSE_DEFS.items():
        for off in meta["official_domains"]:
            for scheme in ("https","http"):
                fav_url=f"{scheme}://{off}/favicon.ico"
                try:
                    async with session.get(fav_url, headers=HEADERS, timeout=TIMEOUT) as r:
                        if r.status==200:
                            b=await r.read()
                            ph=phash_from_bytes(b)
                            if ph and ph not in lib[cse]: lib[cse].append(ph)
                except Exception:
                    pass
    cache[key]=lib; save_json(CACHE_FILE, cache)
    return lib

# ---------------- scoring ----------------
def keyword_url_cse(host: str):
    low=host.lower()
    for cse, meta in CSE_DEFS.items():
        if any(off in low for off in meta["official_domains"]): return cse, "URL"
    for cse, meta in CSE_DEFS.items():
        if any(kw in low for kw in meta["keywords"]): return cse, "URL"
    return None, None

def cse_from_text(text: str):
    low=(text or "").lower()
    for cse, meta in CSE_DEFS.items():
        for off in meta["official_domains"]:
            if off in low: return cse
    for cse, meta in CSE_DEFS.items():
        for kw in meta["keywords"]:
            if kw in low: return cse
    return None

def combine_scores(url_hit, sem_hit, content_hit, fav_hit, rdap_hit):
    # precision-first weights
    weights={}; types={c:set() for c in CSE_DEFS.keys()}
    if url_hit:
        weights[url_hit]=weights.get(url_hit,0)+1.0; types[url_hit].add("URL")
    sem_cse, sem_score = sem_hit
    if sem_cse and sem_score>=SEM_THRESHOLD_PRIMARY:
        weights[sem_cse]=weights.get(sem_cse,0)+float(sem_score); types[sem_cse].add("Semantic")
    if content_hit:
        weights[content_hit]=weights.get(content_hit,0)+0.5; types[content_hit].add("Content")
    fav_cse, fav_hd = fav_hit
    if fav_cse is not None and fav_hd<=FAV_HD_MAX:
        weights[fav_cse]=weights.get(fav_cse,0)+1.2; types[fav_cse].add("Favicon")
    if rdap_hit:
        weights[rdap_hit]=weights.get(rdap_hit,0)+1.0; types[rdap_hit].add("RDAP")

    if not weights: return None, 0.0, []
    best=max(weights, key=weights.get)
    return best, round(float(weights[best]),3), sorted(types[best])

# ---------------- per-domain analysis ----------------
async def analyze(session, d: str, cache: dict, fav_lib: dict,
                  sem_lookup: dict[str, tuple[str,float]]):
    if not d: return None
    if d in cache:
        row=cache[d]
        if row and row.get("Status")=="related": return row
        return None

    # quick HTML (https → http)
    async def best_html(domain):
        for scheme in ("https","http"):
            st,final,html=await fetch_text(session, f"{scheme}://{domain}")
            if st and (html or st<400): return st,final,html
        return 0,"",""
    status, final_url, html = await best_html(d)

    # parked?
    if is_parked_like(html, status):
        cache[d]={"Domain":d,"Related_CSE":"","Relation_Score":0.0,"Match_Types":"",
                  "Status":"parked/inactive","Notes":"",
                  "Final_URL":final_url,"HTTP_Status":status,"Title":"",
                  "Favicon_URL":"","RDAP_Org":"","RDAP_Registrar":"",
                  "First_Seen_ISO":datetime.now(timezone.utc).isoformat(),
                  **url_lexical_features(d),
                  "Domain_Age_Days":None,
                  "Content_CSE_Keyword_Count":0,
                  "Content_Phish_Keyword_Count":0,
                  "Semantic_Score":0.0,
                  "Favicon_HD":999}
        return None

    lex=url_lexical_features(d)
    url_cse,_=keyword_url_cse(d)

    # batched semantics (already computed)
    sem_cse, sem_score = sem_lookup.get(d, ("", 0.0))

    # content counts + hint (only if we got HTML)
    content=html_title_meta_htext(html)
    cse_cont=cse_from_text(content)
    cse_kw_count = sum(count_keywords(content, CSE_DEFS[k]["keywords"]) for k in CSE_DEFS)
    phish_kw_count = count_keywords(content, PHISH_HINTS)

    # CONDITIONAL heavy checks (favicon + RDAP) only if likely related
    do_heavy = bool(url_cse) or (sem_score >= SEM_THRESHOLD_PRIMARY) or bool(cse_cont)
    fav_url=""; fav_ph=None; fav_hit=(None,999)
    rdap={"Domain_Age_Days":None,"RDAP_Org":"","RDAP_Registrar":""}
    if do_heavy:
        try:
            fav_url=extract_favicon_url(final_url or f"https://{d}", html)
            if fav_url:
                b=await fetch_bytes(session, fav_url)
                if b:
                    fav_ph=phash_from_bytes(b)
                    if fav_ph:
                        best_cse=None; best_hd=999
                        for cse, phashes in fav_lib.items():
                            for off_ph in phashes:
                                hd=phash_hd(fav_ph, off_ph)
                                if hd<best_hd: best_hd=hd; best_cse=cse
                        fav_hit=(best_cse,best_hd)
        except Exception:
            pass
        rdap = await rdap_fetch_age_org_reg(session, d)

    rdap_cse=cse_from_text(f"{rdap.get('RDAP_Org','')} {rdap.get('RDAP_Registrar','')}")

    # combine
    best_cse, score, mtypes = combine_scores(url_cse, (sem_cse,sem_score), cse_cont, fav_hit, rdap_cse)
    base_row = {
        "Domain": d,
        "Related_CSE": best_cse or "",
        "Relation_Score": score,
        "Match_Types": ",".join(mtypes) if best_cse else "",
        "Status": "related" if best_cse else "no_relation",
        "Notes": "",
        "Final_URL": final_url,
        "HTTP_Status": status,
        "Title": (BeautifulSoup(html,"html.parser").title.string if BeautifulSoup(html,"html.parser").title else ""),
        "Favicon_URL": fav_url or "",
        "RDAP_Org": rdap.get("RDAP_Org",""),
        "RDAP_Registrar": rdap.get("RDAP_Registrar",""),
        "First_Seen_ISO": datetime.now(timezone.utc).isoformat(),
        **lex,
        "Domain_Age_Days": rdap.get("Domain_Age_Days"),
        "Content_CSE_Keyword_Count": int(cse_kw_count),
        "Content_Phish_Keyword_Count": int(phish_kw_count),
        "Semantic_Score": float(sem_score),
        "Favicon_HD": int(fav_hit[1]) if fav_hit[1] is not None else 999
    }
    cache[d]=base_row
    return base_row if best_cse else None

# ---------------- run ----------------
async def run(input_csv: Path, concurrency: int, save_interval: int):
    if not input_csv.exists():
        logger.error(f"Missing input: {input_csv}"); return
    src=pd.read_csv(input_csv)
    if "Domain" not in src.columns:
        logger.error("Input CSV must have a 'Domain' column"); return
    domains=list(src["Domain"].dropna().astype(str).str.lower().unique())
    total=len(domains)
    print("domains length" + str(len(domains)))

    cache=load_json(CACHE_FILE,{})
    rows_out=[]; done=set()
    if OUT_FILE.exists():
        try:
            prev=pd.read_csv(OUT_FILE)
            rows_out=prev.to_dict(orient="records")
            done=set(prev["Domain"].astype(str).str.lower())
            logger.info(f"⏩ Resuming: {len(done)} already saved")
        except Exception: pass

    # Filter pending + normalize
    pending=[normalize_host(d) for d in domains if normalize_host(d) not in done]
    pending=[p for p in pending if p]  # remove empties
    logger.info(f"Total={total} | Pending={len(pending)} | Concurrency={concurrency}")

    # Precompute semantics (batched) for pending domains
    logger.info("🧠 Precomputing semantic embeddings (batched)…")
    sem_lookup = batch_semantic(pending, batch_size=256)

    connector=aiohttp.TCPConnector(
        limit=concurrency*2, ssl=False, ttl_dns_cache=600,
        force_close=False, enable_cleanup_closed=True
    )
    async with aiohttp.ClientSession(headers=HEADERS, timeout=TIMEOUT, connector=connector) as session:
        fav_lib=await build_official_favicon_library(session, cache)
        logger.info("✅ Official favicon library: " + ", ".join(f"{k}:{len(v)}" for k,v in fav_lib.items()))
        sem=asyncio.Semaphore(concurrency)

        processed=0; wrote_since=0; t0=time.time()

        async def task(h):
            nonlocal processed, wrote_since
            async with sem:
                try:
                    row=await analyze(session, h, cache, fav_lib, sem_lookup)
                except Exception as e:
                    logger.warning(f"task error {h}: {e}")
                    row=None
                processed+=1
                if row:
                    rows_out.append(row); wrote_since+=1
                if processed % 1000 == 0:
                    rate = processed / max(1,(time.time()-t0)/60)
                    logger.info(f"… progress {processed}/{len(pending)} | out={len(rows_out)} | cache={len(cache)} | ~{rate:.0f} domains/min")

        batch=[]
        for i, dom in enumerate(pending,1):
            batch.append(asyncio.create_task(task(dom)))
            if len(batch) >= concurrency*2 or i==len(pending):
                for t in asyncio.as_completed(batch):
                    await t
                    if wrote_since >= save_interval:
                        pd.DataFrame(rows_out).drop_duplicates(subset=["Domain"]).to_csv(OUT_FILE,index=False)
                        save_json(CACHE_FILE, cache)
                        wrote_since=0
                        elapsed=(time.time()-t0)/60
                        logger.info(f"💾 checkpoint: rows={len(rows_out)} cache={len(cache)} time={elapsed:.1f}m")
                batch=[]

        # final save
        pd.DataFrame(rows_out).drop_duplicates(subset=["Domain"]).to_csv(OUT_FILE,index=False)
        save_json(CACHE_FILE, cache)
        logger.info(f"✅ done: wrote {len(rows_out)} rows to {OUT_FILE}; cache={len(cache)}")

def cancel_handler(sig, frame):
    logger.info("🛑 Signal received. Exiting after last checkpoint…")
    sys.exit(0)

if __name__=="__main__":
    import argparse
    signal.signal(signal.SIGINT, cancel_handler)
    signal.signal(signal.SIGTERM, cancel_handler)

    ap=argparse.ArgumentParser(description="CSE Relevance Classifier – FAST")
    ap.add_argument("--input", type=str, default=str(IN_FILE), help="Input CSV with a 'Domain' column")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    args=ap.parse_args()

    IN_FILE = Path(args.input)
    asyncio.run(run(IN_FILE, args.concurrency, args.save_interval))
