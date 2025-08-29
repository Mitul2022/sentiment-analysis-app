import collections
import re
import string
import textwrap
import tempfile
import os
import time
from collections import defaultdict, Counter
from io import BytesIO
import html
import pandas as pd
import numpy as np
import spacy
import matplotlib
matplotlib.use("Agg")  # headless, faster, avoids GUI overhead
import matplotlib.pyplot as plt
from fpdf import FPDF
import gradio as gr
import streamlit as st

from rake_nltk import Rake
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import difflib
from nltk.corpus import wordnet as wn

import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

from rake_nltk import Rake

# Global RAKE instance (reuse across functions)
RAKE = Rake(min_length=1, max_length=3)

#####################
# Dependency checks #
#####################
# Download NLTK resources if missing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Ensure required NLTK data is downloaded
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Optimized: disable unused spaCy components (NER, textcat) for big speedup
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

sia = SentimentIntensityAnalyzer()

# Safer VADER caching: don't cache very long texts (avoid memory blowup)
from functools import lru_cache

def _vader_uncached(text: str) -> float:
    return sia.polarity_scores(text)["compound"]

@lru_cache(maxsize=50_000)
def _vader_cached(text: str) -> float:
    return _vader_uncached(text)

def vader_score(text: str) -> float:
    s = "" if text is None else str(text).strip()
    # only cache short/medium strings
    if len(s) <= 200:
        return _vader_cached(s)
    return _vader_uncached(s)

# Precompile commonly used regexes
_RE_NONWORD = re.compile(r'[^\w\s]')
_RE_MULTISPACE = re.compile(r'\s+')
_RE_PAGE = re.compile(r'page\s*\d+', re.I)

ARTICLES = {'the', 'a', 'an'}
DEMONSTRATORS = {'this', 'that', 'these', 'those'}

# Small helper for spaCy piping with n_process fallback
def _get_nlp_pipe(texts, batch_size=64, n_process=None):
    try:
        if n_process is None:
            n_process = max(1, (os.cpu_count() or 2) - 1)
        return nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
    except TypeError:
        # spaCy older versions or platforms without multi-process support
        return nlp.pipe(texts, batch_size=batch_size)

# -------------------------
# Text utilities
# -------------------------
def clean_phrase(phrase: str) -> str:
    phrase = _RE_NONWORD.sub('', (phrase or "").lower()).strip()
    tokens = phrase.split()
    while tokens and tokens[0] in ARTICLES.union(DEMONSTRATORS):
        tokens.pop(0)
    if not tokens:
        return ""
    # make_doc avoids running the full pipeline (fast)
    doc = nlp.make_doc(' '.join(tokens))
    lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    return ' '.join(lemmas).strip()

def clean_text_for_pdf(text):
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        'Ã¢â‚¬Â¦': '...', 'Ã¢â‚¬â€': '-', 'Ã¢â‚¬â€œ': '-', 'Ã¢Ë†â€™': '-', '\u2212': '-',
        'Ã¢â‚¬Ëœ': "'", 'Ã¢â‚¬â„¢': "'", 'Ã¢â‚¬Å“': '"', 'Ã¢â‚¬Â': '"',
        'Ã¢â‚¬Â¢': '-', 'Ã¢â‚¬â€™': '-', 'Ã¢â‚¬â€¢': '-',
        'Ã¢â‚¬Â³': '"', 'Ã¢â‚¬Â²': "'", '\u2014': '-', '\u2013': '-',
        '\u2010': '-', '\u00A0': ' ', '\u202F': ' ', '\u2009': ' ', '\u200A': ' ',
        '\u2022': '-', '\u2032': "'", '\u2033': '"', '\t': ' '
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = ''.join(ch for ch in text if ch.isprintable() or ch in ['\n', '\t', '\r'])
    return text.strip()

def safe_quote(q):
    q = str(q)
    if len(q) > 300:
        q = q[:297] + "..."
    q = re.sub(r"(\w{20,})", lambda m: ' '.join(textwrap.wrap(m.group(0), 20)), q)
    q = _RE_MULTISPACE.sub(' ', q)
    return q

def perfectly_format_numbered_reviews(text, width=85, indent=' '):
    text = re.sub(r'\n{2,}', '\n', text)
    numbered = re.split(r'(?m)^\s*(\d+\.)', text)
    output = []
    i = 1
    while i < len(numbered):
        num = numbered[i].strip()
        rest = numbered[i + 1].strip()
        lines = rest.split('\n')
        wrapped_first = textwrap.fill(f"{num} {lines[0]}", width=width) if lines else f"{num} "
        other_lines = [line for line in lines[1:] if line.strip()]
        wrapped_others = [
            textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
            for line in other_lines
        ]
        combined = [wrapped_first] + wrapped_others
        output.append('\n'.join(combined))
        i += 2
    return '\n'.join(output)

# -------------------------
# Data helpers
# -------------------------
def auto_detect_column(df, candidates):
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    return None

def auto_detect_review_column(df):
    candidates = [
        'review', 'review_text', 'review content', 'feedback', 'comment', 'text',
        'body', 'message', 'content', 'remarks'
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) == 0:
        raise ValueError("No text column found for analysis.")
    lengths = df[obj_cols].apply(lambda col: col.fillna("").astype(str).map(len).mean())
    return lengths.idxmax()

def auto_detect_nps_column(df):
    candidates = ['nps', 'nps_score', 'score', 'rating', 'net promoter score']
    for cand in candidates:
        col = auto_detect_column(df, [cand])
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return col
    numcols = df.select_dtypes(include=[np.number]).columns
    for col in numcols:
        vals = df[col].dropna()
        if not vals.empty and (vals.between(0, 10).mean() > 0.9):
            return col
    return None

def detect_language_of_reviews(df, review_col):
    try:
        from langdetect import detect
    except ImportError:
        raise ImportError("Please install 'langdetect'; pip install langdetect")
    sample_texts = df[review_col].dropna().astype(str).sample(min(20, len(df)), random_state=42)
    lang_counts = Counter(detect(t) for t in sample_texts if t.strip())
    return lang_counts.most_common(1)[0][0] if lang_counts else "unknown"

def safe_read_csv(file, **kwargs):
    filename = getattr(file, 'name', file)
    if isinstance(filename, str) and filename.endswith('.xlsx'):
        try:
            if hasattr(file, 'file'):
                return pd.read_excel(file.file, **kwargs)
            else:
                return pd.read_excel(file, **kwargs)
        except ImportError:
            raise ImportError("Install 'openpyxl': pip install openpyxl")
        except Exception as e:
            raise RuntimeError(f"Error reading Excel file: {e}")
    else:
        try:
            if hasattr(file, 'file'):
                return pd.read_csv(file.file, **kwargs)
            else:
                return pd.read_csv(file, **kwargs)
        except UnicodeDecodeError:
            try:
                if hasattr(file, 'file'):
                    return pd.read_csv(file.file, encoding='utf-8-sig', **kwargs)
                else:
                    return pd.read_csv(file, encoding='utf-8-sig', **kwargs)
            except UnicodeDecodeError:
                try:
                    if hasattr(file, 'file'):
                        return pd.read_csv(file.file, encoding='latin1', **kwargs)
                    else:
                        return pd.read_csv(file, encoding='latin1', **kwargs)
                except Exception as e:
                    raise RuntimeError(f"Error reading CSV file: {e}")

def limit_large_df(df):
    return df

def normalize_nps_scores(nps_series):
    if nps_series.empty:
        return nps_series
    max_val = nps_series.max()
    if max_val <= 5:
        return nps_series * 2
    elif max_val > 10 and max_val <= 100:
        return nps_series / 10
    return nps_series

def aggregate_sentiment(counts):
    pos = counts.get('Positive', 0)
    neu = counts.get('Neutral', 0)
    neg = counts.get('Negative', 0)
    if pos > neu and pos > neg:
        return 'Positive'
    elif neg > pos and neg > neu:
        return 'Negative'
    else:
        return 'Neutral'

def aggregate_aspect_sentiments(occurrences):
    sentiments = [s for _, s, _ in occurrences]
    return aggregate_sentiment(Counter(sentiments))

def _normalize_user_aspects(user_aspects):
    if isinstance(user_aspects, str):
        aspects = [x.strip().lower() for x in user_aspects.split(",") if x.strip()]
    else:
        aspects = [x.strip().lower() for x in user_aspects if x.strip()]
    return aspects

# -------------------------
# Aspect expansion helpers (returns dict with variants + compiled pattern)
# -------------------------
def expand_aspect_variants(aspects, max_synonyms=6):
    variants = {}
    for asp in aspects:
        base = asp.lower().strip()
        lemm = clean_phrase(base)
        cand_set = {base, lemm}

        # Add simple plural
        if not base.endswith('s'):
            cand_set.add(base + 's')

        # Add split forms for phrases
        if ' ' in base:
            tokens = base.split()
            cand_set.update(tokens)
            cand_set.add(' '.join(tokens))

        # WordNet synonyms (filter by relevance)
        for pos in ('n', 'a', 'v'):
            try:
                synsets = wn.synsets(base, pos=pos)
            except Exception:
                synsets = []
            for s in synsets[:max_synonyms]:
                for lemma in s.lemmas():
                    cand = lemma.name().replace('_', ' ').lower()
                    if cand and len(cand) > 2 and cand != base:
                        cand_set.add(cand)

        # Clean all candidates
        cleaned = {clean_phrase(c) for c in cand_set if c}
        cleaned = {c for c in cleaned if len(c) > 1}
        cleaned_list = sorted(cleaned)

        # build combined regex pattern (longer first to avoid substring matches)
        pattern = None
        if cleaned_list:
            alt = "|".join(re.escape(v) for v in sorted(cleaned_list, key=len, reverse=True))
            try:
                pattern = re.compile(r"\b(?:" + alt + r")\b", flags=re.IGNORECASE)
            except re.error:
                pattern = None

        variants[base] = {"variants": cleaned_list, "pattern": pattern}
    return variants

# -------------------------
# Matching helper (uses compiled pattern + fuzzy token checks)
# -------------------------
def match_aspect_in_text(text, aspect_info, fuzzy_threshold=0.85):
    """
    aspect_info: either list of variants OR dict {"variants": [...], "pattern": compiled_re}
    """
    if not text or not aspect_info:
        return False
    text_lc = text.lower()

    # normalize aspect_info
    if isinstance(aspect_info, dict):
        pattern = aspect_info.get("pattern")
        variants = aspect_info.get("variants", [])
    else:
        pattern = None
        variants = aspect_info

    # exact / regex match first
    if pattern:
        if pattern.search(text):
            return True

    # token-level fuzzy matching (iterate tokens which are usually fewer)
    tokens = re.findall(r"\w+", text_lc)
    if not tokens or not variants:
        return False

    # try rapidfuzz for speed if available
    try:
        from rapidfuzz import process as rf_process, fuzz as rf_fuzz
        # rapidfuzz returns tuple (match, score, idx)
        score_cutoff = int(fuzzy_threshold * 100)
        for tok in tokens:
            match = rf_process.extractOne(
                tok, variants, scorer=rf_fuzz.QRatio, score_cutoff=score_cutoff
            )
            if match:
                return True
    except Exception:
        # fallback to difflib
        for tok in tokens:
            close = difflib.get_close_matches(tok, variants, n=1, cutoff=fuzzy_threshold)
            if close:
                return True

    return False

# -------------------------
# Auto-detect aspects via RAKE (optimized)
# -------------------------
def auto_detect_aspects_via_rake(df, review_col, top_n=10):
    text_blob = " \n ".join(df[review_col].dropna().astype(str).tolist())
    if not text_blob.strip():
        return []

    # Use global RAKE instance
    RAKE.extract_keywords_from_text(text_blob)
    phrases = RAKE.get_ranked_phrases()[:200]

    cand_counter = Counter()
    for ph in phrases:
        ph_clean = _RE_NONWORD.sub('', ph.lower()).strip()
        doc = nlp.make_doc(ph_clean)
        # we only need POS-like signals; use heuristics on tokens
        if any(tok.is_alpha() for tok in doc):
            cand_counter[ph_clean] += 1

    # Add noun chunks from sample reviews using nlp.pipe
    sample_texts = df[review_col].dropna().astype(str).sample(min(200, len(df)), random_state=42).tolist()
    for doc in _get_nlp_pipe(sample_texts, batch_size=32):
        for nc in getattr(doc, "noun_chunks", []):
            nc_text = _RE_NONWORD.sub('', nc.text.lower()).strip()
            if 1 <= len(nc_text.split()) <= 3:
                cand_counter[nc_text] += 1

    # Fallback to frequent words if nothing detected
    if not cand_counter:
        words = re.findall(r"\w+", text_blob.lower())
        counts = Counter(words).most_common(200)
        for w, c in counts:
            if len(w) > 3 and w.isalpha():
                cand_counter[w] += c

    candidates = [k for k, _ in cand_counter.most_common(top_n)]
    normalized = []
    seen = set()
    for c in candidates:
        n = clean_phrase(c)
        if n and n not in seen:
            normalized.append(n)
            seen.add(n)
    return normalized[:top_n]

# -------------------------
# Core extraction (accepts parsed doc or raw text; uses precomputed variants_map)
# -------------------------
def extract_dynamic_aspects_user(doc_or_text, user_aspects, nps_val=None, variants_map=None):
    """
    Accepts either a spaCy doc (preferred) or raw text. Uses variants_map (returned by expand_aspect_variants)
    to avoid recomputing synonyms per review.
    """
    # parse if needed
    if isinstance(doc_or_text, str):
        doc = nlp(str(doc_or_text))
    else:
        doc = doc_or_text

    # Auto-detect aspects if user_aspects is empty
    if not user_aspects:
        noun_chunks = [chunk.lemma_.lower() for chunk in getattr(doc, "noun_chunks", []) if len(chunk.text.strip()) > 2]
        user_aspects = list(set(noun_chunks[:10]))

    user_aspects = _normalize_user_aspects(user_aspects)

    if variants_map is None:
        variants_map = expand_aspect_variants(user_aspects)

    extracted = defaultdict(list)

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # Base sentiment from VADER
        sent_score = vader_score(sent_text)
        if sent_score >= 0.3:
            base_label = "Positive"
        elif sent_score <= -0.3:
            base_label = "Negative"
        else:
            base_label = "Neutral"

        # Hybrid adjustment with NPS
        final_label = base_label
        sentiment_source = "Text"
        mismatch_flag = False

        if nps_val is not None and 0 <= nps_val <= 10:
            if nps_val >= 9:  # Promoter
                if base_label == "Negative" and sent_score <= -0.7:
                    mismatch_flag = True
                else:
                    final_label = "Positive"
                    sentiment_source = "Hybrid (NPS Override)"
            elif nps_val <= 6:  # Detractor
                if base_label == "Positive" and sent_score >= 0.7:
                    mismatch_flag = True
                else:
                    final_label = "Negative"
                    sentiment_source = "Hybrid (NPS Override)"
            elif nps_val in [7, 8]:
                sentiment_source = "Hybrid (NPS Considered)"

        # Match aspects within sentence
        for aspect in user_aspects:
            aspect_info = variants_map.get(aspect)
            # if variants_map entry missing, fallback to simple list with aspect
            if aspect_info is None:
                aspect_info = {"variants": [aspect], "pattern": re.compile(r"\b" + re.escape(aspect) + r"\b", flags=re.IGNORECASE)}
            if match_aspect_in_text(sent_text, aspect_info):
                aspect_key = aspect.lower().strip()
                extracted[aspect_key].append((sent_text, final_label, sent_text, sentiment_source, mismatch_flag))

    # Build structured data
    data = []
    for aspect, mentions in extracted.items():
        agg_sent = aggregate_aspect_sentiments([(m[1], m[1], m[1]) for m in mentions])
        context = "; ".join(sorted(set(m[0] for m in mentions)))
        quotes = [m[2] for m in mentions]
        mismatch = any(m[4] for m in mentions)
        sources = list(set(m[3] for m in mentions))
        data.append({
            "Review_ID": None,
            "Review": None,
            "Aspect": aspect,
            "Aspect_Sentiment": agg_sent,
            "Aspect_Context": context,
            "Quotes": quotes,
            "Sentiment_Source": ", ".join(sources),
            "Sentiment_NPS_Mismatch": mismatch
        })

    return data

# -------------------------
# Analysis pipeline
# -------------------------
def normalize_single_nps_value(nps_val):
    try:
        nps_val = float(nps_val)
        if nps_val < 0:
            return None
        if nps_val <= 5:
            return nps_val * 2
        elif nps_val > 10 and nps_val <= 100:
            return nps_val / 10
        return nps_val
    except (ValueError, TypeError):
        return None

def analyze_review_structured(df, review_col, nps_col=None, user_aspects=None, process_times=None):
    if review_col not in df.columns:
        raise ValueError(f"Review column '{review_col}' not found.")
    if nps_col and nps_col not in df.columns:
        raise ValueError(f"NPS column '{nps_col}' not found.")
    if df[review_col].isnull().all():
        raise ValueError(f"All entries in review column '{review_col}' are empty.")

    t0 = time.time()

    # Auto-detect aspects if user passed blank or asked for 'auto'
    if not user_aspects or (isinstance(user_aspects, str) and not user_aspects.strip()):
        detected = auto_detect_aspects_via_rake(df, review_col, top_n=10)
        if not detected:
            detected = ["delivery", "price", "quality", "service"]
        user_aspects = detected
    else:
        user_aspects = _normalize_user_aspects(user_aspects)

    if not user_aspects:
        raise ValueError("Please provide at least one aspect or allow auto-detection by passing blank or 'auto'.")

    # Precompute aspect variants once
    variants_map = expand_aspect_variants(user_aspects)

    records = []
    total = len(df)

    # placeholder for Streamlit progress
    try:
        placeholder = st.empty()
    except Exception:
        placeholder = None

    # Prepare texts and docs (batched)
    texts = df[review_col].fillna("").astype(str).tolist()
    docs = _get_nlp_pipe(texts, batch_size=64)

    next_progress = 0.05  # progress update every 5%

    # iterate docs and rows together
    # Use itertuples-like access for speed; df.itertuples returns namedtuples in same column order
    for idx, (doc, row) in enumerate(zip(docs, df.itertuples(index=False, name=None)), start=1):
        text = doc.text.strip()
        if not text:
            continue

        # normalize NPS value
        nps_val = None
        if nps_col:
            try:
                # Try to fetch by attribute name (if namedtuple had field names)
                # This may not work for nameless tuples; use df.at fallback below.
                nps_val = getattr(row, nps_col)
            except Exception:
                try:
                    nps_val = df.at[df.index[idx-1], nps_col]
                except Exception:
                    nps_val = None
        nps_val = normalize_single_nps_value(nps_val)

        # extract aspects & sentiments using hybrid logic (pass parsed doc & variants_map)
        aspects = extract_dynamic_aspects_user(doc, user_aspects, nps_val=nps_val, variants_map=variants_map)

        for asp in aspects:
            asp["Review_ID"] = idx
            asp["Review"] = text
            asp["NPS_Score"] = nps_val
            for field in ['supplier_name', 'product_fullname']:
                if field in df.columns:
                    try:
                        asp[field] = df.at[df.index[idx-1], field]
                    except Exception:
                        asp[field] = None
            records.append(asp)

        # throttle progress updates
        if placeholder is not None and (idx / total) >= next_progress:
            try:
                placeholder.text(f"Processed {idx:,}/{total:,} reviews...")
            except Exception:
                pass
            next_progress += 0.05

    if process_times is not None:
        process_times["aspect_sentiment_extraction"] = round(time.time() - t0, 2)

    return pd.DataFrame(records)

# -------------------------
# Grouping / summarization
# -------------------------
def groupby_supplier_product(detail_df, user_aspects):
    user_aspects = _normalize_user_aspects(user_aspects)
    group_cols = [col for col in ['supplier_name', 'product_fullname'] if col in detail_df.columns]
    if not group_cols:
        return pd.DataFrame()
    gb = detail_df.groupby(group_cols + ["Aspect"])
    rows = []
    for keys, group in gb:
        keys = keys if isinstance(keys, tuple) else (keys,)
        aspect = keys[-1]
        if aspect.lower() not in user_aspects:
            continue
        rec = {c: v for c, v in zip(group_cols, keys[:len(group_cols)])}
        rec["Aspect"] = aspect
        counts = Counter(group["Aspect_Sentiment"])
        rec.update({
            "Mentions": len(group),
            "Positive": counts.get("Positive", 0),
            "Neutral": counts.get("Neutral", 0),
            "Negative": counts.get("Negative", 0)
        })
        valid_nps = normalize_nps_scores(group["NPS_Score"].dropna())
        if not valid_nps.empty:
            rec["Avg_NPS"] = round(valid_nps.mean(), 2)
            rec["Promoters"] = valid_nps[(valid_nps >= 9) & (valid_nps <= 10)].count()
            rec["Passives"] = valid_nps[(valid_nps >= 7) & (valid_nps <= 8)].count()
            rec["Detractors"] = valid_nps[(valid_nps >= 0) & (valid_nps <= 6)].count()
        else:
            rec["Avg_NPS"] = "N/A"
            rec["Promoters"] = rec["Passives"] = rec["Detractors"] = 0
        rows.append(rec)
    return pd.DataFrame(rows)

def generate_sentiment_summary(df):
    if df.empty:
        return pd.DataFrame(columns=[
            "Aspect", "Total Mentions", "Positive", "Neutral", "Negative",
            "Positive (%)", "Neutral (%)", "Negative (%)", "Dominant Sentiment",
            "Avg NPS", "Promoters", "Passives", "Detractors", "Sample Quotes"
        ])

    summary_rows = []
    grouped = df.groupby("Aspect")
    for aspect, group in grouped:
        counts = Counter(group["Aspect_Sentiment"])
        total = len(group)
        pos = counts.get("Positive", 0)
        neu = counts.get("Neutral", 0)
        neg = counts.get("Negative", 0)
        pos_pct = total and pos / total * 100 or 0
        neu_pct = total and neu / total * 100 or 0
        neg_pct = total and neg / total * 100 or 0
        dominant = aggregate_sentiment(counts)

        avg_nps = promoters = passives = detractors = 0
        if "NPS_Score" in group.columns and group["NPS_Score"].notna().any():
            valid_nps = normalize_nps_scores(group["NPS_Score"].dropna())
            avg_nps = valid_nps.mean() if not valid_nps.empty else np.nan
            promoters = valid_nps[(valid_nps >= 9) & (valid_nps <= 10)].count()
            passives = valid_nps[(valid_nps >= 7) & (valid_nps <= 8)].count()
            detractors = valid_nps[(valid_nps >= 0) & (valid_nps <= 6)].count()

        raw_quotes = []
        for quotes_cell in group["Quotes"]:
            if isinstance(quotes_cell, list):
                raw_quotes.extend(quotes_cell)
            elif isinstance(quotes_cell, str):
                raw_quotes.append(quotes_cell)

        unique_quotes = []
        seen = set()
        for q in raw_quotes:
            q_clean = q.strip()
            if q_clean and q_clean not in seen:
                unique_quotes.append(q_clean)
                seen.add(q_clean)
            if len(unique_quotes) >= 3:
                break

        summary_rows.append({
            "Aspect": aspect,
            "Total Mentions": total,
            "Positive": pos,
            "Neutral": neu,
            "Negative": neg,
            "Positive (%)": round(pos_pct, 2),
            "Neutral (%)": round(neu_pct, 2),
            "Negative (%)": round(neg_pct, 2),
            "Dominant Sentiment": dominant,
            "Avg NPS": round(avg_nps, 2) if not pd.isna(avg_nps) else "N/A",
            "Promoters": promoters,
            "Passives": passives,
            "Detractors": detractors,
            "Sample Quotes": unique_quotes,
        })
    return pd.DataFrame(summary_rows).sort_values(by="Total Mentions", ascending=False)

def benchmark_kpis(df_summary, df_detail=None):
    total_mentions = int(df_summary["Total Mentions"].sum()) if not df_summary.empty else 0
    pos_total = int(df_summary["Positive"].sum()) if not df_summary.empty else 0
    neu_total = int(df_summary["Neutral"].sum()) if not df_summary.empty else 0
    neg_total = int(df_summary["Negative"].sum()) if not df_summary.empty else 0

    data = {
        "Total Mentions": f"{total_mentions:,}",
        "Positive Mentions (%)": f"{(pos_total / total_mentions * 100):.2f}%" if total_mentions else "0.0%",
        "Neutral Mentions (%)": f"{(neu_total / total_mentions * 100):.2f}%" if total_mentions else "0.0%",
        "Negative Mentions (%)": f"{(neg_total / total_mentions * 100):.2f}%" if total_mentions else "0.0%",
    }

    if df_detail is not None and "NPS_Score" in df_detail.columns:
        nps_vals = df_detail["NPS_Score"].dropna()
        nps_vals = nps_vals[nps_vals >= 0]
        nps_vals = normalize_nps_scores(nps_vals)  # Normalize scale to 0-10

        total = len(nps_vals)
        promoters = nps_vals[(nps_vals >= 9) & (nps_vals <= 10)].count()
        detractors = nps_vals[(nps_vals >= 0) & (nps_vals <= 6)].count()
        nps_score = (promoters / total * 100) - (detractors / total * 100) if total else 0

        data.update({
            "Average NPS Score": f"{nps_vals.mean():.2f}" if total else "N/A",
            "NPS Score (%)": f"{nps_score:.1f}",
            "Promoters": f"{promoters}",
            "Detractors": f"{detractors}",
        })

    return pd.DataFrame([{"KPI": k, "Value": v} for k, v in data.items()])

# -------------------------
# Charts
# -------------------------
def create_aspect_bar_chart(df_summary):
    import matplotlib.pyplot as plt
    from io import BytesIO
    import numpy as np

    df = df_summary.copy()
    df_sorted = df.sort_values(by='Negative', ascending=False).head(10)

    aspects = df_sorted['Aspect'].apply(lambda x: x.title())
    positive = df_sorted['Positive']
    neutral = df_sorted['Neutral']
    negative = df_sorted['Negative']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(aspects))
    width = 0.25

    bars_pos = ax.bar(x - width, positive, width, label='Positive')
    bars_neu = ax.bar(x, neutral, width, label='Neutral')
    bars_neg = ax.bar(x + width, negative, width, label='Negative')

    ax.set_xlabel('Aspect')
    ax.set_ylabel('Mentions')
    ax.set_title('Most Discussed Aspects by Sentiment', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, rotation=30, ha='right', fontsize=11)
    ax.legend()

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color='black'
                )

    for bars in [bars_pos, bars_neu, bars_neg]:
        add_labels(bars)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def create_negative_aspect_bar_chart(df_summary, top_n=10):
    top_neg = df_summary.sort_values(by='Negative', ascending=False).head(top_n).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        top_neg['Aspect'].apply(lambda x: x.title()),
        top_neg['Negative']
    )
    ax.set_title(f"Most Discussed Negative Aspects by Sentiment", fontsize=14, fontweight='bold')
    ax.set_ylabel("Negative Mentions")
    plt.xticks(rotation=30, ha='right')

    max_height = max(top_neg['Negative'].max(), 1) if not top_neg.empty else 1
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='#333')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------------
# PDF Report class
# -------------------------
class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.first_page = True
        try:
            base = os.path.dirname(__file__)
            fonts_path = os.path.join(base, "dejavu-fonts-ttf-2.37", "ttf")
            if not os.path.exists(fonts_path):
                fonts_path = os.path.join(base, "fonts")
            if not os.path.exists(fonts_path):
                fonts_path = base
            self.add_font('DejaVu', '', os.path.join(fonts_path, 'DejaVuSans.ttf'), uni=True)
            self.add_font('DejaVu', 'B', os.path.join(fonts_path, 'DejaVuSans-Bold.ttf'), uni=True)
            self.add_font('DejaVu', 'I', os.path.join(fonts_path, 'DejaVuSans-Oblique.ttf'), uni=True)
            self.add_font('DejaVu', 'BI', os.path.join(fonts_path, 'DejaVuSans-BoldOblique.ttf'), uni=True)
            self.font_family = 'DejaVu'
        except Exception:
            self.font_family = 'Arial'

    def header(self):
        if self.page_no() == 1:
            self.set_font(self.font_family, 'B', 16)
            self.set_text_color(45, 65, 155)
            self.cell(0, 10, "CUSTOMER REVIEW SENTIMENT ANALYSIS", 0, 1, 'C')
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_family, 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

    def section(self, title, size=14, after_space=1):
        self.set_font(self.font_family, 'B', size)
        self.set_text_color(15, 45, 90)
        self.cell(0, self.font_size_pt + 4, clean_text_for_pdf(title.upper()), 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(after_space)

    def subheading(self, text, size=12, after_space=1):
        self.set_font(self.font_family, 'B', size)
        self.set_text_color(40, 40, 120)
        self.cell(0, self.font_size_pt + 3, clean_text_for_pdf(text), 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(after_space)

    def add_paragraph(self, text, size=11, line_height=5, after_space=1):
        self.set_font(self.font_family, '', size)
        cleaned = clean_text_for_pdf(text)
        if not cleaned:
            cleaned = "[Empty or invalid text]"
        self.multi_cell(0, line_height, cleaned)
        self.ln(after_space)

    def add_table(self, df, title=None, fontsize=9, col_title_fontsize=9, truncate_columns=None, col_widths=None):
        if title:
            self.set_font(self.font_family, 'B', fontsize + 2)
            self.cell(0, fontsize + 6, clean_text_for_pdf(title), 0, 1)
        num_cols = len(df.columns)
        available_width = self.epw - (num_cols + 1) * self.c_margin
        if col_widths is None:
            col_widths = [available_width / num_cols] * num_cols
        else:
            assert len(col_widths) == num_cols, "col_widths must match number of columns"
        row_height = 6

        self.set_font(self.font_family, 'B', col_title_fontsize)
        for i, col in enumerate(df.columns):
            colname = col
            if truncate_columns and col in truncate_columns:
                colname = (str(col)[:20] + "...") if len(str(col)) > 20 else str(col)
            self.cell(col_widths[i], row_height, clean_text_for_pdf(colname), border=1, align='C')
        self.ln(row_height)

        self.set_font(self.font_family, '', fontsize)
        for _, row in df.iterrows():
            if self.get_y() + row_height > self.h - self.b_margin:
                self.add_page()
                self.set_font(self.font_family, 'B', col_title_fontsize)
                for i, col in enumerate(df.columns):
                    colname = col
                    if truncate_columns and col in truncate_columns:
                        colname = (str(col)[:20] + "...") if len(str(col)) > 20 else str(col)
                    self.cell(col_widths[i], row_height, clean_text_for_pdf(colname), border=1, align='C')
                self.ln(row_height)
                self.set_font(self.font_family, '', fontsize)

            for i, col in enumerate(df.columns):
                val = row[col]
                val_str = str(val)
                if truncate_columns and col in truncate_columns and len(val_str) > 40:
                    val_str = val_str[:40] + "..."
                self.cell(col_widths[i], row_height, clean_text_for_pdf(val_str), border=1)
            self.ln(row_height)

    def add_image(self, buf, width=None, caption=None, caption_center=True):
        if caption:
            self.set_font(self.font_family, 'I', 10)
            self.cell(0, 8, clean_text_for_pdf(caption), 0, 1, 'C' if caption_center else 'L')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(buf.getvalue())
            tmp.flush()
            image_path = tmp.name
        self.image(image_path, w=width if width else self.epw)
        os.remove(image_path)
        self.ln()

# -------------------------
# Recommendation builder
# -------------------------
KEYWORD_ACTIONS = {
    'late': [
        'Audit last-mile delivery processes to reduce delays',
        'Provide proactive ETA updates and allow rescheduling options',
    ],
    'delay': [
        'Investigate upstream supply chain bottlenecks',
        'Offer compensation or discounts for significantly delayed orders',
    ],
    'broken': [
        'Improve packaging durability and add fragile item alerts',
        'Introduce pre-dispatch inspection protocols',
    ],
    'defect': [
        'Strengthen vendor quality checks and certifications',
        'Add random sample inspections before dispatch',
    ],
    'packag': [
        'Upgrade packaging material and cushioning for sensitive items',
        'Add tamper-evident seals and QC checks for packaging integrity',
    ],
    'refund': [
        'Simplify refund workflows and automate status tracking for customers',
        'Display clear refund timelines at checkout',
    ],
    'support': [
        'Train support teams for faster ticket resolution',
        'Expand chat/self-help options for instant resolutions',
    ],
    'charge': [
        'Show detailed pricing breakdown upfront to avoid billing disputes',
        'Enable real-time billing alerts for transparency',
    ],
    'communication': [
        'Automate real-time communication during key order stages',
        'Send proactive alerts on delays or changes in schedule',
    ]
}

def build_recommendations_for_aspect(aspect, neg_reviews, top_n=5):
    aspect_key = aspect.lower().strip() if isinstance(aspect, str) else str(aspect)
    if not neg_reviews or all(not str(r).strip() for r in neg_reviews):
        return [f"No actionable recommendations for '{aspect_key.title()}' at this time."]

    reviews_blob = " ".join([str(r).lower() for r in neg_reviews if r])
    reviews_blob = re.sub(rf"\b{re.escape(aspect_key)}\b", "", reviews_blob)

    # Use the global RAKE instance (resets internal state on extract)
    RAKE.extract_keywords_from_text(reviews_blob)
    ranked_phrases = RAKE.get_ranked_phrases()[:30]

    if not ranked_phrases:
        doc = nlp(reviews_blob)
        words = [t.lemma_ for t in doc if getattr(t, "pos_", None) in {"NOUN", "ADJ"} and not getattr(t, "is_stop", False)]
        ranked_phrases = [w for w, _ in Counter(words).most_common(top_n)]

    top_issues = []
    for ph in ranked_phrases:
        ph_clean = _RE_NONWORD.sub('', ph.lower()).strip()
        if ph_clean and ph_clean not in top_issues:
            top_issues.append(ph_clean)
        if len(top_issues) >= top_n:
            break

    recs = []
    aspect_nice = aspect_key.title()

    mapped_actions = []
    for issue in top_issues:
        for k, actions in KEYWORD_ACTIONS.items():
            if k in issue or issue in k:
                for act in actions:
                    if act not in mapped_actions:
                        mapped_actions.append(act)

    if top_issues:
        issues_str = "', '".join(top_issues[:3])
        recs.append(f"Customers frequently mention '{issues_str}' related to {aspect_nice}. Investigate root causes immediately.")
        recs.append(f"Prioritize resolving '{top_issues[0]}' to improve {aspect_nice} experience.")

    recs.extend(mapped_actions)

    while len(recs) < top_n:
        if len(recs) == 2:
            recs.append(f"Review all touchpoints impacting {aspect_nice} and set clear SLAs.")
        elif len(recs) == 3:
            recs.append(f"Implement a feedback loop to capture {aspect_nice}-related complaints proactively.")
        elif len(recs) == 4:
            recs.append(f"Track {aspect_nice} KPIs weekly and validate improvements with sentiment analysis.")
        else:
            break

    final = []
    for r in recs:
        if r not in final:
            final.append(r)
        if len(final) >= top_n:
            break

    return final

# -------------------------
# Top negative reviews by aspect (optimized)
# -------------------------
def extract_top_negative_reviews_by_aspect(detail_df, aspects, max_reviews=5):
    def normalize(text):
        if not isinstance(text, str):
            text = str(text)
        text = _RE_PAGE.sub('', text)
        text = re.sub(r'^\d+\.\s*', '', text)
        text = _RE_MULTISPACE.sub(' ', text)
        return text.strip().lower()

    reviews = {}
    if detail_df.empty:
        for a in aspects:
            reviews[a] = []
        return reviews

    unique_reviews = detail_df[['Review_ID', 'Review']].drop_duplicates()
    overall_scores = {row['Review_ID']: vader_score(str(row['Review'])) for _, row in unique_reviews.iterrows()}

    for aspect in aspects:
        aspect_key = aspect.lower().strip()
        filtered = detail_df[
            (detail_df['Aspect'].str.lower() == aspect_key) |
            (detail_df['Aspect'].str.contains(aspect_key, case=False, na=False))
        ].copy()

        if filtered.empty:
            filtered = detail_df[
                detail_df['Aspect_Context'].str.contains(re.escape(aspect_key), case=False, na=False)
            ].copy()

        if not filtered.empty:
            filtered['Overall_Score'] = filtered['Review_ID'].map(overall_scores)
            filtered['Context_Score'] = filtered['Aspect_Context'].astype(str).apply(vader_score)
            filtered['Is_Overall_Negative'] = filtered['Overall_Score'] <= -0.3
            filtered = filtered.sort_values(['Is_Overall_Negative', 'Context_Score'], ascending=[False, True])

            deduped_texts, seen_texts = [], set()
            for _, row in filtered.iterrows():
                norm = normalize(row['Review'])
                if not norm or norm in seen_texts:
                    continue
                seen_texts.add(norm)
                cleaned_review = clean_text_for_pdf(row['Review'])
                if cleaned_review:
                    deduped_texts.append(cleaned_review)
                if len(deduped_texts) >= max_reviews:
                    break
            reviews[aspect] = deduped_texts
        else:
            reviews[aspect] = []

    return reviews

# -------------------------
# PDF builder
# -------------------------
def create_pdf_report(
    detail_df,
    summary_df,
    top_negative_suppliers=None,
    recommendations=None,
    processing_stats=None
):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    pdf.section("EXECUTIVE SUMMARY & KEY INSIGHTS")
    if processing_stats:
        def safe_format(value):
            if isinstance(value, (int, float)):
                return f"{value:,}"
            return str(value)

        uploaded = safe_format(processing_stats.get('uploaded', 'N/A'))
        filtered = safe_format(processing_stats.get('filtered_out', 'N/A'))
        analysed = safe_format(processing_stats.get('analysed', 'N/A'))
        unique_aspects = safe_format(processing_stats.get('unique_aspects', 'N/A'))

        mentions = processing_stats.get('aspect_mentions') if processing_stats else None
        if mentions is None:
            mentions = len(detail_df) if (detail_df is not None and not detail_df.empty) else 0
        mentions_str = safe_format(mentions)

        pdf.set_font(pdf.font_family, '', 11)
        pdf.multi_cell(
            0, 6,
            f"Data Processing Summary:\n"
            f"- Uploaded Reviews: {uploaded}\n"
            f"- Filtered Out: {filtered} (invalid or blank)\n"
            f"- Analysed: {analysed} valid reviews\n"
            f"- Unique Aspects Identified: {unique_aspects}\n"
            f"- Total Aspect Mentions: {mentions_str}\n"
        )
        pdf.ln(3)

    if summary_df.empty:
        pdf.add_paragraph("No summary data available for insights.")
    else:
        top = summary_df.iloc[0]
        pdf.add_paragraph(
            f"Top mentioned aspect: {top['Aspect'].title()} "
            f"({top['Total Mentions']} mentions)."
        )

    pdf.section("KEY METRICS & KPI OVERVIEW")
    kpi_df = benchmark_kpis(summary_df, detail_df)
    pdf.add_table(kpi_df, title="Sentiment & NPS Score KPIs")

    pdf.section("SENTIMENT DISTRIBUTION BY TOP ASPECTS")
    sorted_summary = summary_df.copy()
    if 'Total Mentions' not in sorted_summary.columns:
        sorted_summary['Total'] = (
            sorted_summary.get('Positive', 0) +
            sorted_summary.get('Neutral', 0) +
            sorted_summary.get('Negative', 0)
        )
    else:
        sorted_summary['Total'] = sorted_summary['Total Mentions']
    sorted_summary = sorted_summary.sort_values(by='Total', ascending=False)

    pdf.add_image(create_aspect_bar_chart(sorted_summary))
    pdf.add_image(create_negative_aspect_bar_chart(sorted_summary, top_n=10))

    pdf.section("RECENT NEGATIVE REVIEWS BY ASPECT")
    top_aspects = sorted_summary['Aspect'].head(10).tolist()
    neg_reviews = extract_top_negative_reviews_by_aspect(detail_df, top_aspects, max_reviews=5)

    for asp in top_aspects:
        pdf.subheading(f"{asp.title()} - Negative Reviews")
        reviews = neg_reviews.get(asp, [])
        if not reviews:
            pdf.add_paragraph("No negative reviews found for this aspect.", size=9)
            continue
        for i, rev in enumerate(reviews, 1):
            pdf.add_paragraph(f"{i}. {safe_quote(rev)}", size=9)

    pdf.section("ACTIONABLE RECOMMENDATIONS")
    for asp in top_aspects:
        pdf.subheading(f"{asp.title()} - Recommendations")
        if recommendations and asp in recommendations:
            recs = recommendations.get(asp, [])
        else:
            recs = build_recommendations_for_aspect(asp, neg_reviews.get(asp, []))

        if recs:
            for i, rec in enumerate(recs[:5], 1):
                pdf.add_paragraph(f"{i}. {rec}", size=9)
        else:
            pdf.add_paragraph("No actionable recommendations available.", size=9)

    output = pdf.output(dest='S')
    if isinstance(output, str):
        output = output.encode()
    elif isinstance(output, bytearray):
        output = bytes(output)
    return output

# Globals for chat / UI
global_detail_df = pd.DataFrame()
global_summary_df = pd.DataFrame()
global_top_neg_reviews = {}
global_nps_col = None

def run_analysis(csv_file, review_column=None, nps_column=None, aspects=None, progress=gr.Progress()):
    global global_detail_df, global_summary_df, global_top_neg_reviews, global_nps_col

    if csv_file is None:
        return "Upload CSV file.", None, None, None

    try:
        df = safe_read_csv(csv_file)
        if nps_column and nps_column in df.columns:
            df[nps_column] = pd.to_numeric(df[nps_column], errors='coerce')

        uploaded_count = len(df)
        review_col = review_column if review_column and review_column in df.columns else auto_detect_review_column(df)
        nps_col = nps_column if nps_column and nps_column in df.columns else auto_detect_nps_column(df)

        lang = detect_language_of_reviews(df, review_col)
        if lang != 'en':
            return f"Detected language: {lang}. Only English is supported.", None, None, None

        user_aspects = None
        if aspects and str(aspects).strip():
            user_aspects = [a.strip() for a in str(aspects).split(",") if a.strip()]

        detail_df = analyze_review_structured(df, review_col, nps_col, user_aspects=user_aspects)
        summary_df = generate_sentiment_summary(detail_df)

        global_detail_df = detail_df
        global_summary_df = summary_df
        top_aspects_list = summary_df['Aspect'].head(10).tolist() if not summary_df.empty else []
        global_top_neg_reviews = extract_top_negative_reviews_by_aspect(detail_df, top_aspects_list, max_reviews=5)

        summary_md = summary_df.head(10).to_markdown(index=False)
        detail_md = detail_df.head(20).to_markdown(index=False)

        processing_stats = dict(
            uploaded=uploaded_count,
            filtered_out=uploaded_count - len(detail_df['Review_ID'].unique()) if not detail_df.empty else uploaded_count,
            analysed=len(detail_df['Review_ID'].unique()) if not detail_df.empty else 0,
            unique_aspects=summary_df['Aspect'].nunique() if not summary_df.empty else 0,
            aspect_mentions=len(detail_df) if not detail_df.empty else 0,
        )
        pdf_bytes = create_pdf_report(detail_df, summary_df, processing_stats=processing_stats)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_pdf_path = tmp.name

        msg = f"✅ Analysis complete. {uploaded_count} reviews uploaded, {processing_stats['analysed']} analyzed."
        return msg, summary_md, detail_md, temp_pdf_path

    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}", None, None, None

# -------------------------
# Simple chatbot helper
# -------------------------
def chatbot_query(message, history):
    if global_summary_df.empty or not global_top_neg_reviews:
        return "", history + [[message, "Please upload and analyze data first."]]

    msg_lower = message.lower()
    response = ""

    if any(word in msg_lower for word in ["thank", "bye"]):
        response = "You're welcome! Ask more questions anytime."

    elif any(word in msg_lower for word in ["main negative", "top problems"]):
        neg_aspects = global_summary_df[global_summary_df['Dominant Sentiment'] == 'Negative']
        if neg_aspects.empty:
            response = "No dominant negative aspects found."
        else:
            response = "Main negative aspects:\n"
            for _, row in neg_aspects.head(3).iterrows():
                avg_nps = row['Avg NPS'] if isinstance(row['Avg NPS'], (int, float)) else "N/A"
                response += f"- **{row['Aspect'].title()}** ({row['Total Mentions']} mentions, {row['Negative (%)']}% negative, Avg NPS: {avg_nps})\n"

    elif re.search(r"(sentiment|nps) for (.+)", msg_lower):
        match = re.search(r"(sentiment|nps) for (.+)", msg_lower)
        aspect_name = match.group(2).strip().lower()
        row = global_summary_df[global_summary_df['Aspect'].str.lower() == aspect_name]
        if row.empty:
            row = global_summary_df[global_summary_df['Aspect'].str.contains(aspect_name, case=False)]
        if row.empty:
            response = f"No data for aspect '{aspect_name}'."
        else:
            row = row.iloc[0]
            avg_nps = row['Avg NPS'] if isinstance(row['Avg NPS'], (int, float)) else "N/A"
            response = (
                f"For **{row['Aspect'].title()}**:\n"
                f"- Positive: {row['Positive (%)']}%\n"
                f"- Neutral: {row['Neutral (%)']}%\n"
                f"- Negative: {row['Negative (%)']}%\n"
                f"- Dominant: **{row['Dominant Sentiment']}**\n"
                f"- Avg NPS: **{avg_nps}**"
            )

    elif re.search(r"(recommendations?|suggestions?) for (.+)", msg_lower):
        match = re.search(r"(recommendations?|suggestions?) for (.+)", msg_lower)
        aspect_name = match.group(2).strip().lower()
        rows = global_summary_df[global_summary_df['Aspect'].str.lower() == aspect_name]
        if rows.empty:
            rows = global_summary_df[global_summary_df['Aspect'].str.contains(aspect_name, case=False)]
        if rows.empty:
            response = f"No data for '{aspect_name}'."
        else:
            actual = rows.iloc[0]['Aspect']
            reviews = global_top_neg_reviews.get(actual, [])
            if not reviews and not global_detail_df.empty:
                matches = global_detail_df[global_detail_df['Aspect_Context'].str.contains(re.escape(actual), case=False, na=False)]
                reviews = matches['Review'].dropna().astype(str).head(5).tolist()

            if not reviews:
                response = f"No negative reviews for '{actual}' to generate recommendations."
            else:
                recs = build_recommendations_for_aspect(actual, reviews)
                response = f"Recommendations for '{actual.title()}':\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(recs, 1))

    else:
        response = (
            "Try asking:\n"
            "- What are the main negative aspects?\n"
            "- Sentiment for Food\n"
            "- Recommendations for Service"
        )

    history.append([message, response])
    return "", history

# ==== Gradio UI ====
with gr.Blocks() as demo:
    gr.Markdown("# Customer Reviews and NPS Sentiment Analysis")
    gr.Markdown("Upload your CSV/Excel file to analyze customer feedback and generate insights.")
    with gr.Row():
        with gr.Column(scale=1):
            csv_input = gr.File(label="Upload File (.csv or .xlsx)")
            review_input = gr.Textbox(label="Review Column (Optional)", placeholder="Leave blank to auto-detect")
            nps_input = gr.Textbox(label="NPS Column (Optional)", placeholder="e.g., nps_score")
            aspect_input = gr.Textbox(
                label="Aspects to Analyze (comma-separated)",
                placeholder="e.g. delivery, service, quality (leave blank for auto-detect)",
            )
            analyze_btn = gr.Button("🚀 Analyze Reviews")
            status_text = gr.Markdown("📊 Upload a file to begin.")
            pdf_output_file = gr.File(label="📥 Download Full PDF Report", visible=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Ask Insights", height=400)
            chatbox = gr.Textbox(label="Ask a question:", placeholder="E.g., 'What are the main negative aspects?'")
            clear_btn = gr.Button("🗑️ Clear Chat")

    gr.Markdown("---")
    gr.Markdown("## Results")
    with gr.Tabs():
        with gr.TabItem("Summary (Top 10)"):
            summary_output = gr.Markdown("Summary will appear here.")
        with gr.TabItem("Raw Mentions (First 20)"):
            detail_output = gr.Markdown("Detailed data will appear here.")

    analyze_btn.click(
        fn=run_analysis,
        inputs=[csv_input, review_input, nps_input, aspect_input],
        outputs=[status_text, summary_output, detail_output, pdf_output_file]
    )
    chatbox.submit(fn=chatbot_query, inputs=[chatbox, chatbot], outputs=[chatbox, chatbot])
    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
