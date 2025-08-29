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
import matplotlib.pyplot as plt
from fpdf import FPDF
import gradio as gr
import streamlit as st

from rake_nltk import Rake
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

#####################
# Dependency checks #
#####################

#####################
# Dependency checks #
#####################

# Download NLTK stopwords and VADER if missing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Ensure punkt tokenizer is available (for RAKE)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Ensure punkt_tab (newer NLTK >=3.9) is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Ensure SpaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

sia = SentimentIntensityAnalyzer()

ARTICLES = {'the', 'a', 'an'}
DEMONSTRATORS = {'this', 'that', 'these', 'those'}

def clean_phrase(phrase: str) -> str:
    phrase = re.sub(r'[^\w\s]', '', phrase.lower()).strip()
    tokens = phrase.split()
    while tokens and tokens[0] in ARTICLES.union(DEMONSTRATORS):
        tokens.pop(0)
    doc = nlp(' '.join(tokens))
    lemmas = [token.lemma_ for token in doc]
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
    q = re.sub(r'\s+', ' ', q)
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

    # Fallback: look for a numeric column with reasonable integer-like values
    numcols = df.select_dtypes(include=[np.number]).columns
    for col in numcols:
        vals = df[col].dropna()
        if not vals.empty:
            uniq = vals.unique()
            # Heuristic: NPS columns usually have a small integer range (e.g., 0–10, 1–5, 1–7)
            if len(uniq) >= 3 and np.all(np.equal(np.mod(uniq, 1), 0)):
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


def extract_dynamic_aspects_user(review_text, user_aspects):
    user_aspects = _normalize_user_aspects(user_aspects)
    review_text = html.unescape(review_text)
    doc = nlp(review_text)
    extracted = defaultdict(list)
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        sent_score = sia.polarity_scores(sent_text)["compound"]
        if sent_score >= 0.3:
            label = "Positive"
        elif sent_score <= -0.3:
            label = "Negative"
        else:
            label = "Neutral"
        sent_text_lc = sent_text.lower()
        for aspect in user_aspects:
            if re.search(r'\b' + re.escape(aspect) + r'\b', sent_text_lc):
                extracted[aspect.title()].append((sent_text, label, aspect))
    data = []
    for aspect, mentions in extracted.items():
        agg_sent = aggregate_aspect_sentiments(mentions)
        context = "; ".join(sorted(set(m[0] for m in mentions)))
        quotes = [m[2] for m in mentions]
        data.append({
            "Review_ID": None,
            "Review": None,
            "Aspect": aspect,
            "Aspect_Sentiment": agg_sent,
            "Aspect_Context": context,
            "Quotes": quotes,
        })
    return data


def analyze_review_structured(
    df,
    review_col,
    nps_col=None,
    user_aspects=None,
    process_times=None
):
    if review_col not in df.columns:
        raise ValueError(f"Review column '{review_col}' not found.")
    if nps_col and nps_col not in df.columns:
        raise ValueError(f"NPS column '{nps_col}' not found.")
    if df[review_col].isnull().all():
        raise ValueError(f"All entries in review column '{review_col}' are empty.")
    if not user_aspects:
        raise ValueError("Please provide at least one user aspect.")

    t0 = time.time()
    user_aspects = _normalize_user_aspects(user_aspects)
    records = []
    total = len(df)

    placeholder = st.empty()

    for idx, (i, row) in enumerate(df.iterrows(), 1):
        text = str(row[review_col]).strip()
        if not text:
            continue

        nps_val = row.get(nps_col) if nps_col else None
        if isinstance(nps_val, str):
            nps_val = nps_val.strip()
            if nps_val.lower() in ('none', '', 'nan'):
                nps_val = None
        try:
            nps_val = float(nps_val)
        except (ValueError, TypeError):
            nps_val = None

        aspects = extract_dynamic_aspects_user(text, user_aspects)
        for asp in aspects:
            asp["Review_ID"] = i + 1
            asp["Review"] = text
            asp["NPS_Score"] = nps_val
            for field in ['supplier_name', 'product_fullname']:
                if field in df.columns:
                    asp[field] = row[field]
            records.append(asp)

        placeholder.text(f"Processed {idx:,}/{total:,} reviews...")

    if process_times is not None:
        process_times["aspect_sentiment_extraction"] = round(time.time() - t0, 2)

    return pd.DataFrame(records)

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
        valid_nps = group["NPS_Score"].dropna()
        if not valid_nps.empty:
            rec["Avg_NPS"] = round(valid_nps.mean(), 2)

            min_score, max_score = valid_nps.min(), valid_nps.max()
            range_size = max_score - min_score

            rec["Promoters"] = valid_nps[valid_nps >= max_score - range_size * 0.2].count()
            rec["Passives"] = valid_nps[(valid_nps >= min_score + range_size * 0.4) &
                                        (valid_nps < max_score - range_size * 0.2)].count()
            rec["Detractors"] = valid_nps[valid_nps < min_score + range_size * 0.4].count()
        else:
            rec["Avg_NPS"] = "N/A"
            rec["Promoters"] = rec["Passives"] = rec["Detractors"] = 0
        rows.append(rec)
    return pd.DataFrame(rows)


def generate_sentiment_summary(df):
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
            valid_nps = group["NPS_Score"].dropna()
            avg_nps = valid_nps.mean() if not valid_nps.empty else np.nan

            min_score, max_score = valid_nps.min(), valid_nps.max()
            range_size = max_score - min_score

            promoters = valid_nps[valid_nps >= max_score - range_size * 0.2].count()
            passives = valid_nps[(valid_nps >= min_score + range_size * 0.4) &
                                 (valid_nps < max_score - range_size * 0.2)].count()
            detractors = valid_nps[valid_nps < min_score + range_size * 0.4].count()

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
    total_mentions = df_summary["Total Mentions"].sum()
    pos_total = df_summary["Positive"].sum()
    neu_total = df_summary["Neutral"].sum()
    neg_total = df_summary["Negative"].sum()

    data = {
        "Total Mentions": f"{total_mentions:,}",
        "Positive Mentions (%)": f"{pos_total / total_mentions * 100:.2f}%" if total_mentions else "0.0%",
        "Neutral Mentions (%)": f"{neu_total / total_mentions * 100:.2f}%" if total_mentions else "0.0%",
        "Negative Mentions (%)": f"{neg_total / total_mentions * 100:.2f}%" if total_mentions else "0.0%",
    }

    if df_detail is not None and "NPS_Score" in df_detail.columns:
        nps_vals = df_detail["NPS_Score"].dropna()
        if not nps_vals.empty:
            min_score, max_score = nps_vals.min(), nps_vals.max()
            range_size = max_score - min_score

            total = len(nps_vals)
            promoters = nps_vals[nps_vals >= max_score - range_size * 0.2].count()
            detractors = nps_vals[nps_vals < min_score + range_size * 0.4].count()
            nps_score = (promoters / total * 100) - (detractors / total * 100) if total else 0

            data.update({
                "Average NPS Score": f"{nps_vals.mean():.2f}" if total else "N/A",
                "NPS Score (%)": f"{nps_score:.1f}",
                "Promoters": f"{promoters}",
                "Detractors": f"{detractors}",
            })

    return pd.DataFrame([{"KPI": k, "Value": v} for k, v in data.items()])

def create_aspect_bar_chart(df_summary):
    import matplotlib.pyplot as plt
    from io import BytesIO
    import numpy as np

    # Sort dataframe by Negative counts in descending order
    df = df_summary.copy()
    df_sorted = df.sort_values(by='Negative', ascending=False).head(10)

    aspects = df_sorted['Aspect']
    positive = df_sorted['Positive']
    neutral = df_sorted['Neutral']
    negative = df_sorted['Negative']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(aspects))
    width = 0.25

    bars_pos = ax.bar(x - width, positive, width, label='Positive', color='#53a944')
    bars_neu = ax.bar(x, neutral, width, label='Neutral', color='#f7d948')
    bars_neg = ax.bar(x + width, negative, width, label='Negative', color='#ed3939')

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
    # Sort descending by Negative mentions before taking top_n
    top_neg = df_summary.sort_values(by='Negative', ascending=False).head(top_n).copy()
    fig, ax = plt.subplots(figsize=(8, 6))  # increased height from 4 to 6
    bars = ax.bar(
        top_neg['Aspect'],
        top_neg['Negative'],
        color='#ed2939'
    )
    ax.set_title(f"Most Discussed Negative Aspects by Sentiment", fontsize=14, fontweight='bold')
    ax.set_ylabel("Negative Mentions")
    plt.xticks(rotation=30, ha='right')

    max_height = max(top_neg['Negative'].max(), 1)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        color='#333')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

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
        
def build_recommendations_for_aspect(aspect, neg_reviews, top_n=None):
    """
    Dynamically build actionable recommendations from negative reviews.
    - Extracts all meaningful issues (not capped at 5)
    - Uses RAKE + SpaCy for keyphrase extraction
    - Optional top_n to limit results
    """
    aspect_key = aspect.lower().strip() if isinstance(aspect, str) else str(aspect)
    if not neg_reviews or all(not str(r).strip() for r in neg_reviews):
        return [f"No actionable recommendations for '{aspect_key.title()}' at this time."]

    # 1. Preprocess: join reviews
    reviews_blob = " ".join([str(r).lower() for r in neg_reviews if r])
    reviews_blob = re.sub(rf"\b{re.escape(aspect_key)}\b", "", reviews_blob)

    # 2. Extract key phrases with RAKE
    rake = Rake(min_length=1, max_length=3)
    rake.extract_keywords_from_text(reviews_blob)
    ranked_phrases = rake.get_ranked_phrases()[:50]  # expanded window

    # 3. Fallback to SpaCy if RAKE fails
    if not ranked_phrases:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(reviews_blob)
        words = [t.lemma_ for t in doc if t.pos_ in {"NOUN", "ADJ"} and not t.is_stop]
        ranked_phrases = [w for w, _ in Counter(words).most_common(50)]

    # 4. Deduplicate issues
    top_issues = []
    for ph in ranked_phrases:
        ph_clean = re.sub(r"[^\w\s]", '', ph.lower()).strip()
        if ph_clean and ph_clean not in top_issues:
            top_issues.append(ph_clean)

    if top_n:
        top_issues = top_issues[:top_n]

    # 5. Build recommendations
    recs = []
    aspect_nice = aspect_key.title()
    if top_issues:
        issues_str = "', '".join(top_issues[:min(5, len(top_issues))])
        recs.append(f"Customers frequently mention '{issues_str}' related to {aspect_nice}. Investigate root causes immediately.")
        recs.append(f"Prioritize resolving '{top_issues[0]}' to improve {aspect_nice} experience.")

    # Generic fallback if no strong issues
    if not recs:
        recs.append(f"No strong issue patterns found for {aspect_nice}. Continue monitoring.")

    return recs if not top_n else recs[:top_n]


def extract_top_negative_reviews_by_aspect(detail_df, aspects, max_reviews=None):
    """
    Extract negative reviews dynamically per aspect.
    If max_reviews=None, returns all unique negative reviews.
    """
    def normalize(text):
        text = re.sub(r'page\s*\d+', '', text, flags=re.I)
        text = re.sub(r'^\d+\.\s*', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    reviews = {}
    if detail_df.empty:
        return {a: [] for a in aspects}

    unique_reviews = detail_df[['Review_ID', 'Review']].drop_duplicates()
    overall_scores = {
        row['Review_ID']: sia.polarity_scores(str(row['Review']))['compound']
        for _, row in unique_reviews.iterrows()
    }

    for aspect in aspects:
        aspect_key = aspect.lower().strip()
        filtered = detail_df[
            (detail_df['Aspect'].str.lower() == aspect_key) &
            (detail_df['Aspect_Sentiment'] == 'Negative')
        ].copy()

        if filtered.empty:
            reviews[aspect] = []
            continue

        filtered['Overall_Score'] = filtered['Review_ID'].map(overall_scores)
        filtered['Context_Score'] = filtered['Aspect_Context'].apply(
            lambda x: sia.polarity_scores(str(x))['compound']
        )
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
            if max_reviews and len(deduped_texts) >= max_reviews:
                break
        reviews[aspect] = deduped_texts

    return reviews


def create_pdf_report(detail_df, summary_df, top_negative_suppliers=None,
                      recommendations=None, processing_stats=None):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    # Executive summary
    pdf.section("EXECUTIVE SUMMARY & KEY INSIGHTS")
    if processing_stats:
        def safe_format(value):
            if isinstance(value, (int, float)):
                return f"{value:,}"
            return str(value)

        uploaded_str = safe_format(processing_stats.get('uploaded', 'N/A'))
        filtered_str = safe_format(processing_stats.get('filtered_out', 'N/A'))
        analysed_str = safe_format(processing_stats.get('analysed', 'N/A'))
        unique_str = safe_format(processing_stats.get('unique_aspects', 'N/A'))

        mentions = processing_stats.get('aspect_mentions',
                    len(detail_df) if detail_df is not None else 0)
        mentions_str = safe_format(mentions)

        pdf.set_font(pdf.font_family, '', 11)
        pdf.multi_cell(0, 6,
            f"Data processing summary:\n"
            f"- Uploaded: {uploaded_str} reviews\n"
            f"- Filtered out: {filtered_str} (blanks, undefined, N/A)\n"
            f"- Analysed: {analysed_str} valid reviews\n"
            f"- Unique aspects: {unique_str}\n"
            f"- Total mentions: {mentions_str}\n"
        )
        pdf.ln(3)

    if summary_df.empty:
        pdf.add_paragraph("No data to summarize.")
    else:
        top = summary_df.iloc[0]
        pdf.add_paragraph(
            f"Top mentioned aspect: {top['Aspect']} ({top['Total Mentions']} mentions)."
        )

    # KPI section
    pdf.section("KEY METRICS & KPI OVERVIEW")
    kpi_df = benchmark_kpis(summary_df, detail_df)
    pdf.add_table(kpi_df, title="Sentiment & NPS Score KPIs")

    # Charts
    pdf.section("SENTIMENT DISTRIBUTION BY TOP ASPECTS")
    sorted_summary = summary_df.copy()
    sorted_summary['Total'] = (sorted_summary['Positive'] +
                               sorted_summary['Neutral'] +
                               sorted_summary['Negative'])
    sorted_summary = sorted_summary.sort_values(by='Total', ascending=False)
    pdf.add_image(create_aspect_bar_chart(sorted_summary))
    pdf.add_image(create_negative_aspect_bar_chart(sorted_summary, top_n=10))

    # Negative reviews
    pdf.section("RECENT NEGATIVE REVIEWS BY ASPECT")
    top_aspects = sorted_summary['Aspect'].head(10).tolist()
    neg_reviews = extract_top_negative_reviews_by_aspect(detail_df, top_aspects, max_reviews=5)

    for asp in top_aspects:
        pdf.subheading(f"{asp} - Negative Reviews")
        reviews = neg_reviews.get(asp, [])
        if not reviews:
            pdf.add_paragraph("No negative reviews found for this aspect.", size=9)
            continue
        for i, rev in enumerate(reviews, 1):
            pdf.add_paragraph(f"{i}. {safe_quote(rev)}", size=9)

    # Recommendations
    pdf.section("ACTIONABLE RECOMMENDATIONS")
    if recommendations:
        for asp in top_aspects:
            pdf.subheading(f"{asp} - Recommendations")
            recs = recommendations.get(asp, [])
            if recs:
                for i, rec in enumerate(recs, 1):  # no cap
                    pdf.add_paragraph(f"{i}. {rec}", size=9)
            else:
                pdf.add_paragraph("No actionable recommendations for this aspect.", size=9)
    else:
        for asp in top_aspects:
            pdf.subheading(f"{asp} - Recommendations")
            recs = build_recommendations_for_aspect(asp, neg_reviews.get(asp, []), top_n=None)
            for i, rec in enumerate(recs, 1):  # no cap
                pdf.add_paragraph(f"{i}. {rec}", size=9)

    # Return PDF as bytes
    output = pdf.output(dest='S')
    if isinstance(output, str):
        output = output.encode()
    elif isinstance(output, bytearray):
        output = bytes(output)
    return output

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

        if aspects is None or not str(aspects).strip():
            return "Please enter at least one aspect (comma-separated, e.g. 'delivery,service,quality').", None, None, None
        user_aspects = [a.strip() for a in str(aspects).split(",") if a.strip()]

        detail_df = analyze_review_structured(df, review_col, nps_col, user_aspects=user_aspects)
        summary_df = generate_sentiment_summary(detail_df)

        global_detail_df = detail_df
        global_summary_df = summary_df
        global_top_neg_reviews = extract_top_negative_reviews_by_aspect(
            detail_df,
            summary_df['Aspect'].head(10).tolist(),
            max_reviews=5
        )

        summary_md = summary_df.head(10).to_markdown(index=False)
        detail_md = detail_df.head(20).to_markdown(index=False)

        # ✅ Updated keys to match create_pdf_report expectations
        processed_count = len(detail_df['Review_ID'].unique())
        processing_stats = dict(
            uploaded=uploaded_count,
            filtered_out=max(uploaded_count - processed_count, 0),
            analysed=processed_count,
            unique_aspects=summary_df['Aspect'].nunique(),
            aspect_mentions=len(detail_df),
        )

        pdf_bytes = create_pdf_report(detail_df, summary_df, processing_stats=processing_stats)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_pdf_path = tmp.name

        msg = f"✅ Analysis complete. {uploaded_count} reviews uploaded, {processed_count} analyzed."
        return msg, summary_md, detail_md, temp_pdf_path

    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}", None, None, None

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
                response += f"- **{row['Aspect']}** ({row['Total Mentions']} mentions, {row['Negative (%)']}% negative, Avg NPS: {avg_nps})\n"

    elif re.search(r"(sentiment|nps) for (.+)", msg_lower):
        match = re.search(r"(sentiment|nps) for (.+)", msg_lower)
        aspect_name = match.group(2).strip().title()
        row = global_summary_df[global_summary_df['Aspect'].str.lower() == aspect_name.lower()]
        if row.empty:
            row = global_summary_df[global_summary_df['Aspect'].str.contains(aspect_name, case=False)]
        if row.empty:
            response = f"No data for aspect '{aspect_name}'."
        else:
            row = row.iloc[0]
            avg_nps = row['Avg NPS'] if isinstance(row['Avg NPS'], (int, float)) else "N/A"
            response = (
                f"For **{row['Aspect']}**:\n"
                f"- Positive: {row['Positive (%)']}%\n"
                f"- Neutral: {row['Neutral (%)']}%\n"
                f"- Negative: {row['Negative (%)']}%\n"
                f"- Dominant: **{row['Dominant Sentiment']}**\n"
                f"- Avg NPS: **{avg_nps}**"
            )

    elif re.search(r"(recommendations?|suggestions?) for (.+)", msg_lower):
        match = re.search(r"(recommendations?|suggestions?) for (.+)", msg_lower)
        aspect_name = match.group(2).strip().title()
        rows = global_summary_df[global_summary_df['Aspect'].str.lower() == aspect_name.lower()]
        if rows.empty:
            rows = global_summary_df[global_summary_df['Aspect'].str.contains(aspect_name, case=False)]
        if rows.empty:
            response = f"No data for '{aspect_name}'."
        else:
            actual = rows.iloc[0]['Aspect']
            reviews = global_top_neg_reviews.get(actual, [])
            if not reviews:
                response = f"No negative reviews for '{actual}' to generate recommendations."
            else:
                recs = build_recommendations_for_aspect(actual, reviews)
                response = f"Recommendations for '{actual}':\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(recs, 1))

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
                placeholder="e.g. delivery, service, quality",
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
