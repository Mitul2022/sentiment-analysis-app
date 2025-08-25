
import streamlit as st
st.set_page_config(page_title="SentimentIQ", page_icon="📊", layout="wide")

from auth import auth_controller

if auth_controller():

    import pandas as pd
    import time
    import traceback
    import re
    import matplotlib.pyplot as plt
    from io import BytesIO
    from collections import Counter
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    # Try import wordcloud, or skip feature if not available
    try:
        from wordcloud import WordCloud, STOPWORDS
        _wordcloud_available = True
    except ImportError:
        _wordcloud_available = False

    from sentiment_engine import (
        analyze_review_structured,
        generate_sentiment_summary,
        create_pdf_report,
        extract_top_negative_reviews_by_aspect,
        build_recommendations_for_aspect,
        clean_text_for_pdf,
        auto_detect_review_column,
        auto_detect_nps_column,
        detect_language_of_reviews,
        safe_read_csv,
        limit_large_df
    )

    TOP_N_ASPECTS = 10  # For summary charts and negative mentions
    
    # Custom CSS for title, tagline, buttons, and other styling
    st.markdown("""
    <style>
    .big-title { font-size: 2.7rem; font-weight: 900; letter-spacing: .035rem; color: #005282; }
    .subtitle { font-size: 1.2rem; color:#0073b8; font-weight:600; }
    .footer { color:#29506d; font-size:0.9rem; padding:2em 0; text-align:center; }
    .stButton>button, .stDownloadButton>button {
        color: white !important;
        background: linear-gradient(90deg,#0073b8,#009ece);
        border: 0px;
        border-radius: 7px;
        font-weight: 600;
        font-size: 1.05em;
        margin-bottom: 1.5em;
        transition: box-shadow .2s;
        box-shadow: 0 3px 10px #0073b880;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        box-shadow: 0 5px 15px #0073b8aa;
    }
    .stButton>button:active, .stDownloadButton>button:active {
        box-shadow: none;
    }
    .stButton>button[disabled], .stDownloadButton>button[disabled] {
        background: #a0a0a0 !important;
    }
    /* Title and tagline styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: #003366;
        text-align: center;
        margin-bottom: 0.2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-tagline {
        font-size: 1.5rem;
        font-weight: 500;
        color: #005b99;
        font-style: italic;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem; border-radius: 12px; background: linear-gradient(90deg, #d6e9ff 15%, #b5dff 85%);
                    border: 1px solid #82b0ff; margin-bottom: 1rem;">
            <h3 style="color: #004080; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; margin-bottom: 0.5rem;">
                Quick Start Guide
            </h3>
            <ul style="font-size: 1rem; color: #003366; padding-left: 1.2rem; margin: 0;">
                <li>1. Upload your file (.csv or .xlsx)</li>
                <li>2. Select review and NPS columns (auto-detected by default)</li>
                <li>3. Optional: Filter by supplier/vendor if available</li>
                <li>4. Select or add up to 10 aspects for analysis</li>
                <li>5. Click <b>Analyze Reviews</b></li>
                <li>6. Wait for analysis to finish</li>
                <li>7. View insights and visualizations</li>
                <li>8. Download CSV and PDF reports</li>
                <li>9. Chat with your data for recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        #st.info("Your data is processed locally in your browser; nothing is sent or stored.")

    # Main page header - Title and Tagline
    st.markdown("""
    <h1 class="main-title">SentimentIQ</h1>
    <p class="main-tagline">Turning Customer Voices into Victories with NLP-Powered Clarity</p>
    """, unsafe_allow_html=True)

    # File upload, loading, cleaning, and UI for selecting columns + aspects
    uploaded_file = st.file_uploader("📂 Upload your reviews file (CSV or XLSX)", type=["csv", "xlsx"])
    df = None
    df_error = None

    def clear_analysis_state():
        for key in ["absa_results", "absa_summary", "top_neg_reviews_by_aspect", "pdf_bytes", "chat_suggestions", "messages"]:
            if key in st.session_state:
                del st.session_state[key]

    user_aspects = []
    uploaded_count = None
    filtered_count = None
    num_after = None

    if uploaded_file:
        # Reset session if new file uploaded
        if "current_file_name" not in st.session_state or st.session_state["current_file_name"] != uploaded_file.name:
            clear_analysis_state()
            st.session_state["current_file_name"] = uploaded_file.name

        with st.spinner("Loading data..."):
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df = safe_read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                df = limit_large_df(df)
                if len(df) > 20000:
                    st.warning("Large file detected; processing may take longer.")
            except Exception as e:
                df_error = f"Failed to load file: {e}"

        if df is not None:
            uploaded_count = len(df)

            # Auto detect columns for reviews and NPS
            try:
                auto_review_col = auto_detect_review_column(df)
            except Exception as e:
                auto_review_col = df.columns[0] if len(df.columns) > 0 else ""
                st.error(f"Auto-detect review column failed: {e}")

            try:
                auto_nps_col = auto_detect_nps_column(df)
            except Exception:
                auto_nps_col = ""

            # Detect all categorical columns with manageable cardinality (e.g., max 100 unique values)
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            filtered_cat_cols = [col for col in cat_cols if df[col].nunique() <= 100]
            
            filtered_df = df.copy()
            
            if filtered_cat_cols:
                st.markdown("### Filter by a categorical column (Optional)")
            
                # Let user select a single categorical column to filter by
                selected_filter_column = st.selectbox(
                    "Select a categorical column to filter (optional):",
                    options=["None"] + filtered_cat_cols,
                    index=0
                )
                
                if selected_filter_column == "None":
                    selected_filter_column = None
                    
                if selected_filter_column:
                    value_counts = filtered_df[selected_filter_column].value_counts(dropna=True)
                    sorted_values = value_counts.index.tolist()
                
                    selected_values = st.multiselect(
                        f"Filter by values in '{selected_filter_column}' (optional):",
                        options=sorted_values,
                        key=f"filter_values_{selected_filter_column}"
                    )
                
                    if selected_values:
                        filtered_df = filtered_df[filtered_df[selected_filter_column].isin(selected_values)].copy()

                # Notify user if filtering excluded some rows
                if len(filtered_df) < len(df):
                    st.info(f"Filtered out {len(df) - len(filtered_df):,} reviews by categorical filter.")
            else:
                filtered_df = df.copy()


            # Remove invalid reviews
            invalid_markers = {'', 'unidentified', 'na', 'n/a', 'none', 'no review', 'unknown', 'nan', 'null', 'undefined', 'no_review', '-', '--', 'review unavailable', 'n\\a', '[blank]'}
            def is_valid_review(v):
                if pd.isna(v):
                    return False
                s = str(v).strip().lower()
                return s not in invalid_markers and len(s) > 0

            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df[auto_review_col].apply(is_valid_review)].copy()
            filtered_count = len(filtered_df)
            num_after = filtered_count
            filtered_out = before_count - filtered_count

            if filtered_out > 0:
                st.info(f"Filtered out {filtered_out:,} invalid or blank reviews.")

            if filtered_df.empty:
                st.error("No valid reviews after filtering. Please check your file.")

            # UI elements for selecting columns & aspects
            review_columns = filtered_df.columns.tolist()
            review_col_default_idx = review_columns.index(auto_review_col) if auto_review_col in review_columns else 0
            review_col = st.selectbox("Select review column", options=review_columns, index=review_col_default_idx)

            nps_options = ["<Auto Detect>"] + review_columns
            nps_col_default_idx = nps_options.index(auto_nps_col) if auto_nps_col in nps_options else 0
            nps_col = st.selectbox("Select NPS score column (0-10)", options=nps_options, index=nps_col_default_idx)
            if nps_col == "<Auto Detect>":
                nps_col = auto_nps_col if auto_nps_col else None

            # Aspects selection
            COMMON_ASPECTS = ["Quality", "Delivery", "Price", "Customer Service", "Packaging", "Refund", "Order", "Website", "Value", "Communication"]
            st.markdown("### Select up to 10 aspects (optional)")
            user_selected_aspects = st.multiselect("Choose common aspects", options=COMMON_ASPECTS)
            user_custom_aspects = st.text_input("Add custom aspects (comma-separated)")

            user_aspects_list = list(user_selected_aspects)
            if user_custom_aspects:
                user_aspects_list += [a.strip() for a in user_custom_aspects.split(",") if a.strip()]
            if len(user_aspects_list) > 10:
                st.warning("Maximum of 10 aspects allowed; extra ignored.")
            user_aspects = user_aspects_list[:10]

            df = filtered_df.copy()
        else:
            if df_error:
                st.error(df_error)


    def analyze_reviews(df, review_col, nps_col, user_aspects):
        lang = detect_language_of_reviews(df, review_col)
        if lang != "en":
            st.warning(f"Detected language: {lang}. Results may be less accurate.")

        with st.spinner("Analyzing reviews..."):
            start_time = time.time()

            df_out = analyze_review_structured(
                df,
                review_col=review_col,
                nps_col=nps_col,
                user_aspects=user_aspects
            )
            summary_df = generate_sentiment_summary(df_out)
            top_neg_reviews = extract_top_negative_reviews_by_aspect(df_out, summary_df['Aspect'].tolist()[:TOP_N_ASPECTS], max_reviews=5)
            recommendations = {a: build_recommendations_for_aspect(a, reviews) for a, reviews in top_neg_reviews.items()}

            global uploaded_count, filtered_out, num_after
            processing_stats = {
                'uploaded': uploaded_count or 0,
                'filtered_out': filtered_out or 0,
                'analysed': num_after or len(df),
                'unique_aspects': summary_df['Aspect'].nunique(),
                'mentions': len(df_out)
            }

            pdf_bytes = create_pdf_report(
                df_out,
                summary_df,
                top_neg_reviews,
                recommendations,
                processing_stats,
            )
            st.success(f"Analysis completed in {time.time() - start_time:.1f} seconds")
            return df_out, summary_df, top_neg_reviews, pdf_bytes


    def analyze_reviews_auto(df, review_col, nps_col):
        sample_texts = df[review_col].dropna().astype(str).tolist()
        words = re.findall(r'\b\w+\b', " ".join(sample_texts).lower())
        freq_words = [w for w, c in Counter(words).most_common(100) if len(w) > 3 and w.lower() not in {
            "this", "that", "there", "have", "with", "from", "much", "very", "will",
            "they", "what", "your", "not", "but", "are", "all", "can", "our"
        }]

        auto_aspects = freq_words[:TOP_N_ASPECTS] if freq_words else ["Quality", "Delivery", "Price", "Service"]
        auto_aspects = [a.title() for a in auto_aspects]

        df_out = analyze_review_structured(df, review_col=review_col, nps_col=nps_col, user_aspects=auto_aspects)
        summary_df = generate_sentiment_summary(df_out)
        top_neg_reviews = extract_top_negative_reviews_by_aspect(df_out, auto_aspects, max_reviews=5)
        recommendations = {a: build_recommendations_for_aspect(a, reviews) for a, reviews in top_neg_reviews.items()}

        global uploaded_count, filtered_out, num_after
        processing_stats = {
            'uploaded': uploaded_count or 0,
            'filtered_out': filtered_out or 0,
            'analysed': num_after or len(df),
            'unique_aspects': summary_df['Aspect'].nunique(),
            'mentions': len(df_out)
        }

        pdf_bytes = create_pdf_report(
            df_out,
            summary_df,
            top_neg_reviews,
            recommendations,
            processing_stats,
        )
        return df_out, summary_df, top_neg_reviews, pdf_bytes


    # Visualization Helpers

    def plot_nps_gauge(df):
        st.markdown("**Net Promoter Score (NPS) Analysis**")
        nps_colname = None
        for c in df.columns:
            if c.lower() in ('nps_score', 'nps', 'score', 'rating', 'net promoter score'):
                nps_colname = c
                break
        if not nps_colname:
            st.info("No NPS score column found.")
            return

        scores = df[nps_colname]
        scores = scores.dropna()
        scores = scores[(scores >= 0) & (scores <= 10)]

        total = len(scores)
        promoters = ((scores >= 9) & (scores <= 10)).sum()
        detractors = ((scores >= 0) & (scores <= 6)).sum()

        pct_promoters = promoters / total * 100 if total else 0
        pct_detractors = detractors / total * 100 if total else 0
        nps_score = pct_promoters - pct_detractors

        # Define gauge parameters
        gauges = [(-100, 0), (0, 30), (30, 70), (70, 100)]
        colors = ["#d7191c", "#fdae61", "#a6d96a", "#1a9641"]
        labels = ["Poor", "Fair", "Good", "Excellent"]
        ticks = [-100, 0, 30, 70, 100]

        fig, ax = plt.subplots(figsize=(8, 1.5))
        for i, (start, end) in enumerate(gauges):
            ax.barh(0, end - start, left=start, height=0.6, color=colors[i])

        ax.axvline(nps_score, 0, 1, color='navy', lw=3)
        ax.set_xlim(-100, 100)
        ax.set_yticks([])
        ax.set_xlabel("NPS Score", fontsize=11, fontweight='bold')

        for t in ticks:
            ax.text(t, -0.25, str(t), fontsize=10, fontweight='bold', ha='center')

        for i, (start, end) in enumerate(gauges):
            ax.text((start + end) / 2, -0.15, labels[i], fontsize=10, fontweight='bold', ha='center')

        ax.text(nps_score, 0.4, f"{nps_score:.1f}", fontsize=13, fontweight='bold', ha='center', color='navy')
        plt.box(False)
        plt.axis('off')
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
        st.caption("NPS = %Promoters − %Detractors; scale: -100 (worst) to 100 (best).")


    def plot_review_length(df, main_col, figsize=(8, 4.5)):
        st.markdown("**Review Length Distribution**")
        lengths = df[main_col].astype(str).str.len()
        lengths = lengths[lengths <= 500]
        fig, ax = plt.subplots(figsize=figsize)
        counts, bins, patches = ax.hist(lengths, bins=25, color="#2493b4", edgecolor='black', alpha=0.7)
        ax.set_xlim(0, 500)
        ax.set_xlabel("Review Length (characters)")
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Review Length Distribution", fontsize=14, fontweight='bold', pad=10)

        # Add bar labels
        for count, patch in zip(counts, patches):
            if count > 0:
                ax.text(patch.get_x() + patch.get_width() / 2, count, str(int(count)), ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    def plot_wordcloud(df, main_col):
        if not _wordcloud_available:
            return
        text_blob = " ".join(str(v) for v in df[main_col].dropna() if isinstance(v, str))
        if not text_blob.strip():
            st.info("Not enough text data for word cloud.")
            return
        wc = WordCloud(width=900, height=350, background_color='white', stopwords=STOPWORDS, max_words=120, colormap='Blues').generate(text_blob)
        st.image(wc.to_array(), use_container_width=True)
        st.caption("Word cloud: Most frequent words in reviews (excluding stopwords).")


    def plot_top_ngrams(df, main_col, n):
        top_n = 5  # Fixed number of n-grams to display
        # Map n to descriptive chart title
        ngram_titles = {
            2: "Bigrams (2-Word Phrases)",
            3: "Trigrams (3-Word Phrases)",
            4: "Four-Word Phrases (4-grams)"
        }
        title = ngram_titles.get(n, f"{n}-grams")
        st.markdown(f"**Top {top_n} {title} in Reviews**")
        texts = df[main_col].astype(str).tolist()
        cv = CountVectorizer(ngram_range=(n, n), stop_words='english', max_features=15)
        try:
            matrix = cv.fit_transform(texts)
            sums = matrix.sum(axis=0)
            freq = [(word, sums[0, idx]) for word, idx in cv.vocabulary_.items()]
            top_f = sorted(freq, key=lambda x: x[1], reverse=True)[:top_n]
        except Exception:
            top_f = []
        if not top_f:
            st.info("No frequent phrases found.")
            return
        labels, counts = zip(*top_f)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.6*len(labels))))
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, counts[::-1], color='#1f77b4')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels[::-1], fontsize=11)
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top {len(labels)} {title}")
        for bar in bars:
            ax.annotate(f"{int(bar.get_width())}", (bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2),
                        va='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def plot_aspect_popularity(summary_df):
        st.markdown("**Most Discussed Aspects by Sentiment**")
        df_ = summary_df.head(TOP_N_ASPECTS)
        aspects = df_['Aspect']
        pos = df_['Positive']
        neu = df_['Neutral']
        neg = df_['Negative']
        x = np.arange(len(aspects))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))
        bars_pos = ax.bar(x - width, pos, width, label='Positive', color='#2ca02c')
        bars_neu = ax.bar(x, neu, width, label='Neutral', color='#ffbb78')
        bars_neg = ax.bar(x + width, neg, width, label='Negative', color='#d62728')
        ax.set_xticks(x)
        ax.set_xticklabels(aspects, rotation=25, ha='right', fontsize=11)
        ax.set_ylabel("Mentions")
        ax.set_title("Most Discussed Aspects by Sentiment")
        ax.legend()
        for bars in [bars_pos, bars_neu, bars_neg]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f"{int(height)}", (bar.get_x() + bar.get_width() / 2, height + 0.5),
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def plot_top_negative_aspects(summary_df):
        st.markdown(f"**Most Discussed Negative Aspects by Sentiment**")
        df_ = summary_df.sort_values(by='Negative', ascending=False).head(TOP_N_ASPECTS)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(df_['Aspect'], df_['Negative'], color='#d62728')
        ax.set_ylabel("Negative Mentions")
        ax.set_title(f"Most Discussed Negative Aspects by Sentiment")
        ax.tick_params(axis='x', rotation=25)
        ax.tick_params(axis='x', labelsize=11)

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{int(height)}", (bar.get_x()+bar.get_width()/2, height + 0.5),
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def show_eda_metrics(df, summary_df, review_col):
        st.markdown("<h3>🔍 Data Summary & Analytics</h3>", unsafe_allow_html=True)
    
        analysed_count = len(df)
    
        # First row: 3 columns for first 3 KPIs
        col1, col2, col3 = st.columns(3)
    
        # 1. Total Reviews
        col1.metric("Total Reviews", f"{analysed_count:,}")
    
        # 2. Top Positive
        if not summary_df.empty and 'Positive' in summary_df.columns and summary_df['Positive'].notnull().any():
            sorted_pos_df = summary_df.sort_values(by="Positive", ascending=False)
            top_positive = sorted_pos_df.iloc[0]
            col2.metric("Top Positive", f"{top_positive['Aspect']} ({int(top_positive['Positive'])})")
        else:
            col2.metric("Top Positive", "-")
    
        # 3. Top Negative
        if not summary_df.empty and 'Negative' in summary_df.columns and summary_df['Negative'].notnull().any():
            sorted_neg_df = summary_df.sort_values(by="Negative", ascending=False)
            top_negative = sorted_neg_df.iloc[0]
            col3.metric("Top Negative", f"{top_negative['Aspect']} ({int(top_negative['Negative'])})")
        else:
            col3.metric("Top Negative", "-")
    
        # Second row: 2 columns for 4th KPI
        col4, _ = st.columns([2, 1])  # leave space empty for alignment
    
        # 4. Highest Negative %
        if not summary_df.empty and all(col in summary_df.columns for col in ['Negative', 'Total Mentions']):
            summary_df['Neg Ratio'] = summary_df['Negative'] / summary_df['Total Mentions']
            sorted_neg_ratio_df = summary_df.sort_values(by="Neg Ratio", ascending=False)
            worst_skewed = sorted_neg_ratio_df.iloc[0]
            col4.metric("Highest Negative %", f"{worst_skewed['Aspect']} ({worst_skewed['Neg Ratio']*100:.1f}%)")
        else:
            col4.metric("Highest Negative %", "-")
    
        # Third row: full width for 5th KPI
        col_full = st.columns(1)
        col = col_full[0]
    
        # 5. Most Polarizing
        needed_columns = ['Positive (%)', 'Negative (%)', 'Neutral (%)']
        if not summary_df.empty and all(col in summary_df.columns for col in needed_columns):
            summary_df['Polarity Gap'] = abs(summary_df['Positive (%)'] - summary_df['Negative (%)'])
            filtered_df = summary_df[summary_df['Neutral (%)'] < 50]
            if not filtered_df.empty:
                polarizing = filtered_df.sort_values(by="Polarity Gap", ascending=True).iloc[0]
                col.metric(
                    "Most Polarizing",
                    f"{polarizing['Aspect']} ({polarizing['Positive (%)']:.1f}% 👍 / {polarizing['Negative (%)']:.1f}% 👎)"
                )
            else:
                col.metric("Most Polarizing", "-")
        else:
            col.metric("Most Polarizing", "-")


    def plot_overall_sentiment(df_out):
        st.markdown("**Overall Sentiment Distribution (All Aspects)**")
        sentiments = df_out['Aspect_Sentiment'].value_counts()
        sentiments = sentiments.reindex(["Positive", "Neutral", "Negative"], fill_value=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(
            sentiments,
            labels=sentiments.index,
            autopct='%1.1f%%',
            colors=["#2ca02c", "#d3d3d3", "#d62728"],
            textprops={'fontsize': 14, 'fontweight': 'bold'}
        )
        ax.set_aspect('equal')
        #ax.set_title("Overall Sentiment Distribution", fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)


    # Main execution block

    if uploaded_file and df is not None:

        if st.button("Analyze Reviews"):
            if len(user_aspects) > 10:
                st.warning("Please select maximum 10 aspects.")
            else:
                try:
                    if user_aspects:
                        df_out, summary_df, top_neg_reviews, pdf_bytes = analyze_reviews(df, review_col, nps_col, user_aspects)
                    else:
                        df_out, summary_df, top_neg_reviews, pdf_bytes = analyze_reviews_auto(df, review_col, nps_col)

                    # Store results to session state
                    st.session_state["absa_results"] = df_out
                    st.session_state["absa_summary"] = summary_df
                    st.session_state["top_neg_reviews_by_aspect"] = top_neg_reviews
                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.session_state["chat_suggestions"] = None
                    st.session_state["messages"] = None

                except Exception as ex:
                    st.error(f"Analysis failed: {ex}")
                    st.code(traceback.format_exc())

        # Load results from session to display
        if "absa_results" in st.session_state and "absa_summary" in st.session_state:
            df_out = st.session_state["absa_results"]
            summary_df = st.session_state["absa_summary"]
            top_neg_reviews = st.session_state.get("top_neg_reviews_by_aspect", {})
            review_col_actual = review_col if review_col in df.columns else df.columns[0]

            # Show summary info
            uploaded_display = uploaded_count if uploaded_count else 0
            filtered_display = filtered_count if filtered_count else 0
            filtered_out = uploaded_display - filtered_display
            analysed_count = filtered_display  # actual analysed after filtering
            unique_aspects = summary_df['Aspect'].nunique()
            total_mentions = len(df_out)

            st.markdown(f"""
            <div style="background: #e6f2ff; border:1px solid #99c2ff; border-radius: 12px; padding: 15px; margin-bottom: 1.5rem; color: #004080; font-weight: 600;">
                <p><b>Data Summary & Processing</b></p>
                <ul style="list-style-type:none; padding-left: 0;">
                    <li>Uploaded Reviews: {uploaded_display:,}</li>
                    <li>Filtered Out: {filtered_out:,}</li>
                    <li>Analysed Reviews: {analysed_count:,}</li>
                    <li>Unique Aspects: {unique_aspects}</li>
                    <li>Total Aspect Mentions: {total_mentions}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Show analytics and charts
            show_eda_metrics(df, summary_df, review_col_actual)

            plot_nps_gauge(df_out)
            plot_overall_sentiment(df_out)

            plot_review_length(df, review_col_actual, figsize=(8, 4.5))

            plot_wordcloud(df, review_col_actual)

            st.markdown("---")

            plot_aspect_popularity(summary_df)

            # Top negative aspects plot with increased height and sorted by negative mentions
            st.markdown(f"### Most Discussed Negative Aspects by Sentiment")
            fig, ax = plt.subplots(figsize=(8, 5))
            neg_df = summary_df.sort_values(by="Negative", ascending=False).head(TOP_N_ASPECTS)
            bars = ax.bar(neg_df["Aspect"], neg_df["Negative"], color="#d62728")
            ax.set_ylabel("Negative Mentions")
            ax.set_title("Top Negative Aspects", fontsize=16, fontweight="bold")
            ax.tick_params(axis='x', rotation=25, labelsize=11)

            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2, height + 0.5),
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        fontweight='bold',
                        color='black'
                    )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Continue other ngram plots
            #plot_top_ngrams(df, review_col_actual, 2)
            plot_top_ngrams(df, review_col_actual, 3)
            plot_top_ngrams(df, review_col_actual, 4)

            # Downloads
            col1, col2, col3 = st.columns([3, 3, 4])
            col1.download_button(
                label="⬇️ Download Full Results (CSV)",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="sentimentiq_full_results.csv",
                mime="text/csv"
            )
            col2.download_button(
                label="⬇️ Download Summary (CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="sentimentiq_summary.csv",
                mime="text/csv"
            )
            col3.download_button(
                label="⬇️ Download Report (PDF)",
                data=st.session_state.get("pdf_bytes", b''),
                file_name="sentimentiq_report.pdf",
                mime="application/pdf"
            )

            st.caption("Results are generated locally and are not uploaded or saved on any server.")

            st.markdown("---")
            st.markdown("### Chat with Your Data")
            
            # Dynamic suggestions for chat UI
            def generate_suggestions(summ_df):
                suggestions = []
                top_aspects = summ_df["Aspect"].head(3).tolist()
                for aspect in top_aspects:
                    suggestions.append(f"Sentiment summary for {aspect}")
                    suggestions.append(f"Recommendations for {aspect}")
                suggestions.append("What are the main negative aspects?")
                return suggestions

            if "chat_suggestions" not in st.session_state or not st.session_state["chat_suggestions"]:
                st.session_state["chat_suggestions"] = generate_suggestions(summary_df)

            if "messages" not in st.session_state or st.session_state["messages"] is None:
                st.session_state["messages"] = [
                    {"role": "assistant", "content": "Hello! Ask me about sentiment, recommendations, or other insights from your data."}
                ]

            # Render chat history
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Show buttons for suggestions
            btn_cols = st.columns(len(st.session_state["chat_suggestions"]))
            clicked = None
            for i, sugg in enumerate(st.session_state["chat_suggestions"]):
                if btn_cols[i].button(sugg, key=f"sugg-{i}"):
                    clicked = sugg

            user_input = st.chat_input("Ask your question or pick a suggestion above.")
            user_text = clicked or user_input

            if user_text:
                st.session_state["messages"].append({"role": "user", "content": user_text})
                with st.chat_message("user"):
                    st.markdown(user_text)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Simple chatbot logic
                        def chatbot_response(text, summ_df, neg_reviews):
                            text_l = text.lower()
                            if "negative aspects" in text_l or "main negatives" in text_l or "top problems" in text_l:
                                negs = summ_df[summ_df["Dominant Sentiment"] == "Negative"].head(3)
                                if negs.empty:
                                    return "No significant negative aspects detected!"
                                reply = "Main negative aspects:\n"
                                for _, row in negs.iterrows():
                                    reply += f"- **{row['Aspect']}**, mentioned {int(row['Total Mentions'])} times, with {row['Negative (%)']:.2f}% negative.\n"
                                return reply

                            if any(k in text_l for k in ["sentiment", "nps"]):
                                m = re.search(r'(?:sentiment|nps) (?:for )?(.+)', text_l)
                                if not m:
                                    return "Please specify the aspect name, e.g., 'Sentiment for Delivery'."
                                asp = m.group(1).title()
                                filtered = summ_df[summ_df['Aspect'].str.lower() == asp.lower()]
                                if filtered.empty:
                                    filtered = summ_df[summ_df['Aspect'].str.contains(asp, case=False, na=False)]
                                if filtered.empty:
                                    return f"No data found for aspect '{asp}'."
                                row = filtered.iloc[0]
                                return (
                                    f"Sentiment for **{row['Aspect']}**:\n"
                                    f"- Positive: {row['Positive (%)']:.2f}%\n"
                                    f"- Neutral: {row['Neutral (%)']:.2f}%\n"
                                    f"- Negative: {row['Negative (%)']:.2f}%\n"
                                    f"- Dominant Sentiment: {row['Dominant Sentiment']}\n"
                                    f"- Average NPS: {row.get('Avg NPS', 'N/A')}\n"
                                    f"- Promoters: {int(row.get('Promoters', 0))}\n"
                                    f"- Passives: {int(row.get('Passives', 0))}\n"
                                    f"- Detractors: {int(row.get('Detractors', 0))}"
                                )

                            if any(k in text_l for k in ["recommendations", "suggestion"]):
                                m = re.search(r'(?:recommendations|suggestions) (?:for )?(.+)', text_l)
                                if not m:
                                    return "Please specify the aspect name for recommendations, e.g., 'Recommendations for Price'."
                                asp = m.group(1).title()
                                reviews_list = neg_reviews.get(asp, [])
                                if not reviews_list:
                                    # Try partial match
                                    match_key = next((k for k in neg_reviews if asp.lower() in k.lower()), None)
                                    reviews_list = neg_reviews.get(match_key, [])
                                # Use reviews to generate recommendations summary
                                recs = build_recommendations_for_aspect(asp, reviews_list)
                                if not recs:
                                    return f"No specific recommendations found for {asp}."
                                rec_text = "\n".join([f"{i+1}. {clean_text_for_pdf(r)}" for i, r in enumerate(recs)])
                                return f"Recommendations for **{asp}**:\n{rec_text}"

                            if "thank" in text_l or "bye" in text_l:
                                return "You're welcome! Feel free to ask any time."

                            # Default fallback
                            suggestions = st.session_state["chat_suggestions"]
                            sugg_str = "; ".join(f"`{s}`" for s in suggestions)
                            return f"Try asking: {sugg_str}"

                        reply_text = chatbot_response(user_text, summary_df, top_neg_reviews)
                        st.session_state["messages"].append({"role": "assistant", "content": reply_text})
                        st.markdown(reply_text)

            # Display footer
            st.markdown("""
                <div class="footer">
                    Built with ❤️ by SentimentIQ | Powered by Streamlit, spaCy, and NLP.
                </div>
            """, unsafe_allow_html=True)
    else:
        if uploaded_file and df_error:
            st.error(df_error)
