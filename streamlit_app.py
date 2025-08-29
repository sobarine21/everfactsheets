# Streamlit Index Creation & Enforcement Scoring Portal
# ---------------------------------------------------
# Features
# - Upload a single "Index Constituents" CSV with columns: Symbol, Name, Country, Sector
# - Upload multiple CSV files containing potential enforcement-related rows mentioning company names
# - Fuzzy match company Name to text/names in uploaded files with adjustable similarity threshold (default 75%)
# - Flexible scoring: count hits, or sum a numeric score column if available (e.g., severity, score, weight, penalty_amount)
# - Optional date-aware time decay if a date column exists (half-life adjustable)
# - Multiple normalization methods to compute an "Enforcement Index" (Raw, Min-Max to 100, Z-Score, Log1p)
# - Ranks companies by enforcement index and provides downloadable CSVs
# - Robust matching via RapidFuzz (with safe fallback to difflib if RapidFuzz unavailable)
# - Caching for performance, progress indicators, and clear UI

import io
import math
import re
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- Try RapidFuzz for fast and accurate fuzzy matching; fallback to difflib ---
try:
    from rapidfuzz import process, fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    import difflib
    HAVE_RAPIDFUZZ = False

# -------------------------
# Utility & Matching Helpers
# -------------------------

def _clean_text(x: str) -> str:
    if pd.isna(x):
        return ""
    # normalize whitespace and case; strip punctuation except & and . which may be part of names
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"[^a-z0-9&.\-() ]", "", s)
    return s

@st.cache_data(show_spinner=False)
def load_constituents(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    required = {"Symbol", "Name", "Country", "Sector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Constituent file missing required columns: {sorted(missing)}")
    df = df.copy()
    df["__name_clean"] = df["Name"].map(_clean_text)
    if df["__name_clean"].duplicated().any():
        st.warning("Duplicate cleaned names detected in constituents; matching may merge them.")
    return df

@st.cache_data(show_spinner=False)
def load_bulk_csv(file, encoding_options: List[str]) -> pd.DataFrame:
    # Try user-provided encodings, fall back to utf-8
    last_err = None
    for enc in encoding_options + ["utf-8", "latin-1"]:
        try:
            df = pd.read_csv(file, encoding=enc)
            return df
        except Exception as e:
            last_err = e
            continue
    raise last_err

@st.cache_data(show_spinner=False)
def list_text_like_columns(df: pd.DataFrame) -> List[str]:
    # Heuristic: object dtype or any column with 'name' in it
    cols = [c for c in df.columns if df[c].dtype == object]
    rank = []
    for c in cols:
        score = 0
        cl = c.lower()
        if "name" in cl:
            score += 2
        if any(k in cl for k in ["text", "title", "description", "narrative"]):
            score += 1
        rank.append((score, c))
    rank.sort(reverse=True)
    return [c for _, c in rank] or cols

@st.cache_data(show_spinner=False)
def list_possible_date_columns(df: pd.DataFrame) -> List[str]:
    cands = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["date", "dt", "created", "updated", "time"]):
            cands.append(c)
    return cands

@st.cache_data(show_spinner=False)
def list_possible_score_columns(df: pd.DataFrame) -> List[str]:
    cands = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["score", "weight", "severity", "penalty", "amount", "fine", "points", "impact"]):
            cands.append(c)
        elif np.issubdtype(df[c].dtype, np.number):
            # numeric columns as fallback candidates
            cands.append(c)
    # Keep unique order
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

# Fuzzy matching core

def best_match(target: str, choices: List[str], threshold: int) -> Tuple[Optional[str], float]:
    """Return best matching choice and score (0-100). If below threshold, returns (None, 0)."""
    t = _clean_text(target)
    if not t:
        return None, 0.0
    if HAVE_RAPIDFUZZ:
        m = process.extractOne(t, choices, scorer=fuzz.token_set_ratio)
        if not m:
            return None, 0.0
        choice, score, _ = m
        return (choice, float(score)) if score >= threshold else (None, float(score))
    else:
        m = difflib.get_close_matches(t, choices, n=1, cutoff=threshold / 100.0)
        if not m:
            return None, 0.0
        # Approximate a score using SequenceMatcher
        s = int(100 * difflib.SequenceMatcher(None, t, m[0]).ratio())
        return (m[0], float(s)) if s >= threshold else (None, float(s))

# -------------------
# Scoring & Indexing
# -------------------

def apply_time_decay(score_series: pd.Series, dates: pd.Series, half_life_days: float) -> pd.Series:
    if dates is None or half_life_days is None or half_life_days <= 0:
        return score_series
    # Convert dates to datetime and compute age in days
    dt = pd.to_datetime(dates, errors="coerce")
    age_days = (pd.Timestamp.utcnow().tz_localize(None) - dt).dt.days
    decay = np.power(0.5, age_days / half_life_days)
    decay = decay.fillna(1.0)
    return score_series * decay

def normalize_scores(s: pd.Series, method: str) -> pd.Series:
    method = method.lower()
    if method == "raw":
        return s
    if method == "min-max (0–100)":
        if s.max() == s.min():
            return pd.Series(0.0, index=s.index)
        return 100 * (s - s.min()) / (s.max() - s.min())
    if method == "z-score":
        mu, sigma = s.mean(), s.std(ddof=0)
        if sigma == 0:
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sigma
    if method == "log1p":
        return np.log1p(s.clip(lower=0))
    return s

# -------------------
# Streamlit UI
# -------------------

st.set_page_config(page_title="Index Creation & Enforcement Scoring Portal", layout="wide")

st.title("📊 Index Creation & Enforcement Scoring Portal")
st.caption(
    "Upload index constituents and bulk CSVs of enforcement-related data. We'll fuzzy-match names (≥ threshold) "
    "and compute enforcement scores and indices with customizable options."
)

with st.sidebar:
    st.header("⚙️ Matching & Scoring Settings")
    threshold = st.slider("Name match threshold (%, token-set)", 50, 95, 75, 1,
                          help="Lower threshold matches more loosely; 75% recommended.")
    encodings = st.text_input("CSV encodings to try (comma-separated)", "utf-8,utf-16,cp1252").split(",")
    encodings = [e.strip() for e in encodings if e.strip()]

    st.subheader("Scoring")
    scoring_mode = st.selectbox(
        "Per-row contribution",
        [
            "Count each matched row as 1",
            "Use a numeric score column if available",
        ],
        help=(
            "If 'numeric score' is selected but the chosen column is missing/invalid in a file, "
            "that file falls back to count=1 per matched row."
        ),
    )

    st.subheader("Optional Date Decay")
    use_decay = st.checkbox("Apply time decay if a date column exists", value=False)
    half_life = st.slider("Half-life (days)", 7, 720, 180, 1,
                          help="How fast past events lose weight. 180 days = score halves every 6 months.") if use_decay else None

    st.subheader("Index Normalization")
    norm_method = st.selectbox("Index method", ["Raw", "Min-Max (0–100)", "Z-Score", "Log1p"], index=1)

st.markdown("---")

c_left, c_right = st.columns([1, 1])
with c_left:
    st.subheader("1) Upload Constituents CSV")
    cons_file = st.file_uploader(
        "Constituents (CSV with columns: Symbol, Name, Country, Sector)", type=["csv"], accept_multiple_files=False
    )

with c_right:
    st.subheader("2) Upload Bulk CSV Files (Multiple)")
    bulk_files = st.file_uploader(
        "Upload one or more CSV files: these contain rows mentioning company names", type=["csv"], accept_multiple_files=True
    )

if cons_file is None:
    st.info("Please upload the constituents CSV to begin.")
    st.stop()

# Load constituents
try:
    constituents = load_constituents(cons_file)
except Exception as e:
    st.error(f"Failed to read constituents CSV: {e}")
    st.stop()

st.success(f"Loaded {len(constituents)} constituents.")

if not bulk_files:
    st.warning("Upload at least one bulk CSV file to compute enforcement scores.")
    st.stop()

# Prepare matching search space (cleaned constituent names)
choices = constituents["__name_clean"].tolist()

# Containers for results
all_matches = []  # detailed row-level matches
per_file_summaries = []

progress = st.progress(0)
status = st.empty()

for i, file in enumerate(bulk_files, start=1):
    status.info(f"Reading file {i}/{len(bulk_files)}: {file.name}")
    try:
        df = load_bulk_csv(file, encodings)
    except Exception as e:
        st.error(f"Failed to read {file.name}: {e}")
        continue

    if df.empty:
        st.warning(f"{file.name}: empty file; skipping")
        continue

    text_cols = list_text_like_columns(df)
    date_cols = list_possible_date_columns(df)
    score_cols = list_possible_score_columns(df)

    with st.expander(f"Configure columns for {file.name}", expanded=False):
        sel_text_col = st.selectbox(
            f"Text/Name column to search in ({file.name})", text_cols or list(df.columns), key=f"textcol_{i}"
        )
        sel_date_col = (
            st.selectbox(f"Date column (optional) in {file.name}", ["<none>"] + date_cols, index=0, key=f"datecol_{i}")
        )
        if scoring_mode == "Use a numeric score column if available":
            sel_score_col = st.selectbox(
                f"Numeric score column (fallback to 1 if invalid) in {file.name}", ["<none>"] + score_cols, index=0, key=f"scorecol_{i}"
            )
        else:
            sel_score_col = "<none>"

    s_text = df[sel_text_col].astype(str)
    if sel_date_col != "<none>":
        s_date = df[sel_date_col]
    else:
        s_date = None

    # Determine per-row base weight
    if scoring_mode == "Use a numeric score column if available" and sel_score_col != "<none>" and sel_score_col in df.columns:
        # Coerce to numeric with errors as NaN then fill 1.0 fallback
        s_weight = pd.to_numeric(df[sel_score_col], errors="coerce").fillna(1.0)
    else:
        s_weight = pd.Series(1.0, index=df.index)

    # Optional time decay
    if use_decay and s_date is not None:
        s_weight = apply_time_decay(s_weight, s_date, half_life)

    # Perform matching row by row (vectorized via choices search list)
    # For performance, pre-clean the series
    s_text_clean = s_text.map(_clean_text)

    matched_choice = []
    match_score = []

    # Use RapidFuzz batch if available for speed
    if HAVE_RAPIDFUZZ:
        from rapidfuzz import process, fuzz
        # Build a dict index to speed up repeated lookups
        # process.extractOne is already optimized; iterate rows
        for val in s_text_clean:
            if not val:
                matched_choice.append(None)
                match_score.append(0.0)
                continue
            m = process.extractOne(val, choices, scorer=fuzz.token_set_ratio)
            if m and m[1] >= threshold:
                matched_choice.append(m[0])
                match_score.append(float(m[1]))
            else:
                matched_choice.append(None)
                match_score.append(float(m[1]) if m else 0.0)
    else:
        import difflib
        for val in s_text_clean:
            if not val:
                matched_choice.append(None)
                match_score.append(0.0)
                continue
            m = difflib.get_close_matches(val, choices, n=1, cutoff=threshold / 100.0)
            if m:
                score = int(100 * difflib.SequenceMatcher(None, val, m[0]).ratio())
                if score >= threshold:
                    matched_choice.append(m[0])
                    match_score.append(float(score))
                else:
                    matched_choice.append(None)
                    match_score.append(float(score))
            else:
                matched_choice.append(None)
                match_score.append(0.0)

    # Collect matched rows only
    df_matched = df.loc[pd.Series(matched_choice).notna()].copy()
    if not df_matched.empty:
        df_matched["__matched_name_clean"] = [mc for mc in matched_choice if mc is not None]
        df_matched["__match_score"] = [ms for mc, ms in zip(matched_choice, match_score) if mc is not None]
        df_matched["__row_weight"] = s_weight.loc[df_matched.index].values
        df_matched["__source_file"] = file.name
        # Map back to the full constituent row for output enrichment
        cons_map = constituents.set_index("__name_clean")[
            ["Symbol", "Name", "Country", "Sector"]
        ]
        df_matched = df_matched.join(cons_map, on="__matched_name_clean", how="left")

        # Reorder columns: index fields first
        front_cols = [
            "Symbol", "Name", "Country", "Sector",
            "__match_score", "__row_weight", "__source_file"
        ]
        front_cols = [c for c in front_cols if c in df_matched.columns]
        other_cols = [c for c in df_matched.columns if c not in front_cols]
        df_matched = df_matched[front_cols + other_cols]

        all_matches.append(df_matched)

        # Summarize per file
        per_file_summaries.append({
            "file": file.name,
            "rows": len(df),
            "matched_rows": len(df_matched),
            "distinct_companies": df_matched["__matched_name_clean"].nunique(),
            "avg_match_score": round(df_matched["__match_score"].mean(), 2),
        })
    else:
        per_file_summaries.append({
            "file": file.name,
            "rows": len(df),
            "matched_rows": 0,
            "distinct_companies": 0,
            "avg_match_score": 0.0,
        })

    progress.progress(i / len(bulk_files))

status.empty()

if not all_matches:
    st.error("No matches found across the uploaded files with the current threshold. Try lowering the threshold.")
    st.stop()

matches = pd.concat(all_matches, ignore_index=True)

st.markdown("### ✅ Matched Enforcement Rows")
st.dataframe(matches.head(500))

# --- Aggregate to company-level scores ---
# Base contribution = sum of row weights (already decayed if configured)
company_scores = (
    matches.groupby(["Symbol", "Name", "Country", "Sector"], dropna=False)["__row_weight"].sum().rename("raw_score").reset_index()
)

# Normalize index
company_scores["enforcement_index"] = normalize_scores(company_scores["raw_score"], norm_method)

# Rank (1 = highest enforcement)
company_scores = company_scores.sort_values(["enforcement_index", "raw_score", "Name"], ascending=[False, False, True])
company_scores["rank"] = np.arange(1, len(company_scores) + 1)
company_scores = company_scores[["rank", "Symbol", "Name", "Country", "Sector", "raw_score", "enforcement_index"]]

# --- Per-file summary ---
summary_df = pd.DataFrame(per_file_summaries)

st.markdown("---")
colA, colB = st.columns([2, 1])
with colA:
    st.subheader("🏆 Company Enforcement Scores & Index")
    st.dataframe(company_scores)
with colB:
    st.subheader("📁 File-Level Summary")
    st.dataframe(summary_df)

# --- Downloads ---
@st.cache_data(show_spinner=False)
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="⬇️ Download Matched Rows CSV",
        data=_to_csv_bytes(matches),
        file_name="matched_rows_detailed.csv",
        mime="text/csv",
    )
with col2:
    st.download_button(
        label="⬇️ Download Company Scores CSV",
        data=_to_csv_bytes(company_scores),
        file_name="company_enforcement_scores.csv",
        mime="text/csv",
    )

# --- Simple Charts (optional) ---
try:
    import altair as alt
    top_n = st.slider("Show Top N in chart", 5, min(50, len(company_scores)), min(15, len(company_scores)))
    chart_df = company_scores.head(top_n)
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Name:N", sort=None),
        y=alt.Y("enforcement_index:Q"),
        tooltip=["rank", "Name", "raw_score", "enforcement_index"],
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)
except Exception:
    pass

st.markdown("---")

st.info(
    "**Notes**:\n"
    "• Matching is performed against the constituents' `Name` only (as requested). All constituent fields are included in outputs.\n"
    "• If a numeric score column is selected but contains non-numeric values, those rows default to weight=1.\n"
    "• Time decay applies per-row prior to aggregation when a date column is provided.\n"
    "• Index normalization is applied to aggregated company scores to produce the `enforcement_index`."
)

st.caption("Built with ❤️ in Streamlit. For best results, ensure your constituents CSV uses consistent naming.")
