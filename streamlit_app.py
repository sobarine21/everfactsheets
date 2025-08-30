import streamlit as st
import pandas as pd
from fuzzywuzzy import process, fuzz
from io import BytesIO
import zipfile
import time

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Enforcement Index Portal")

# --- Caching & Core Logic (Rewritten for Speed & Precision) ---

@st.cache_data
def load_and_prepare_df(file_content: bytes) -> pd.DataFrame:
    """Loads a CSV from bytes into a DataFrame."""
    return pd.read_csv(BytesIO(file_content))

@st.cache_data
def perform_fuzzy_matching(
    _enforcement_df_hash: int,  # Add hash to bust cache when df changes
    enforcement_df: pd.DataFrame,
    constituents_df: pd.DataFrame,
    enforcement_cols_to_search: list[str],
    match_threshold: int,
) -> pd.DataFrame:
    """
    Performs optimized fuzzy matching on user-selected columns.

    This function is significantly faster because it:
    1. Collects all unique potential company names from the selected columns.
    2. Matches each unique name only ONCE.
    3. Maps the results back to the original DataFrame.
    """
    # 1. Prepare constituents data for fast lookups
    constituent_names = constituents_df['Name'].dropna().astype(str).str.strip().tolist()
    name_to_info = constituents_df.set_index('Name')[['Symbol', 'Sector']].to_dict('index')

    # Initialize results DataFrame to store the best match for each row
    results_df = pd.DataFrame(index=enforcement_df.index)
    results_df['Best Match Name'] = None
    results_df['Match Score'] = 0
    results_df['Source Column'] = None
    results_df['Source Value'] = None

    # 2. Iterate through ONLY the user-selected columns
    progress_bar = st.progress(0, text="Starting matching process...")
    total_cols = len(enforcement_cols_to_search)

    for i, col_name in enumerate(enforcement_cols_to_search):
        progress_text = f"Analyzing column '{col_name}' ({i+1}/{total_cols})..."
        progress_bar.progress((i) / total_cols, text=progress_text)

        # 3. Get unique non-empty values from the column to avoid re-matching
        unique_values = enforcement_df[col_name].dropna().astype(str).str.strip().unique()
        unique_values = [val for val in unique_values if len(val) > 2] # Filter out short strings

        if not unique_values:
            continue

        # 4. Create a match map for these unique values
        match_map = {}
        for val in unique_values:
            match = process.extractOne(
                val, constituent_names, scorer=fuzz.token_sort_ratio, score_cutoff=match_threshold
            )
            if match:
                match_map[val] = {'match_name': match[0], 'score': match[1]}

        # 5. Apply the matches back to the column
        # This is much faster than iterating row-by-row
        col_matches = enforcement_df[col_name].map(match_map).dropna()

        if col_matches.empty:
            continue
            
        # 6. Update the main results if the current column provides a better match
        # Create a temporary DataFrame for comparison
        temp_df = pd.DataFrame(col_matches.tolist(), index=col_matches.index)
        temp_df['source_column'] = col_name
        temp_df['source_value'] = enforcement_df.loc[col_matches.index, col_name]

        # Find rows where the new score is better than the existing best score
        is_better_match = temp_df['score'] > results_df.loc[temp_df.index, 'Match Score']
        
        # Update only those rows
        results_df.loc[temp_df.index[is_better_match], 'Best Match Name'] = temp_df.loc[is_better_match, 'match_name']
        results_df.loc[temp_df.index[is_better_match], 'Match Score'] = temp_df.loc[is_better_match, 'score']
        results_df.loc[temp_df.index[is_better_match], 'Source Column'] = temp_df.loc[is_better_match, 'source_column']
        results_df.loc[temp_df.index[is_better_match], 'Source Value'] = temp_df.loc[is_better_match, 'source_value']


    progress_bar.progress(1.0, text="Matching complete!")
    time.sleep(1)
    progress_bar.empty()

    # 7. Add Symbol and Sector from the best match
    results_df['Matched Constituent Symbol'] = results_df['Best Match Name'].map(lambda x: name_to_info.get(x, {}).get('Symbol'))
    results_df['Matched Constituent Sector'] = results_df['Best Match Name'].map(lambda x: name_to_info.get(x, {}).get('Sector'))

    # 8. Combine original data with matching results
    final_df = pd.concat([enforcement_df, results_df], axis=1)
    return final_df


@st.cache_data
def calculate_enforcement_score(
    _matched_df_hash: int,
    matched_df: pd.DataFrame,
    calculation_method: str,
    score_multiplier: float,
    fine_column: str,
) -> pd.Series:
    """Calculates the enforcement score for each company."""
    if calculation_method == 'Count Enforcements':
        return matched_df.groupby('Matched Constituent Symbol').size() * score_multiplier
    
    # Ensure fine column exists and is numeric
    if not fine_column or fine_column not in matched_df.columns:
        st.error(f"Fine column '{fine_column}' not found. Please re-run matching or select a different method.")
        return pd.Series(dtype='float64')
        
    matched_df[fine_column] = pd.to_numeric(matched_df[fine_column], errors='coerce').fillna(0)

    if calculation_method == 'Sum Fines':
        return matched_df.groupby('Matched Constituent Symbol')[fine_column].sum()
    elif calculation_method == 'Average Fines':
        return matched_df.groupby('Matched Constituent Symbol')[fine_column].mean()
    elif calculation_method == 'Max Fine':
        return matched_df.groupby('Matched Constituent Symbol')[fine_column].max()
    
    return pd.Series(dtype='float64')


# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'constituents_df': None,
        'enforcement_dfs': {},
        'combined_enforcement_df': None,
        'matched_df': None,
        'final_ranked_df': None,
        'data_loaded': False,
        'matching_complete': False,
        'index_calculated': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# --- UI Layout & Styling ---
st.markdown("""
<style>
    .stButton>button { font-weight: bold; }
    h1 { color: #1e3a8a; text-align: center; }
    .success-banner { background: #10b981; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; font-weight: bold;}
    .info-box { background: #e0f2fe; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #0277bd; margin: 1rem 0;}
    .warning-box { background: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Fast & Precise Enforcement Index Portal")
st.markdown("**Step-by-step wizard to upload, match, and generate enforcement indices.**")


# --- Main Application ---
tab1, tab2, tab3 = st.tabs(["**Step 1: 🚀 Load Data**", "**Step 2: ⚙️ Match Companies**", "**Step 3: 📊 Generate Index**"])

# ==============================================================================
# TAB 1: LOAD DATA
# ==============================================================================
with tab1:
    st.header("Upload Your Data Files")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("A) Index Constituents File (CSV)")
        st.info("**Required columns:** `Name`, `Symbol`, `Sector`.")
        uploaded_constituents = st.file_uploader("Upload a single CSV file for constituents.", type="csv")
        
        if uploaded_constituents:
            try:
                df = load_and_prepare_df(uploaded_constituents.getvalue())
                required = ['Name', 'Symbol', 'Sector']
                if not all(col in df.columns for col in required):
                    st.error(f"Constituents file is missing required columns. It must contain: {', '.join(required)}")
                    st.session_state.constituents_df = None
                else:
                    st.session_state.constituents_df = df
                    st.success(f"✅ Loaded {len(df):,} constituent companies.")
                    st.dataframe(df[required].head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading constituents file: {e}")

    with col2:
        st.subheader("B) Enforcement Data Files (CSV)")
        st.info("Upload one or more CSV files with enforcement records.")
        uploaded_enforcement_files = st.file_uploader("Upload enforcement files.", type="csv", accept_multiple_files=True)

        if uploaded_enforcement_files:
            st.session_state.enforcement_dfs = {}
            for file in uploaded_enforcement_files:
                try:
                    df = load_and_prepare_df(file.getvalue())
                    st.session_state.enforcement_dfs[file.name] = df
                    st.success(f"✅ Loaded '{file.name}' with {len(df):,} records.")
                except Exception as e:
                    st.error(f"Error loading '{file.name}': {e}")
    
    # Check if both sets of data are loaded to proceed
    if st.session_state.constituents_df is not None and st.session_state.enforcement_dfs:
        st.session_state.data_loaded = True
        # Combine all enforcement data into a single DataFrame for easier processing
        st.session_state.combined_enforcement_df = pd.concat(st.session_state.enforcement_dfs.values(), ignore_index=True)
        st.markdown(f'<div class="success-banner">🎉 Data Loaded! Total {len(st.session_state.combined_enforcement_df):,} enforcement records ready. Proceed to Step 2.</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 2: CONFIGURE & MATCH
# ==============================================================================
with tab2:
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload both Constituents and Enforcement data in Step 1.")
        st.stop()
    
    st.header("Configure Matching Parameters")

    # --- Column Selection (The Key New Feature) ---
    st.subheader("1. Select Columns to Search for Company Names")
    enforcement_cols = st.session_state.combined_enforcement_df.columns.tolist()
    # Suggest columns that are likely to contain company names
    suggested_cols = [c for c in enforcement_cols if any(keyword in c.lower() for keyword in ['name', 'company', 'entity', 'organization', 'respondent', 'firm'])]
    
    selected_cols = st.multiselect(
        "From your enforcement files, select all columns that might contain company names.",
        options=enforcement_cols,
        default=suggested_cols,
        help="The tool will search for matches ONLY in these columns. This makes it faster and more precise."
    )

    # --- Other Configurations ---
    st.subheader("2. Set Matching Threshold")
    match_threshold = st.slider("Fuzzy Match Threshold (%)", 50, 100, 80, help="Higher values require a closer match.")

    if not selected_cols:
        st.warning("Please select at least one column to search for company names.")
        st.stop()
        
    st.markdown("---")
    
    # --- Execute Matching ---
    if st.button("🚀 Run Company Matching", type="primary", use_container_width=True):
        with st.spinner("Performing high-speed matching... This may take a moment for large datasets."):
            try:
                # Use hash of the dataframe to ensure caching works correctly with mutable objects
                enforcement_df_hash = pd.util.hash_pandas_object(st.session_state.combined_enforcement_df).sum()
                
                matched_df = perform_fuzzy_matching(
                    enforcement_df_hash,
                    st.session_state.combined_enforcement_df,
                    st.session_state.constituents_df,
                    selected_cols,
                    match_threshold
                )
                st.session_state.matched_df = matched_df
                st.session_state.matching_complete = True
                st.balloons()
                st.success("✅ Matching Complete!")

            except Exception as e:
                st.error(f"An error occurred during matching: {e}")
                st.stop()
    
    # --- Display Matching Results ---
    if st.session_state.matching_complete:
        matched_df = st.session_state.matched_df
        num_matched = matched_df['Matched Constituent Symbol'].notna().sum()
        total_records = len(matched_df)
        match_rate = (num_matched / total_records * 100) if total_records > 0 else 0

        st.subheader("Matching Results Summary")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Records", f"{total_records:,}")
        m_col2.metric("Matched Records", f"{num_matched:,}")
        m_col3.metric("Match Rate", f"{match_rate:.1f}%")

        if num_matched > 0:
            st.subheader("✅ Sample Matched Records")
            preview_cols = ['Source Column', 'Source Value', 'Best Match Name', 'Match Score', 'Matched Constituent Symbol']
            st.dataframe(matched_df[matched_df['Best Match Name'].notna()][preview_cols].head(10), use_container_width=True)

            st.subheader("📊 Matches by Source Column")
            st.info("This chart shows which of your selected columns produced the most matches.")
            source_counts = matched_df['Source Column'].value_counts()
            st.bar_chart(source_counts)
            
        else:
            st.warning("No matches were found. Try lowering the match threshold or selecting different columns.")

# ==============================================================================
# TAB 3: GENERATE & DOWNLOAD
# ==============================================================================
with tab3:
    if not st.session_state.matching_complete:
        st.warning("⚠️ Please run the matching process in Step 2 first.")
        st.stop()
        
    st.header("Generate Index and Download Results")
    
    # --- Index Calculation Configuration ---
    st.subheader("1. Configure Index Calculation")
    col1, col2 = st.columns(2)
    with col1:
        calculation_method = st.selectbox(
            "Index Calculation Method",
            ['Count Enforcements', 'Sum Fines', 'Average Fines', 'Max Fine'],
            key='calc_method'
        )
    with col2:
        if calculation_method == 'Count Enforcements':
            score_multiplier = st.number_input("Score Multiplier", 0.1, 10.0, 1.0, 0.1)
            fine_column = None
        else:
            # Detect potential fine columns
            numeric_cols = st.session_state.matched_df.select_dtypes(include='number').columns.tolist()
            potential_fine_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in ['fine', 'penalty', 'amount', 'sanction'])]
            fine_column = st.selectbox(
                "Select the Fine/Penalty Column",
                options=st.session_state.matched_df.columns,
                index=st.session_state.matched_df.columns.get_loc(potential_fine_cols[0]) if potential_fine_cols else 0
            )
            score_multiplier = 1.0
    
    # --- Execute Calculation ---
    if st.button("📊 Calculate Enforcement Index", type="primary", use_container_width=True):
        with st.spinner("Calculating scores and ranking..."):
            valid_matches = st.session_state.matched_df.dropna(subset=['Matched Constituent Symbol']).copy()
            if valid_matches.empty:
                st.error("No valid matches found to calculate an index.")
                st.stop()

            # Use hash for caching
            matched_df_hash = pd.util.hash_pandas_object(valid_matches).sum()
            
            scores = calculate_enforcement_score(
                matched_df_hash,
                valid_matches,
                calculation_method,
                score_multiplier,
                fine_column
            )
            
            final_df = st.session_state.constituents_df.copy()
            final_df = final_df.set_index('Symbol')
            final_df['Enforcement Score'] = scores
            final_df['Enforcement Score'].fillna(0, inplace=True)
            
            # Add summary stats
            final_df['Total Enforcements'] = valid_matches.groupby('Matched Constituent Symbol').size()
            final_df['Total Enforcements'].fillna(0, inplace=True)

            final_df = final_df.sort_values(by='Enforcement Score', ascending=False).reset_index()
            final_df['Rank'] = range(1, len(final_df) + 1)
            
            # Reorder columns for clarity
            cols_order = ['Rank', 'Symbol', 'Name', 'Sector', 'Enforcement Score', 'Total Enforcements']
            final_df = final_df[cols_order + [c for c in final_df.columns if c not in cols_order]]

            st.session_state.final_ranked_df = final_df
            st.session_state.index_calculated = True
            st.success("✅ Index Calculation Complete!")
            
    # --- Display Final Results & Download ---
    if st.session_state.index_calculated:
        final_df = st.session_state.final_ranked_df
        st.subheader("🏆 Final Ranked Index")
        st.dataframe(final_df.head(20), use_container_width=True)

        st.subheader("📥 Download Results")
        col1, col2 = st.columns(2)
        with col1:
            ranked_csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Ranked Index (CSV)", ranked_csv, "ranked_index.csv", "text/csv", use_container_width=True)
        with col2:
            detailed_csv = st.session_state.matched_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Detailed Matches (CSV)", detailed_csv, "detailed_matches.csv", "text/csv", use_container_width=True)
        
        # ZIP Package
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ranked_index.csv", ranked_csv)
            zf.writestr("detailed_matching_results.csv", detailed_csv)
        st.download_button("📦 Download All as ZIP", zip_buffer.getvalue(), "enforcement_index_package.zip", "application/zip", use_container_width=True)
