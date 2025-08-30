import streamlit as st
import pandas as pd
from fuzzywuzzy import process, fuzz
import numpy as np
from io import StringIO, BytesIO
import zipfile
import base64
from typing import Dict, List, Tuple, Optional
import time

# Performance optimizations
st.set_page_config(layout="wide", page_title="Enforcement Index Portal")

# Cache expensive operations
@st.cache_data
def load_csv_file(file_content: bytes, filename: str) -> pd.DataFrame:
    """Cached CSV loading function"""
    return pd.read_csv(BytesIO(file_content))

@st.cache_data
def fuzzy_match_companies_cached(enforcement_data: str, constituents_data: str, 
                                match_threshold: int = 75, company_col: str = "Company Name") -> str:
    """Cached fuzzy matching with serialized DataFrames"""
    # Deserialize DataFrames
    enforcement_df = pd.read_json(StringIO(enforcement_data))
    constituents_df = pd.read_json(StringIO(constituents_data))
    
    # Ensure company name column exists
    if company_col not in enforcement_df.columns:
        # Try to find the best column
        potential_cols = [col for col in enforcement_df.columns 
                         if any(keyword in col.lower() for keyword in ['company', 'name', 'entity'])]
        if potential_cols:
            company_col = potential_cols[0]
            enforcement_df.rename(columns={company_col: 'Company Name'}, inplace=True)
        else:
            raise ValueError("No suitable company name column found")
    elif company_col != 'Company Name':
        enforcement_df.rename(columns={company_col: 'Company Name'}, inplace=True)
    
    # Prepare constituent names list for faster matching
    constituent_names = constituents_df['Name'].dropna().astype(str).tolist()
    
    # Pre-allocate results for better performance
    results = []
    
    # Vectorized operations where possible
    enforcement_df['Company Name'] = enforcement_df['Company Name'].fillna('').astype(str)
    
    # Batch process for better performance
    batch_size = 100
    total_rows = len(enforcement_df)
    
    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        batch_df = enforcement_df.iloc[i:batch_end]
        
        batch_results = []
        for _, row in batch_df.iterrows():
            company_name = row['Company Name'].strip()
            
            if not company_name:
                batch_results.append({
                    'Best Match Name': None, 'Match Score': 0,
                    'Matched Constituent Symbol': None, 'Matched Constituent Country': None,
                    'Matched Constituent Sector': None
                })
                continue
            
            matches = process.extractOne(company_name, constituent_names, 
                                       scorer=fuzz.token_sort_ratio, score_cutoff=match_threshold)
            
            if matches:
                matched_name, score = matches
                matched_row = constituents_df[constituents_df['Name'] == matched_name].iloc[0]
                batch_results.append({
                    'Best Match Name': matched_name, 'Match Score': score,
                    'Matched Constituent Symbol': matched_row.get('Symbol'),
                    'Matched Constituent Country': matched_row.get('Country'),
                    'Matched Constituent Sector': matched_row.get('Sector')
                })
            else:
                batch_results.append({
                    'Best Match Name': None, 'Match Score': 0,
                    'Matched Constituent Symbol': None, 'Matched Constituent Country': None,
                    'Matched Constituent Sector': None
                })
        
        results.extend(batch_results)
    
    # Combine results
    results_df = pd.DataFrame(results)
    final_df = pd.concat([enforcement_df.reset_index(drop=True), results_df], axis=1)
    
    # Return serialized result
    return final_df.to_json()

@st.cache_data
def calculate_enforcement_score_cached(grouped_data: str, calculation_method: str, 
                                     score_multiplier: float = 1.0) -> str:
    """Cached enforcement score calculation"""
    # Deserialize grouped data
    grouped_df = pd.read_json(StringIO(grouped_data))
    
    if calculation_method == 'Count Enforcements':
        scores = grouped_df.groupby('Matched Constituent Symbol').size() * score_multiplier
    elif calculation_method == 'Sum Fines':
        if 'Fine Amount' not in grouped_df.columns:
            raise ValueError("'Fine Amount' column required for Sum Fines method")
        scores = grouped_df.groupby('Matched Constituent Symbol')['Fine Amount'].sum()
    elif calculation_method == 'Average Fines':
        if 'Fine Amount' not in grouped_df.columns:
            raise ValueError("'Fine Amount' column required for Average Fines method")
        scores = grouped_df.groupby('Matched Constituent Symbol')['Fine Amount'].mean()
    elif calculation_method == 'Max Fine':
        if 'Fine Amount' not in grouped_df.columns:
            raise ValueError("'Fine Amount' column required for Max Fine method")
        scores = grouped_df.groupby('Matched Constituent Symbol')['Fine Amount'].max()
    else:
        scores = grouped_df.groupby('Matched Constituent Symbol').size() * score_multiplier
    
    return scores.to_json()

# Initialize session state with performance optimizations
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'constituents_df': None,
        'constituents_file_hash': None,
        'enforcement_raw_dfs': {},
        'enforcement_file_hashes': {},
        'matched_enforcement_df': None,
        'final_ranked_df': None,
        'calculation_method': 'Count Enforcements',
        'score_multiplier': 1.0,
        'fine_amount_column': None,
        'match_threshold': 75,
        'data_loaded': False,
        'matching_complete': False,
        'index_calculated': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Custom CSS (optimized)
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stButton>button { 
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white; font-weight: bold; border: none;
        padding: 0.5rem 1rem; border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); }
    .metric-container { 
        background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; 
        margin: 0.5rem 0;
    }
    h1 { color: #1e3a8a; text-align: center; }
    .success-banner { 
        background: #10b981; color: white; padding: 1rem; 
        border-radius: 0.5rem; text-align: center; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Regulatory Enforcement Index Portal")
st.markdown("**Optimized for Speed** - Upload your data and generate enforcement indices instantly")

# Progress indicator
progress_col1, progress_col2, progress_col3 = st.columns(3)
with progress_col1:
    st.metric("Data Loaded", "✅" if st.session_state.data_loaded else "⏳", 
              delta="Ready" if st.session_state.data_loaded else "Pending")
with progress_col2:
    st.metric("Matching Complete", "✅" if st.session_state.matching_complete else "⏳",
              delta="Ready" if st.session_state.matching_complete else "Pending")
with progress_col3:
    st.metric("Index Calculated", "✅" if st.session_state.index_calculated else "⏳",
              delta="Ready" if st.session_state.index_calculated else "Pending")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🚀 Load Data", "⚙️ Configure & Match", "📊 Generate & Download"])

with tab1:
    st.header("Load Your Data Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Index Constituents (CSV)")
        uploaded_constituents = st.file_uploader(
            "Upload constituents file (must have 'Name' column)",
            type=["csv"], key="constituents"
        )
        
        if uploaded_constituents:
            file_hash = hash(uploaded_constituents.read())
            uploaded_constituents.seek(0)  # Reset file pointer
            
            # Only reload if file changed
            if st.session_state.constituents_file_hash != file_hash:
                try:
                    df = load_csv_file(uploaded_constituents.read(), uploaded_constituents.name)
                    
                    if 'Name' not in df.columns:
                        st.error("❌ Missing 'Name' column. Required columns: Name, Symbol, Country, Sector")
                        st.session_state.constituents_df = None
                        st.session_state.data_loaded = False
                    else:
                        st.session_state.constituents_df = df
                        st.session_state.constituents_file_hash = file_hash
                        st.success(f"✅ Loaded {len(df):,} constituents")
                        
                        # Show preview with key stats
                        st.metric("Companies", len(df))
                        st.dataframe(df.head(), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
    
    with col2:
        st.subheader("Enforcement Data (CSV)")
        uploaded_enforcement = st.file_uploader(
            "Upload enforcement files (can be multiple)",
            type=["csv"], accept_multiple_files=True, key="enforcement"
        )
        
        if uploaded_enforcement:
            new_files = {}
            total_records = 0
            
            for file in uploaded_enforcement:
                file_hash = hash(file.read())
                file.seek(0)
                
                if st.session_state.enforcement_file_hashes.get(file.name) != file_hash:
                    try:
                        df = load_csv_file(file.read(), file.name)
                        new_files[file.name] = df
                        st.session_state.enforcement_file_hashes[file.name] = file_hash
                        total_records += len(df)
                        st.success(f"✅ {file.name}: {len(df):,} records")
                    except Exception as e:
                        st.error(f"❌ Error in {file.name}: {e}")
            
            if new_files:
                st.session_state.enforcement_raw_dfs.update(new_files)
                st.metric("Total Enforcement Records", f"{total_records:,}")
    
    # Auto-detect data loading completion
    if (st.session_state.constituents_df is not None and 
        st.session_state.enforcement_raw_dfs):
        if not st.session_state.data_loaded:
            st.session_state.data_loaded = True
            st.balloons()
            st.markdown('<div class="success-banner">🎉 Data Loading Complete!</div>', 
                       unsafe_allow_html=True)

with tab2:
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data in the 'Load Data' tab first")
        st.stop()
    
    st.header("Configure Matching Parameters")
    
    # Quick configuration panel
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        match_threshold = st.slider(
            "Fuzzy Match Threshold", 50, 100, 
            st.session_state.match_threshold, key="threshold"
        )
        st.session_state.match_threshold = match_threshold
    
    with config_col2:
        # Auto-detect fine columns
        all_enforcement = pd.concat(st.session_state.enforcement_raw_dfs.values(), ignore_index=True)
        fine_cols = [col for col in all_enforcement.columns 
                    if any(term in col.lower() for term in ['fine', 'penalty', 'amount', 'sanction'])]
        
        calculation_method = st.selectbox(
            "Calculation Method",
            ['Count Enforcements', 'Sum Fines', 'Average Fines', 'Max Fine'],
            index=['Count Enforcements', 'Sum Fines', 'Average Fines', 'Max Fine'].index(
                st.session_state.calculation_method)
        )
        st.session_state.calculation_method = calculation_method
    
    with config_col3:
        if calculation_method == 'Count Enforcements':
            score_multiplier = st.number_input(
                "Score Multiplier", 0.1, 10.0, 
                st.session_state.score_multiplier, 0.1
            )
            st.session_state.score_multiplier = score_multiplier
        
        if calculation_method in ['Sum Fines', 'Average Fines', 'Max Fine']:
            if fine_cols:
                fine_col = st.selectbox("Fine Amount Column", fine_cols)
                st.session_state.fine_amount_column = fine_col
            else:
                st.warning("⚠️ No fine columns detected")
    
    st.markdown("---")
    
    # One-click matching
    if st.button("🚀 Execute Fuzzy Matching", type="primary", use_container_width=True):
        with st.spinner("Performing fuzzy matching..."):
            try:
                # Prepare data
                all_enforcement = pd.concat(st.session_state.enforcement_raw_dfs.values(), ignore_index=True)
                
                # Handle fine amount column
                if (st.session_state.fine_amount_column and 
                    st.session_state.fine_amount_column in all_enforcement.columns):
                    all_enforcement = all_enforcement.copy()
                    all_enforcement.rename(columns={
                        st.session_state.fine_amount_column: 'Fine Amount'
                    }, inplace=True)
                    all_enforcement['Fine Amount'] = pd.to_numeric(
                        all_enforcement['Fine Amount'], errors='coerce'
                    ).fillna(0)
                
                # Find company column
                company_col = None
                for col in all_enforcement.columns:
                    if any(term in col.lower() for term in ['company', 'name', 'entity']):
                        company_col = col
                        break
                
                if not company_col:
                    st.error("❌ No company name column found")
                    st.stop()
                
                # Perform cached matching
                matched_json = fuzzy_match_companies_cached(
                    all_enforcement.to_json(),
                    st.session_state.constituents_df.to_json(),
                    match_threshold,
                    company_col
                )
                
                st.session_state.matched_enforcement_df = pd.read_json(StringIO(matched_json))
                st.session_state.matching_complete = True
                
                # Show results
                matched_df = st.session_state.matched_enforcement_df
                num_matched = matched_df['Matched Constituent Symbol'].notna().sum()
                total_records = len(matched_df)
                match_rate = num_matched / total_records
                
                # Results metrics
                result_col1, result_col2, result_col3 = st.columns(3)
                with result_col1:
                    st.metric("Total Records", f"{total_records:,}")
                with result_col2:
                    st.metric("Matched Records", f"{num_matched:,}")
                with result_col3:
                    st.metric("Match Rate", f"{match_rate:.1%}")
                
                st.success("✅ Matching completed!")
                st.dataframe(matched_df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Matching failed: {e}")

with tab3:
    if not st.session_state.matching_complete:
        st.warning("⚠️ Please complete matching in the 'Configure & Match' tab first")
        st.stop()
    
    st.header("Generate Index & Download Results")
    
    # One-click index calculation
    if st.button("📊 Calculate Enforcement Index", type="primary", use_container_width=True):
        with st.spinner("Calculating enforcement scores..."):
            try:
                # Filter matched records
                valid_df = st.session_state.matched_enforcement_df[
                    st.session_state.matched_enforcement_df['Matched Constituent Symbol'].notna()
                ].copy()
                
                if valid_df.empty:
                    st.error("❌ No valid matches found for index calculation")
                    st.stop()
                
                # Calculate scores using cached function
                scores_json = calculate_enforcement_score_cached(
                    valid_df.to_json(),
                    st.session_state.calculation_method,
                    st.session_state.score_multiplier
                )
                
                scores = pd.read_json(StringIO(scores_json), typ='series')
                
                # Build final dataframe
                final_df = st.session_state.constituents_df.copy()
                
                if 'Symbol' in final_df.columns:
                    final_df = final_df.set_index('Symbol')
                else:
                    # Create a simple index if Symbol column doesn't exist
                    final_df['Symbol'] = final_df.index
                    final_df = final_df.set_index('Symbol')
                
                final_df['Enforcement Score'] = scores
                final_df['Enforcement Score'] = final_df['Enforcement Score'].fillna(0)
                
                # Add summary statistics
                enforcement_counts = valid_df.groupby('Matched Constituent Symbol').size()
                final_df['Total Enforcements'] = enforcement_counts
                final_df['Total Enforcements'] = final_df['Total Enforcements'].fillna(0).astype(int)
                
                if 'Fine Amount' in valid_df.columns:
                    total_fines = valid_df.groupby('Matched Constituent Symbol')['Fine Amount'].sum()
                    final_df['Total Fines'] = total_fines
                    final_df['Total Fines'] = final_df['Total Fines'].fillna(0)
                
                # Rank and sort
                final_df = final_df.sort_values('Enforcement Score', ascending=False)
                final_df['Rank'] = range(1, len(final_df) + 1)
                final_df = final_df.reset_index()
                
                st.session_state.final_ranked_df = final_df
                st.session_state.index_calculated = True
                
                st.success("✅ Index calculation complete!")
                
                # Display results
                st.subheader("Top 10 Companies by Enforcement Score")
                top10 = final_df.head(10)[['Rank', 'Name', 'Enforcement Score', 'Total Enforcements']]
                st.dataframe(top10, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Index calculation failed: {e}")
    
    # Download section
    if st.session_state.index_calculated:
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            # Ranked index download
            ranked_csv = st.session_state.final_ranked_df.to_csv(index=False)
            st.download_button(
                "📋 Download Ranked Index",
                ranked_csv,
                "enforcement_index_ranked.csv",
                "text/csv",
                use_container_width=True
            )
        
        with download_col2:
            # Detailed results download
            detailed_csv = st.session_state.matched_enforcement_df.to_csv(index=False)
            st.download_button(
                "📄 Download Detailed Results",
                detailed_csv,
                "enforcement_details.csv",
                "text/csv",
                use_container_width=True
            )
        
        # ZIP download
        if st.button("📦 Download Complete Package (ZIP)", use_container_width=True):
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("ranked_index.csv", ranked_csv)
                zf.writestr("detailed_results.csv", detailed_csv)
                zf.writestr("original_constituents.csv", 
                           st.session_state.constituents_df.to_csv(index=False))
            
            st.download_button(
                "Download ZIP Package",
                zip_buffer.getvalue(),
                "enforcement_index_package.zip",
                "application/zip"
            )

# Sidebar with quick stats
if st.session_state.data_loaded:
    with st.sidebar:
        st.markdown("### 📊 Quick Stats")
        if st.session_state.constituents_df is not None:
            st.metric("Constituents", len(st.session_state.constituents_df))
        
        if st.session_state.enforcement_raw_dfs:
            total_enforcement_records = sum(len(df) for df in st.session_state.enforcement_raw_dfs.values())
            st.metric("Enforcement Records", f"{total_enforcement_records:,}")
        
        if st.session_state.matching_complete:
            matched_count = st.session_state.matched_enforcement_df['Matched Constituent Symbol'].notna().sum()
            st.metric("Matched Records", f"{matched_count:,}")
        
        st.markdown("---")
        st.markdown("*Optimized for blazing fast performance* ⚡")
