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
                                match_threshold: int = 75) -> str:
    """Cached fuzzy matching with serialized DataFrames - searches all enforcement columns"""
    # Deserialize DataFrames
    enforcement_df = pd.read_json(StringIO(enforcement_data))
    constituents_df = pd.read_json(StringIO(constituents_data))
    
    # Ensure required columns exist in constituents
    required_cols = ['Name', 'Symbol', 'Sector']
    missing_cols = [col for col in required_cols if col not in constituents_df.columns]
    if missing_cols:
        raise ValueError(f"Constituents file missing required columns: {missing_cols}")
    
    # Prepare constituent names list for faster matching - ONLY use Name column
    constituent_names = constituents_df['Name'].dropna().astype(str).str.strip().tolist()
    
    # Create a mapping for faster lookups
    name_to_info = {}
    for _, row in constituents_df.iterrows():
        name = str(row['Name']).strip()
        name_to_info[name] = {
            'Symbol': row.get('Symbol', ''),
            'Sector': row.get('Sector', ''),
            'Country': row.get('Country', '')  # Keep Country if available
        }
    
    # Search all enforcement columns for company matches
    results = []
    batch_size = 100
    total_rows = len(enforcement_df)
    
    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        batch_df = enforcement_df.iloc[i:batch_end]
        
        batch_results = []
        for _, row in batch_df.iterrows():
            best_match = None
            best_score = 0
            best_source_column = None
            best_source_value = None
            
            # Search through ALL columns in the enforcement row
            for column_name, cell_value in row.items():
                if pd.isna(cell_value):
                    continue
                    
                cell_str = str(cell_value).strip()
                if not cell_str or len(cell_str) < 2:  # Skip very short strings
                    continue
                
                # Perform fuzzy matching on this cell value
                matches = process.extractOne(cell_str, constituent_names, 
                                           scorer=fuzz.token_sort_ratio, score_cutoff=match_threshold)
                
                if matches and matches[1] > best_score:
                    best_match = matches[0]
                    best_score = matches[1]
                    best_source_column = column_name
                    best_source_value = cell_str
            
            # Store the best match found across all columns
            if best_match:
                matched_info = name_to_info[best_match]
                batch_results.append({
                    'Best Match Name': best_match,
                    'Match Score': best_score,
                    'Source Column': best_source_column,
                    'Source Value': best_source_value,
                    'Matched Constituent Symbol': matched_info['Symbol'],
                    'Matched Constituent Sector': matched_info['Sector'],
                    'Matched Constituent Country': matched_info['Country']
                })
            else:
                batch_results.append({
                    'Best Match Name': None,
                    'Match Score': 0,
                    'Source Column': None,
                    'Source Value': None,
                    'Matched Constituent Symbol': None,
                    'Matched Constituent Sector': None,
                    'Matched Constituent Country': None
                })
        
        results.extend(batch_results)
    
    # Combine results
    results_df = pd.DataFrame(results)
    final_df = pd.concat([enforcement_df.reset_index(drop=True), results_df], axis=1)
    
    # Return serialized result
    return final_df.to_json()

@st.cache_data
def calculate_enforcement_score_cached(grouped_data: str, calculation_method: str, 
                                     score_multiplier: float = 1.0, fine_column: str = None) -> str:
    """Cached enforcement score calculation"""
    # Deserialize grouped data
    grouped_df = pd.read_json(StringIO(grouped_data))
    
    if calculation_method == 'Count Enforcements':
        scores = grouped_df.groupby('Matched Constituent Symbol').size() * score_multiplier
    elif calculation_method in ['Sum Fines', 'Average Fines', 'Max Fine']:
        if not fine_column or fine_column not in grouped_df.columns:
            raise ValueError(f"Fine column '{fine_column}' not found for {calculation_method} method")
        
        if calculation_method == 'Sum Fines':
            scores = grouped_df.groupby('Matched Constituent Symbol')[fine_column].sum()
        elif calculation_method == 'Average Fines':
            scores = grouped_df.groupby('Matched Constituent Symbol')[fine_column].mean()
        else:  # Max Fine
            scores = grouped_df.groupby('Matched Constituent Symbol')[fine_column].max()
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
    .info-box {
        background: #e0f2fe; padding: 1rem; border-radius: 0.5rem;
        border-left: 4px solid #0277bd; margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd; padding: 1rem; border-radius: 0.5rem;
        border-left: 4px solid #ffc107; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Regulatory Enforcement Index Portal")
st.markdown("**Optimized for Speed** - Upload your data and generate enforcement indices instantly")

# Important matching info
st.markdown("""
<div class="info-box">
<strong>🔍 Universal Matching Logic:</strong> The system searches <strong>ALL columns</strong> in your enforcement files for company matches against the <strong>Name</strong> column from constituents. 
No need to specify company columns - it finds them automatically! Output includes <strong>Symbol</strong>, <strong>Name</strong>, and <strong>Sector</strong>.
</div>
""", unsafe_allow_html=True)

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
        st.info("**Required columns:** Name, Symbol, Sector. Optional: Country")
        uploaded_constituents = st.file_uploader(
            "Upload constituents file",
            type=["csv"], key="constituents"
        )
        
        if uploaded_constituents:
            file_hash = hash(uploaded_constituents.read())
            uploaded_constituents.seek(0)  # Reset file pointer
            
            # Only reload if file changed
            if st.session_state.constituents_file_hash != file_hash:
                try:
                    df = load_csv_file(uploaded_constituents.read(), uploaded_constituents.name)
                    
                    # Check required columns
                    required_cols = ['Name', 'Symbol', 'Sector']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                        st.info("Required: Name, Symbol, Sector. Optional: Country")
                        st.session_state.constituents_df = None
                        st.session_state.data_loaded = False
                    else:
                        st.session_state.constituents_df = df
                        st.session_state.constituents_file_hash = file_hash
                        st.success(f"✅ Loaded {len(df):,} constituents")
                        
                        # Show preview with key columns
                        display_cols = ['Name', 'Symbol', 'Sector']
                        if 'Country' in df.columns:
                            display_cols.append('Country')
                        
                        st.metric("Companies", len(df))
                        st.dataframe(df[display_cols].head(), use_container_width=True)
                        
                        # Show matching info
                        st.markdown("""
                        <div class="warning-box">
                        <strong>🎯 Matching Target:</strong> System will search for these company names across ALL columns in enforcement files.
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
    
    with col2:
        st.subheader("Enforcement Data (CSV)")
        st.info("**Any format accepted!** System searches ALL columns automatically for company matches.")
        uploaded_enforcement = st.file_uploader(
            "Upload enforcement files (can be multiple)",
            type=["csv"], accept_multiple_files=True, key="enforcement"
        )
        
        if uploaded_enforcement:
            new_files = {}
            total_records = 0
            all_columns = set()
            
            for file in uploaded_enforcement:
                file_hash = hash(file.read())
                file.seek(0)
                
                if st.session_state.enforcement_file_hashes.get(file.name) != file_hash:
                    try:
                        df = load_csv_file(file.read(), file.name)
                        new_files[file.name] = df
                        st.session_state.enforcement_file_hashes[file.name] = file_hash
                        total_records += len(df)
                        all_columns.update(df.columns.tolist())
                        st.success(f"✅ {file.name}: {len(df):,} records, {len(df.columns)} columns")
                        
                    except Exception as e:
                        st.error(f"❌ Error in {file.name}: {e}")
            
            if new_files:
                st.session_state.enforcement_raw_dfs.update(new_files)
                st.metric("Total Enforcement Records", f"{total_records:,}")
                st.metric("Total Unique Columns", len(all_columns))
                
                # Show sample columns
                sample_cols = list(all_columns)[:10]
                if len(all_columns) > 10:
                    sample_cols.append(f"... and {len(all_columns)-10} more")
                st.info(f"**Columns to search:** {', '.join(sample_cols)}")
    
    # Auto-detect data loading completion
    if (st.session_state.constituents_df is not None and 
        st.session_state.enforcement_raw_dfs):
        if not st.session_state.data_loaded:
            st.session_state.data_loaded = True
            st.balloons()
            st.markdown('<div class="success-banner">🎉 Data Loading Complete! Ready for universal matching.</div>', 
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
            st.session_state.match_threshold, key="threshold",
            help="Higher values = stricter matching. System searches ALL columns automatically."
        )
        st.session_state.match_threshold = match_threshold
    
    with config_col2:
        # Auto-detect fine columns from all enforcement data
        all_enforcement = pd.concat(st.session_state.enforcement_raw_dfs.values(), ignore_index=True)
        fine_cols = [col for col in all_enforcement.columns 
                    if any(term in col.lower() for term in ['fine', 'penalty', 'amount', 'sanction', 'settlement', 'monetary', 'payment', 'fee'])]
        
        calculation_method = st.selectbox(
            "Calculation Method",
            ['Count Enforcements', 'Sum Fines', 'Average Fines', 'Max Fine'],
            index=['Count Enforcements', 'Sum Fines', 'Average Fines', 'Max Fine'].index(
                st.session_state.calculation_method),
            help="How to calculate enforcement scores"
        )
        st.session_state.calculation_method = calculation_method
    
    with config_col3:
        if calculation_method == 'Count Enforcements':
            score_multiplier = st.number_input(
                "Score Multiplier", 0.1, 10.0, 
                st.session_state.score_multiplier, 0.1,
                help="Multiply enforcement count by this value"
            )
            st.session_state.score_multiplier = score_multiplier
        
        if calculation_method in ['Sum Fines', 'Average Fines', 'Max Fine']:
            if fine_cols:
                fine_col = st.selectbox("Fine Amount Column", fine_cols,
                                       help="Select column containing monetary amounts")
                st.session_state.fine_amount_column = fine_col
            else:
                st.warning("⚠️ No monetary columns detected. Switch to 'Count Enforcements' method.")
                st.session_state.fine_amount_column = None
    
    st.markdown("---")
    
    # Preview data structures
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.constituents_df is not None:
            st.subheader("📋 Constituent Names (Search Targets)")
            display_cols = ['Name', 'Symbol', 'Sector']
            if 'Country' in st.session_state.constituents_df.columns:
                display_cols.append('Country')
            st.dataframe(st.session_state.constituents_df[display_cols].head(), use_container_width=True)
    
    with col2:
        if st.session_state.enforcement_raw_dfs:
            st.subheader("🔍 Enforcement Data (All Columns Searched)")
            sample_df = next(iter(st.session_state.enforcement_raw_dfs.values()))
            st.dataframe(sample_df.head(3), use_container_width=True)
            st.caption(f"Showing sample from first file. All {len(sample_df.columns)} columns will be searched.")
    
    # One-click matching
    if st.button("🚀 Execute Universal Fuzzy Matching", type="primary", use_container_width=True):
        with st.spinner("Searching ALL enforcement columns for company matches..."):
            try:
                # Prepare data
                all_enforcement = pd.concat(st.session_state.enforcement_raw_dfs.values(), ignore_index=True)
                
                # Handle fine amount column if specified
                if (st.session_state.fine_amount_column and 
                    st.session_state.fine_amount_column in all_enforcement.columns):
                    all_enforcement = all_enforcement.copy()
                    # Keep original column and create a normalized 'Fine Amount' column
                    all_enforcement['Fine Amount'] = pd.to_numeric(
                        all_enforcement[st.session_state.fine_amount_column], errors='coerce'
                    ).fillna(0)
                
                # Perform universal matching (searches all columns automatically)
                matched_json = fuzzy_match_companies_cached(
                    all_enforcement.to_json(),
                    st.session_state.constituents_df.to_json(),
                    match_threshold
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
                
                st.success("✅ Universal matching completed!")
                
                # Show matched results with source information
                if num_matched > 0:
                    st.subheader("✅ Sample Matched Results (with Source Columns)")
                    match_preview = matched_df[matched_df['Matched Constituent Symbol'].notna()][
                        ['Source Column', 'Source Value', 'Best Match Name', 'Match Score', 
                         'Matched Constituent Symbol', 'Matched Constituent Sector']
                    ].head(10)
                    st.dataframe(match_preview, use_container_width=True)
                    
                    # Show which columns provided matches
                    source_columns = matched_df[matched_df['Matched Constituent Symbol'].notna()]['Source Column'].value_counts()
                    st.subheader("📊 Matches by Source Column")
                    st.dataframe(source_columns.reset_index(), use_container_width=True)
                
                # Show unmatched for review
                if num_matched < total_records:
                    st.subheader("❌ Sample Unmatched Records")
                    unmatched = matched_df[matched_df['Matched Constituent Symbol'].isna()]
                    # Show a few sample unmatched records with their data
                    st.dataframe(unmatched.head(5), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Universal matching failed: {e}")

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
                    st.session_state.score_multiplier,
                    st.session_state.fine_amount_column
                )
                
                scores = pd.read_json(StringIO(scores_json), typ='series')
                
                # Build final dataframe with Symbol, Name, Sector
                final_df = st.session_state.constituents_df.copy()
                
                # Ensure we have Symbol column for merging
                if 'Symbol' not in final_df.columns:
                    st.error("❌ Symbol column required in constituents data")
                    st.stop()
                
                # Set Symbol as index for merging
                final_df = final_df.set_index('Symbol')
                
                # Add enforcement scores
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
                
                # Reorder columns to show Symbol, Name, Sector prominently
                core_cols = ['Rank', 'Symbol', 'Name', 'Sector', 'Enforcement Score', 'Total Enforcements']
                if 'Total Fines' in final_df.columns:
                    core_cols.append('Total Fines')
                if 'Country' in final_df.columns:
                    core_cols.append('Country')
                
                # Add any remaining columns
                remaining_cols = [col for col in final_df.columns if col not in core_cols]
                final_df = final_df[core_cols + remaining_cols]
                
                st.session_state.final_ranked_df = final_df
                st.session_state.index_calculated = True
                
                st.success("✅ Index calculation complete!")
                
                # Display results with Symbol, Name, Sector
                st.subheader("🏆 Top 10 Companies by Enforcement Score")
                top10_display = final_df.head(10)[['Rank', 'Symbol', 'Name', 'Sector', 'Enforcement Score', 'Total Enforcements']]
                st.dataframe(top10_display, use_container_width=True)
                
                # Summary stats
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    companies_with_enforcements = (final_df['Total Enforcements'] > 0).sum()
                    st.metric("Companies with Enforcements", companies_with_enforcements)
                with stats_col2:
                    max_score = final_df['Enforcement Score'].max()
                    st.metric("Highest Score", f"{max_score:.2f}")
                with stats_col3:
                    total_enforcements = final_df['Total Enforcements'].sum()
                    st.metric("Total Enforcements", int(total_enforcements))
                with stats_col4:
                    avg_score = final_df[final_df['Enforcement Score'] > 0]['Enforcement Score'].mean()
                    st.metric("Average Score", f"{avg_score:.2f}" if not pd.isna(avg_score) else "0")
                
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
                "📋 Download Ranked Index (Symbol, Name, Sector)",
                ranked_csv,
                "enforcement_index_ranked.csv",
                "text/csv",
                use_container_width=True
            )
        
        with download_col2:
            # Detailed results download
            detailed_csv = st.session_state.matched_enforcement_df.to_csv(index=False)
            st.download_button(
                "📄 Download Detailed Matching Results",
                detailed_csv,
                "enforcement_matching_details.csv",
                "text/csv",
                use_container_width=True
            )
        
        # ZIP download
        if st.button("📦 Download Complete Package (ZIP)", use_container_width=True):
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("ranked_index.csv", ranked_csv)
                zf.writestr("detailed_matching_results.csv", detailed_csv)
                zf.writestr("original_constituents.csv", 
                           st.session_state.constituents_df.to_csv(index=False))
                
                # Add matching summary report
                if st.session_state.matched_enforcement_df is not None:
                    matched_df = st.session_state.matched_enforcement_df
                    source_summary = matched_df[matched_df['Matched Constituent Symbol'].notna()]['Source Column'].value_counts()
                    summary_report = f"""Universal Matching Summary Report
=====================================

Total Records Processed: {len(matched_df):,}
Successfully Matched: {matched_df['Matched Constituent Symbol'].notna().sum():,}
Match Rate: {(matched_df['Matched Constituent Symbol'].notna().sum() / len(matched_df) * 100):.1f}%

Matches by Source Column:
{source_summary.to_string()}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                    zf.writestr("matching_summary_report.txt", summary_report)
            
            st.download_button(
                "Download Complete Package with Reports",
                zip_buffer.getvalue(),
                "enforcement_index_universal_matching.zip",
                "application/zip"
            )
        
        # Preview final results
        st.subheader("📊 Complete Ranked Results Preview")
        st.dataframe(st.session_state.final_ranked_df, use_container_width=True)

# Sidebar with quick stats
if st.session_state.data_loaded:
    with st.sidebar:
        st.markdown("### 📊 Quick Stats")
        if st.session_state.constituents_df is not None:
            st.metric("Constituents", len(st.session_state.constituents_df))
            # Show sectors breakdown
            if 'Sector' in st.session_state.constituents_df.columns:
                sector_counts = st.session_state.constituents_df['Sector'].value_counts()
                st.markdown("**Top Sectors:**")
                for sector, count in sector_counts.head(5).items():
                    st.text(f"• {sector}: {count}")
        
        if st.session_state.enforcement_raw_dfs:
            total_enforcement_records = sum(len(df) for df in st.session_state.enforcement_raw_dfs.values())
            st.metric("Enforcement Records", f"{total_enforcement_records:,}")
            
            # Show total columns being searched
            all_cols = set()
            for df in st.session_state.enforcement_raw_dfs.values():
                all_cols.update(df.columns.tolist())
            st.metric("Columns Searched", len(all_cols))
        
        if st.session_state.matching_complete:
            matched_count = st.session_state.matched_enforcement_df['Matched Constituent Symbol'].notna().sum()
            st.metric("Matched Records", f"{matched_count:,}")
            
            # Show match rate
            total_count = len(st.session_state.matched_enforcement_df)
            match_rate = (matched_count / total_count) * 100
            st.metric("Match Rate", f"{match_rate:.1f}%")
