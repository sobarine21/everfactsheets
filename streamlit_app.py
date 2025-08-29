import streamlit as st
import pandas as pd
from fuzzywuzzy import process, fuzz
import numpy as np
from io import StringIO
import zipfile
import base64

# --- Helper Functions (moved from utils.py) ---

def fuzzy_match_companies(enforcement_df, constituents_df, match_threshold=75):
    """
    Performs fuzzy matching between company names in enforcement data and index constituents.

    Args:
        enforcement_df (pd.DataFrame): DataFrame containing enforcement data.
        constituents_df (pd.DataFrame): DataFrame containing index constituents.
        match_threshold (int): Minimum fuzzy matching score to consider a match.

    Returns:
        pd.DataFrame: Enforcement DataFrame with matched constituent info, fuzzy scores, and best matches.
    """
    if 'Name' not in constituents_df.columns:
        raise ValueError("Constituents file must have a 'Name' column.")
    
    # Try to infer company name column in enforcement_df
    enforcement_company_col = None
    potential_cols = [col for col in enforcement_df.columns if 'company' in col.lower() or 'name' in col.lower()]
    if 'Company Name' in enforcement_df.columns:
        enforcement_company_col = 'Company Name'
    elif 'Company' in enforcement_df.columns:
        enforcement_company_col = 'Company'
    elif potential_cols:
        enforcement_company_col = potential_cols[0]
        st.info(f"Using '{enforcement_company_col}' as the company name column for enforcement data.")
    else:
        raise ValueError("Enforcement file must have a 'Company Name', 'Company' or similar column.")
    
    # Rename for internal consistency if necessary
    if enforcement_company_col != 'Company Name':
        enforcement_df.rename(columns={enforcement_company_col: 'Company Name'}, inplace=True)
            
    # Create a list of constituent names for matching
    constituent_names = constituents_df['Name'].tolist()

    # Apply fuzzy matching
    results = []
    for _, row in enforcement_df.iterrows():
        company_name_to_match = str(row['Company Name'])
        
        if pd.isna(company_name_to_match) or company_name_to_match.strip() == "":
            results.append({
                'Best Match Name': None,
                'Match Score': 0,
                'Matched Constituent Symbol': None,
                'Matched Constituent Country': None,
                'Matched Constituent Sector': None
            })
            continue
            
        matches = process.extractOne(company_name_to_match, constituent_names, scorer=fuzz.token_sort_ratio)
        
        if matches and matches[1] >= match_threshold:
            matched_name = matches[0]
            score = matches[1]
            # Find the full constituent row
            matched_constituent_row = constituents_df[constituents_df['Name'] == matched_name].iloc[0]
            results.append({
                'Best Match Name': matched_name,
                'Match Score': score,
                'Matched Constituent Symbol': matched_constituent_row['Symbol'],
                'Matched Constituent Country': matched_constituent_row['Country'],
                'Matched Constituent Sector': matched_constituent_row['Sector']
            })
        else:
            results.append({
                'Best Match Name': None,
                'Match Score': 0,
                'Matched Constituent Symbol': None,
                'Matched Constituent Country': None,
                'Matched Constituent Sector': None
            })

    results_df = pd.DataFrame(results)
    return pd.concat([enforcement_df.reset_index(drop=True), results_df], axis=1)

def calculate_enforcement_score(grouped_df, calculation_method, score_multiplier_per_enforcement=1):
    """
    Calculates the enforcement score for each company based on the chosen method.

    Args:
        grouped_df (pd.DataFrame): DataFrame grouped by 'Matched Constituent Symbol'
                                   containing enforcement counts/values.
        calculation_method (str): Method to use ('Count Enforcements', 'Sum Fines', 'Average Fines', etc.)
                                  Assumes 'Fine Amount' column exists if 'Sum Fines' or 'Average Fines' is chosen.
        score_multiplier_per_enforcement (float): Multiplier for each enforcement count.

    Returns:
        pd.Series: A Series of enforcement scores, indexed by 'Matched Constituent Symbol'.
    """
    scores = {}
    if calculation_method == 'Count Enforcements':
        scores = grouped_df['Company Name'].count() * score_multiplier_per_enforcement
    elif calculation_method == 'Sum Fines':
        if 'Fine Amount' not in grouped_df.columns:
            st.error("Error: 'Fine Amount' column not found in enforcement data for 'Sum Fines' method.")
            return pd.Series()
        scores = grouped_df['Fine Amount'].sum()
    elif calculation_method == 'Average Fines':
        if 'Fine Amount' not in grouped_df.columns:
            st.error("Error: 'Fine Amount' column not found in enforcement data for 'Average Fines' method.")
            return pd.Series()
        scores = grouped_df['Fine Amount'].mean()
    elif calculation_method == 'Max Fine':
        if 'Fine Amount' not in grouped_df.columns:
            st.error("Error: 'Fine Amount' column not found in enforcement data for 'Max Fine' method.")
            return pd.Series()
        scores = grouped_df['Fine Amount'].max()
    else:
        # Default to count if method is unknown or not explicitly handled
        scores = grouped_df['Company Name'].count() * score_multiplier_per_enforcement
    
    return scores

# --- Streamlit Application ---

st.set_page_config(layout="wide", page_title="Enforcement Index Portal")

# Custom CSS for a bit more styling
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stFileUploader label {
        font-weight: bold;
        color: #31333F;
        font-size: 1.1em;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    h1 {
        color: #007bff;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    h2 {
        color: #0056b3;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    h3 {
        color: #0056b3;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stAlert {
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .dataframe {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Regulatory Enforcement Index Portal")
st.markdown("Upload your index constituents and enforcement data to create a ranked list of companies based on regulatory enforcements.")

# Initialize session state for storing data
if 'constituents_df' not in st.session_state:
    st.session_state.constituents_df = None
if 'enforcement_raw_dfs' not in st.session_state:
    st.session_state.enforcement_raw_dfs = {} # Stores dict of {filename: df}
if 'matched_enforcement_df' not in st.session_state:
    st.session_state.matched_enforcement_df = None
if 'final_ranked_df' not in st.session_state:
    st.session_state.final_ranked_df = None
if 'calculation_method' not in st.session_state:
    st.session_state.calculation_method = 'Count Enforcements'
if 'score_multiplier' not in st.session_state:
    st.session_state.score_multiplier = 1.0
if 'fine_amount_column' not in st.session_state:
    st.session_state.fine_amount_column = None

tab1, tab2, tab3, tab4 = st.tabs(["1. Upload Data", "2. Review & Configure", "3. Generate Index", "4. Download Results"])

with tab1:
    st.header("1. Upload Your Data Files")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Index Constituents (CSV)")
        st.info("This file defines the universe of companies for your index. It **must** contain a column named 'Name'.")
        uploaded_constituents_file = st.file_uploader(
            "Upload Index Constituents CSV",
            type=["csv"],
            key="constituents_upload"
        )

        if uploaded_constituents_file is not None:
            try:
                constituents_df = pd.read_csv(uploaded_constituents_file)
                if 'Name' not in constituents_df.columns:
                    st.error("Error: The constituents file must contain a column named 'Name'. Please check your CSV.")
                    st.session_state.constituents_df = None
                else:
                    st.session_state.constituents_df = constituents_df
                    st.success(f"Index constituents loaded: {len(constituents_df)} companies.")
                    st.dataframe(constituents_df.head())
            except Exception as e:
                st.error(f"Error loading constituents file: {e}")

    with col2:
        st.subheader("Upload Bulk Enforcement Data (CSV(s))")
        st.info("Upload one or more CSV files containing enforcement records. These files should ideally have a 'Company Name' or similar column for matching.")
        uploaded_enforcement_files = st.file_uploader(
            "Upload Enforcement CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key="enforcement_upload"
        )

        if uploaded_enforcement_files:
            new_enforcement_dfs = {}
            for file in uploaded_enforcement_files:
                try:
                    df = pd.read_csv(file)
                    new_enforcement_dfs[file.name] = df
                    st.success(f"Loaded enforcement file: {file.name} ({len(df)} records)")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error loading enforcement file {file.name}: {e}")
            st.session_state.enforcement_raw_dfs = new_enforcement_dfs
            st.info(f"Total enforcement files loaded: {len(st.session_state.enforcement_raw_dfs)}")
        elif not uploaded_enforcement_files and st.session_state.enforcement_raw_dfs:
            st.info("No new enforcement files uploaded. Using previously loaded files (if any).")


with tab2:
    st.header("2. Review Data & Configure Matching/Calculation")

    if st.session_state.constituents_df is None:
        st.warning("Please upload Index Constituents in '1. Upload Data' tab.")
    if not st.session_state.enforcement_raw_dfs:
        st.warning("Please upload Enforcement Data in '1. Upload Data' tab.")

    if st.session_state.constituents_df is not None and st.session_state.enforcement_raw_dfs:
        st.subheader("Fuzzy Matching Configuration")
        match_threshold = st.slider(
            "Fuzzy Match Threshold (%)",
            min_value=50,
            max_value=100,
            value=75,
            key="match_threshold_slider",
            help="Companies will only be matched if their name similarity score is above this threshold (using Token Sort Ratio)."
        )

        st.subheader("Enforcement Score Calculation")
        st.info("Select how to calculate the enforcement score for each company. If 'Sum Fines', 'Average Fines', or 'Max Fine' is chosen, ensure your enforcement data has a 'Fine Amount' or similar column.")
        
        # Try to infer fine amount column from first enforcement DF
        potential_fine_cols = []
        if st.session_state.enforcement_raw_dfs:
            # Concatenate all enforcement data temporarily to check columns
            temp_all_enforcement_data = pd.concat(st.session_state.enforcement_raw_dfs.values(), ignore_index=True)
            potential_fine_cols = [col for col in temp_all_enforcement_data.columns if 'fine' in col.lower() or 'penalty' in col.lower() or 'amount' in col.lower()]
            
        calculation_method = st.selectbox(
            "Select Enforcement Score Calculation Method",
            options=['Count Enforcements', 'Sum Fines', 'Average Fines', 'Max Fine', 'Custom Weighted Score (Future)'],
            index=0,
            key="calculation_method_select",
            help="Choose how the enforcement score is determined. 'Custom Weighted Score' is a placeholder for future enhancements."
        )
        st.session_state.calculation_method = calculation_method # Store for Tab 3

        score_multiplier = 1.0
        if calculation_method == 'Count Enforcements':
            score_multiplier = st.number_input(
                "Score Multiplier per Enforcement",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.score_multiplier, # Initialize with session state
                step=0.1,
                key="score_multiplier_input",
                help="Multiply the count of enforcements by this value to get the score."
            )
            st.session_state.score_multiplier = score_multiplier # Store for Tab 3
        else:
            st.session_state.score_multiplier = 1.0 # Reset if not counting

        if calculation_method in ['Sum Fines', 'Average Fines', 'Max Fine']:
            if potential_fine_cols:
                fine_col_options = ['-- Select --'] + potential_fine_cols
                try:
                    default_index = fine_col_options.index(st.session_state.fine_amount_column) if st.session_state.fine_amount_column in fine_col_options else 0
                except ValueError:
                    default_index = 0

                fine_col = st.selectbox(
                    "Select Fine Amount Column (from enforcement data)",
                    options=fine_col_options,
                    index=default_index,
                    key="fine_col_selector",
                    help="Choose the column in your enforcement files that contains monetary fine amounts."
                )
                if fine_col == '-- Select --':
                    fine_col = None
            else:
                fine_col = st.text_input(
                    "Enter Fine Amount Column Name (e.g., 'Fine Amount')",
                    value=st.session_state.fine_amount_column if st.session_state.fine_amount_column else '',
                    key="fine_col_input",
                    help="Manually enter the column name if not automatically detected."
                )
            
            st.session_state.fine_amount_column = fine_col # Store for Tab 3
            if not fine_col:
                 st.warning("Please specify the 'Fine Amount' column or select a different calculation method.")
        else: # Reset fine amount column if not relevant for current calculation method
            st.session_state.fine_amount_column = None

        st.markdown("---")
        st.subheader("Preview Matched Data (After Fuzzy Matching)")

        if st.button("Perform Matching & Preview Data", key="perform_matching"):
            if st.session_state.constituents_df is None:
                st.error("Please upload Index Constituents in '1. Upload Data' tab first.")
            elif not st.session_state.enforcement_raw_dfs:
                st.error("Please upload Enforcement Data in '1. Upload Data' tab first.")
            else:
                all_enforcement_data = pd.concat(st.session_state.enforcement_raw_dfs.values(), ignore_index=True)
                
                # Check if a fine column is specified and exists before renaming
                if st.session_state.fine_amount_column and st.session_state.fine_amount_column in all_enforcement_data.columns:
                    # Make a copy to avoid SettingWithCopyWarning later and ensure original is not modified unexpectedly
                    all_enforcement_data_copy = all_enforcement_data.copy()
                    all_enforcement_data_copy.rename(columns={st.session_state.fine_amount_column: 'Fine Amount'}, inplace=True)
                    # Ensure 'Fine Amount' is numeric, coercing errors to NaN
                    all_enforcement_data_copy['Fine Amount'] = pd.to_numeric(all_enforcement_data_copy['Fine Amount'], errors='coerce')
                    all_enforcement_data_copy['Fine Amount'].fillna(0, inplace=True) # Treat NaNs as 0 for sum/avg/max operations
                    df_for_matching = all_enforcement_data_copy
                else:
                    df_for_matching = all_enforcement_data.copy() # Use a copy anyway

                try:
                    matched_df = fuzzy_match_companies(df_for_matching, st.session_state.constituents_df, match_threshold)
                    st.session_state.matched_enforcement_df = matched_df
                    st.success("Fuzzy matching completed! See preview below.")
                    st.dataframe(matched_df.head(10))

                    num_matched = matched_df['Matched Constituent Symbol'].notna().sum()
                    num_unmatched = matched_df['Matched Constituent Symbol'].isna().sum()
                    total_enforcements = len(matched_df)

                    st.info(f"Matched {num_matched} out of {total_enforcements} enforcement records ({num_matched/total_enforcements:.2%}). "
                            f"({num_unmatched} records unmatched)")

                    # Display unmatched entries
                    if num_unmatched > 0:
                        st.subheader("Unmatched Enforcement Records (Top 10 of original company names)")
                        st.dataframe(matched_df[matched_df['Matched Constituent Symbol'].isna()]['Company Name'].value_counts().reset_index().head(10).rename(columns={'index': 'Unmatched Company Name', 'Company Name': 'Count'}))

                except ValueError as ve:
                    st.error(f"Configuration Error: {ve}")
                except Exception as e:
                    st.error(f"An error occurred during fuzzy matching: {e}")
    else:
        st.warning("Please upload both constituent and enforcement data in the '1. Upload Data' tab and then click 'Perform Matching & Preview Data'.")


with tab3:
    st.header("3. Generate Enforcement Index")

    if st.session_state.matched_enforcement_df is None:
        st.warning("Please complete step 2 ('Perform Matching & Preview Data') before generating the index.")
    else:
        st.subheader("Index Calculation Summary")
        
        calculation_method = st.session_state.calculation_method
        score_multiplier = st.session_state.score_multiplier
        fine_col_name = st.session_state.fine_amount_column

        # Check if 'Fine Amount' column exists and is valid if needed by the calculation method
        fine_column_exists_and_valid = False
        if 'Fine Amount' in st.session_state.matched_enforcement_df.columns:
            # Check if it has any non-zero/non-null values
            if st.session_state.matched_enforcement_df['Fine Amount'].sum() > 0:
                fine_column_exists_and_valid = True
        
        if calculation_method in ['Sum Fines', 'Average Fines', 'Max Fine'] and not fine_column_exists_and_valid:
            st.error(f"Selected calculation method '{calculation_method}' requires a valid 'Fine Amount' column with numeric values. "
                     f"Please ensure it's correctly specified and available in the matched data (check Tab 2). Current column: {fine_col_name}. "
                     f"If your data had a fine column, please check its name and ensure it contains numeric data.")
        elif calculation_method == 'Custom Weighted Score (Future)':
            st.warning("The 'Custom Weighted Score' method is a placeholder and not yet implemented. Please choose another method.")
        else:
            st.info(f"Using **'{calculation_method}'** method.")
            if calculation_method == 'Count Enforcements':
                st.info(f"Each enforcement counts as **{score_multiplier:.1f}** points.")
            elif fine_col_name:
                st.info(f"Using fine amount data from column: **'{fine_col_name}'**.")

            if st.button("Calculate Enforcement Index & Rank Companies", key="calculate_index"):
                # Filter out unmatched records for index calculation
                valid_enforcements_df = st.session_state.matched_enforcement_df[
                    st.session_state.matched_enforcement_df['Matched Constituent Symbol'].notna()
                ].copy()

                if valid_enforcements_df.empty:
                    st.warning("No matched enforcement records to calculate the index. Please review your data and matching threshold in Tab 2.")
                    st.session_state.final_ranked_df = pd.DataFrame() # Clear previous results
                else:
                    try:
                        # Group by the matched constituent symbol to calculate scores
                        grouped_enforcements = valid_enforcements_df.groupby('Matched Constituent Symbol')
                        
                        enforcement_scores = calculate_enforcement_score(
                            grouped_enforcements,
                            calculation_method,
                            score_multiplier
                        )

                        # Merge scores back to the original constituents DataFrame
                        final_df = st.session_state.constituents_df.copy()
                        final_df = final_df.set_index('Symbol') # Set Symbol as index for easy merging
                        final_df['Enforcement Score'] = enforcement_scores
                        final_df['Enforcement Score'].fillna(0, inplace=True) # Companies with no enforcements get score 0

                        # Add other details from matched enforcement data (e.g., total enforcements, total fines)
                        total_enforcements_per_company = grouped_enforcements['Company Name'].count()
                        
                        # Only add Total Fines if 'Fine Amount' column is actually present and used
                        if 'Fine Amount' in valid_enforcements_df.columns:
                            total_fines_per_company = grouped_enforcements['Fine Amount'].sum()
                            final_df['Total Fines'] = total_fines_per_company
                            final_df['Total Fines'].fillna(0, inplace=True)
                        else:
                            final_df['Total Fines'] = 0 # Default to 0 if no fine data

                        final_df['Total Enforcements'] = total_enforcements_per_company
                        final_df['Total Enforcements'].fillna(0, inplace=True)
                        
                        # Sort by enforcement score (descending) and assign rank
                        final_df = final_df.sort_values(by='Enforcement Score', ascending=False)
                        final_df['Rank'] = range(1, len(final_df) + 1)
                        final_df = final_df.reset_index() # Bring Symbol back as a column

                        st.session_state.final_ranked_df = final_df
                        st.success("Enforcement Index calculated successfully!")
                        
                        st.subheader("Top 10 Ranked Companies by Enforcement Score")
                        st.dataframe(final_df.head(10))
                        
                        st.subheader("Full Ranked List")
                        st.dataframe(final_df)


                    except Exception as e:
                        st.error(f"An error occurred during index calculation: {e}")

with tab4:
    st.header("4. Download Results")

    if st.session_state.final_ranked_df is None or st.session_state.final_ranked_df.empty:
        st.warning("Please generate the index in '3. Generate Index' tab first.")
    else:
        st.subheader("Download Ranked Companies Data")
        st.info("This CSV contains the final ranked list of companies with their calculated enforcement scores.")
        csv_output_ranked = st.session_state.final_ranked_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Ranked Index CSV",
            data=csv_output_ranked,
            file_name="ranked_enforcement_index.csv",
            mime="text/csv",
            key="download_ranked_csv"
        )
        
        st.subheader("Download Matched Enforcement Data (Detailed)")
        st.info("This file contains all individual enforcement records with their matching details, including match scores.")
        matched_enforcement_csv_output = st.session_state.matched_enforcement_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Matched Enforcement Details CSV",
            data=matched_enforcement_csv_output,
            file_name="matched_enforcement_details.csv",
            mime="text/csv",
            key="download_matched_csv"
        )
        
        # Option to download all results as a single ZIP file
        st.subheader("Download All Results (ZIP)")
        st.info("Download a ZIP file containing the ranked index, detailed matched enforcement data, and your original constituent list.")
        if st.button("Generate ZIP for Download"):
            zip_buffer = StringIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add ranked index
                ranked_csv = st.session_state.final_ranked_df.to_csv(index=False)
                zip_file.writestr("ranked_enforcement_index.csv", ranked_csv)

                # Add matched enforcement details
                if st.session_state.matched_enforcement_df is not None:
                    matched_csv = st.session_state.matched_enforcement_df.to_csv(index=False)
                    zip_file.writestr("matched_enforcement_details.csv", matched_csv)
                
                # Add original constituents (optional, but good for record-keeping)
                if st.session_state.constituents_df is not None:
                    constituents_csv = st.session_state.constituents_df.to_csv(index=False)
                    zip_file.writestr("original_index_constituents.csv", constituents_csv)

            zip_buffer.seek(0)
            st.download_button(
                label="Download All Results as ZIP",
                data=zip_buffer.getvalue().encode('utf-8'),
                file_name="enforcement_index_results.zip",
                mime="application/zip",
                key="download_all_zip"
            )

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Your Name/Company")
