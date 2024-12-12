import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai

# Function to generate line chart for performance comparison (Fund vs Benchmark)
def generate_line_chart(data, columns, title):
    # Ensure the columns in 'y' exist in the dataframe
    columns = [col for col in columns if col in data.columns]
    
    if len(columns) < 1:
        st.warning("Not enough columns available for the requested comparison! Proceeding with available data.")
        # Use all available columns for the line chart
        columns = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
        if len(columns) < 1:
            st.error("No numeric data available for visualization.")
            return None
    
    fig = px.line(data, x=data.index, y=columns, title=title)
    return fig

# Function to generate correlation heatmap
def generate_correlation_heatmap(data, title):
    # Exclude non-numeric columns (e.g., date columns)
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # If there's no numeric data, we can't compute correlations
    if numeric_data.empty:
        st.warning("No numeric columns available for correlation analysis.")
        return None

    # Calculate correlation matrix
    corr = numeric_data.corr()

    # Create and display the heatmap
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    plt.title(title)
    st.pyplot(fig)

# Configure the API key securely from Streamlit's secrets
# Make sure to add GOOGLE_API_KEY in secrets.toml (for local) or Streamlit Cloud Secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App
st.title('Fund Visualization Dashboard with AI Summary')

# File uploader to upload the Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

# Function to generate a fund summary from metrics
def generate_fund_summary(data):
    try:
        # Extract relevant columns for the summary
        fund_name = data["Fund Details"]["Fund Name"].iloc[0] if "Fund Details" in data else "Unknown Fund"
        fund_manager = data["Fund Details"]["Fund Manager"].iloc[0] if "Fund Details" in data else "Unknown Manager"
        launch_date = data["Fund Details"]["Launch Date"].iloc[0] if "Fund Details" in data else "Unknown Date"
        total_assets = data["Fund Details"]["Total Assets"].iloc[0] if "Fund Details" in data else "Unknown Assets"
        fund_type = data["Fund Details"]["Fund Type"].iloc[0] if "Fund Details" in data else "Unknown Type"
        
        # Performance Data for Fund vs Benchmark
        performance_data = data["Performance Metrics"] if "Performance Metrics" in data else pd.DataFrame()
        avg_return = performance_data["Fund Return (%)"].mean() if "Fund Return (%)" in performance_data else "N/A"
        benchmark_return = performance_data["Benchmark Return (%)"].mean() if "Benchmark Return (%)" in performance_data else "N/A"
        
        # Risk Metrics
        risk_data = data["Risk Metrics"] if "Risk Metrics" in data else pd.DataFrame()
        volatility = risk_data["Volatility (%)"].mean() if "Volatility (%)" in risk_data else "N/A"
        var = risk_data["VaR (%)"].mean() if "VaR (%)" in risk_data else "N/A"

        # Prepare the prompt for Gemini AI
        prompt = f"""
        Fund Name: {fund_name}
        Fund Manager: {fund_manager}
        Launch Date: {launch_date}
        Total Assets: {total_assets}
        Fund Type: {fund_type}
        
        Average Fund Return: {avg_return}%
        Average Benchmark Return: {benchmark_return}%
        Volatility: {volatility}%
        Value at Risk (VaR): {var}%
        
        Please provide a detailed analysis of the fund performance, risk metrics, and overall investment characteristics.
        """
        
        # Return the prompt for Gemini AI to summarize
        return prompt
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return ""

# If file is uploaded
if uploaded_file:
    # Load sheet names and allow user to select sheets
    sheets = pd.ExcelFile(uploaded_file).sheet_names
    selected_sheets = st.multiselect("Select Sheets to Include", sheets, default=sheets)

    data = {}

    # Extract data from each selected sheet
    for sheet in selected_sheets:
        sheet_data = pd.read_excel(uploaded_file, sheet_name=sheet)

        # Convert all datetime columns to string (to handle Timestamps)
        for column in sheet_data.columns:
            if sheet_data[column].dtype == 'datetime64[ns]':
                sheet_data[column] = sheet_data[column].astype(str)

        # Apply str conversion to all other columns to avoid Timestamp issues
        sheet_data = sheet_data.applymap(str)

        data[sheet] = sheet_data

    # Generate Fund Summary using AI
    if uploaded_file:
        st.header("AI-Generated Fund Summary and Analysis")
        
        # Generate the fund summary based on the metrics
        fund_summary_prompt = generate_fund_summary(data)
        
        try:
            # Load and configure the model
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Generate the content using the AI model
            response = model.generate_content(fund_summary_prompt)
            
            # Display the AI generated response
            st.write(response.text)
        except Exception as e:
            st.error(f"Error: {e}")

    # Fund Details Section
    if "Fund Details" in data:
        st.header("Fund Details")
        fund_details = data["Fund Details"]
        st.dataframe(fund_details)

    # Performance Metrics Visualization
    if "Performance Metrics" in data:
        st.header("Fund Performance Metrics")
        performance_data = data["Performance Metrics"]
        st.dataframe(performance_data)

        # Line Chart for Fund vs Benchmark performance
        performance_chart = generate_line_chart(performance_data, ["Fund Return (%)", "Benchmark Return (%)"], "Performance Comparison")
        if performance_chart:
            st.plotly_chart(performance_chart)

    # Asset Allocation Visualization (if data exists)
    if "Asset Allocation" in data:
        st.header("Asset Allocation")
        asset_data = data["Asset Allocation"]
        st.dataframe(asset_data)

        # Pie chart for Asset Allocation
        st.subheader("Asset Allocation Pie Chart")
        asset_alloc = asset_data.groupby("Asset Class")["Value"].sum()
        fig = px.pie(asset_alloc, names=asset_alloc.index, values=asset_alloc.values, title="Asset Allocation")
        st.plotly_chart(fig)

        # Bar chart for top asset classes
        st.subheader("Top Asset Classes")
        top_n = 5
        top_assets = asset_data["Asset Class"].value_counts().head(top_n)
        fig = px.bar(top_assets, x=top_assets.index, y=top_assets.values, title="Top Asset Classes")
        st.plotly_chart(fig)

    # Risk Metrics Visualization
    if "Risk Metrics" in data:
        st.header("Risk Metrics")
        risk_data = data["Risk Metrics"]
        st.dataframe(risk_data)

        # Volatility chart
        st.subheader("Volatility Over Time")
        if "Volatility (%)" in risk_data.columns:
            st.line_chart(risk_data["Volatility (%)"])

        # Value at Risk (VaR) histogram
        st.subheader("Value at Risk (VaR) Histogram")
        if "VaR (%)" in risk_data.columns:
            var_data = risk_data["VaR (%)"].astype(float)
            fig, ax = plt.subplots()
            ax.hist(var_data, bins=20, alpha=0.7, color="red")
            ax.set_title("Value at Risk (VaR) Distribution")
            ax.set_xlabel("VaR")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
