import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import io

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
    corr = data.corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    plt.title(title)
    st.pyplot(fig)

# Function to generate bar chart for top assets
def generate_bar_chart(data, column, top_n, title):
    top_assets = data[column].value_counts().head(top_n)
    fig = px.bar(top_assets, x=top_assets.index, y=top_assets.values, title=title)
    st.plotly_chart(fig)

# Streamlit App
st.title('Dynamic Fund Visualization Dashboard')

# File uploader to upload the Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

# Option to download a template Excel file
def generate_template():
    # Create a template DataFrame
    fund_details = pd.DataFrame({
        "Fund Name": [""],
        "Fund Manager": [""],
        "Launch Date": [""],
        "Total Assets": [""],
        "Fund Type": [""]
    })

    performance_metrics = pd.DataFrame({
        "Date": [""],
        "Fund Return (%)": [""],
        "Benchmark Return (%)": [""],
        "Sharpe Ratio": [""]
    })

    asset_allocation = pd.DataFrame({
        "Asset Class": [""],
        "Value": [""]
    })

    performance_over_time = pd.DataFrame({
        "Date": [""],
        "Fund Return (%)": [""],
        "Benchmark Return (%)": [""]
    })

    risk_metrics = pd.DataFrame({
        "Date": [""],
        "Volatility (%)": [""],
        "VaR (%)": [""]
    })

    sector_allocation = pd.DataFrame({
        "Sector": [""],
        "Value": [""]
    })

    investment_simulation = pd.DataFrame({
        "Annual Return (%)": [""]
    })

    # Save these DataFrames into an Excel file
    with pd.ExcelWriter('fund_dashboard_template.xlsx') as writer:
        fund_details.to_excel(writer, sheet_name='Fund Details', index=False)
        performance_metrics.to_excel(writer, sheet_name='Performance Metrics', index=False)
        asset_allocation.to_excel(writer, sheet_name='Asset Allocation', index=False)
        performance_over_time.to_excel(writer, sheet_name='Performance Over Time', index=False)
        risk_metrics.to_excel(writer, sheet_name='Risk Metrics', index=False)
        sector_allocation.to_excel(writer, sheet_name='Sector Allocation', index=False)
        investment_simulation.to_excel(writer, sheet_name='Investment Simulation', index=False)

    # Return the file in memory
    with open('fund_dashboard_template.xlsx', 'rb') as f:
        template_file = f.read()
    
    return template_file

# Download button for the template
if st.sidebar.button("Download Data Template"):
    template_file = generate_template()
    st.sidebar.download_button(
        label="Download Fund Dashboard Template",
        data=template_file,
        file_name="fund_dashboard_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

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
        performance_chart = generate_line_chart(performance_data, ["Fund Return", "Benchmark Return"], "Performance Comparison")
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
        generate_bar_chart(asset_data, "Asset Class", 5, "Top 5 Asset Classes")

    # Performance over time line chart (if data exists)
    if "Performance Over Time" in data:
        st.header("Performance Over Time")
        performance_over_time = data["Performance Over Time"]
        performance_time_chart = generate_line_chart(performance_over_time, ["Fund Return", "Benchmark Return"], "Performance Over Time")
        if performance_time_chart:
            st.plotly_chart(performance_time_chart)

    # Correlation heatmap for performance metrics
    if "Performance Metrics" in data:
        st.header("Correlation Heatmap of Performance Metrics")
        performance_metrics = data["Performance Metrics"]
        generate_correlation_heatmap(performance_metrics, "Correlation of Performance Metrics")

    # Sector Distribution (if data exists)
    if "Sector Allocation" in data:
        st.header("Sector Allocation")
        sector_data = data["Sector Allocation"]
        sector_alloc = sector_data.groupby("Sector")["Value"].sum()
        fig = px.pie(sector_alloc, names=sector_alloc.index, values=sector_alloc.values, title="Sector Allocation")
        st.plotly_chart(fig)

    # Cumulative Returns Over Time
    if "Performance Over Time" in data:
        st.header("Cumulative Fund Return Over Time")
        performance_over_time = data["Performance Over Time"]
        performance_over_time["Cumulative Fund Return"] = (1 + performance_over_time["Fund Return"].astype(float) / 100).cumprod() - 1
        st.line_chart(performance_over_time["Cumulative Fund Return"])

    # Risk Metrics Visualization
    if "Risk Metrics" in data:
        st.header("Risk Metrics")
        risk_data = data["Risk Metrics"]
        st.dataframe(risk_data)

        # Volatility chart
        st.subheader("Volatility Over Time")
        if "Volatility" in risk_data.columns:
            st.line_chart(risk_data["Volatility"])

        # Value at Risk (VaR) histogram
        st.subheader("Value at Risk (VaR) Histogram")
        if "VaR" in risk_data.columns:
            var_data = risk_data["VaR"].astype(float)
            fig, ax = plt.subplots()
            ax.hist(var_data, bins=20, alpha=0.7, color="red")
            ax.set_title("Value at Risk (VaR) Distribution")
            ax.set_xlabel("VaR")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    # Sharpe Ratio Analysis
    if "Performance Metrics" in data:
        st.header("Sharpe Ratio Analysis")
        performance_data = data["Performance Metrics"]
        if "Sharpe Ratio" in performance_data.columns:
            sharpe_ratio = performance_data["Sharpe Ratio"].astype(float)
            st.line_chart(sharpe_ratio)

    # Interactive Table with Performance Data
    if "Performance Metrics" in data:
        st.header("Interactive Performance Data Table")
        performance_data = data["Performance Metrics"]
        st.dataframe(performance_data)

    # Moving Average Chart
    if "Performance Over Time" in data:
        st.header("Moving Average of Fund Returns")
        performance_data = data["Performance Over Time"]
        moving_avg = performance_data["Fund Return"].astype(float).rolling(window=12).mean()
        st.line_chart(moving_avg)

    # Create an investment growth simulation
    if "Investment Simulation" in data:
        st.header("Investment Growth Simulation")
        simulation_data = data["Investment Simulation"]
        initial_investment = st.number_input("Initial Investment Amount", min_value=0, value=10000)
        years = st.slider("Investment Duration (years)", 1, 30, 10)
        growth_rate = simulation_data["Annual Return"].mean()
        future_value = initial_investment * (1 + growth_rate / 100) ** years
        st.write(f"Estimated Future Value: ${future_value:,.2f}")
