import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Function to generate line chart for performance comparison (Fund vs Benchmark)
def generate_line_chart(data, columns, title):
    # Ensure the columns in 'y' exist in the dataframe
    columns = [col for col in columns if col in data.columns]
    
    if len(columns) < 2:
        st.error("Not enough columns available for comparison!")
        return None
    
    fig = px.line(data, x=data.index, y=columns, title=title)
    return fig

# Streamlit App
st.title('Dynamic Fund Visualization Dashboard')

# File uploader to upload the Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

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

    # Display Fund Details Section
    if "Fund Details" in data:
        st.header("Fund Details")
        fund_details = data["Fund Details"]
        
        # Display the first row of fund details as a table
        st.dataframe(fund_details)

    # Performance Metrics Visualization
    if "Performance Metrics" in data:
        st.header("Fund Performance Metrics")
        performance_data = data["Performance Metrics"]

        # Display Performance Metrics as a table
        st.dataframe(performance_data)

        # Line Chart for Fund vs Benchmark performance
        performance_chart = generate_line_chart(performance_data, ["Fund Return", "Benchmark Return"], "Performance Comparison")
        if performance_chart:
            st.plotly_chart(performance_chart)

        # Histogram for Fund Returns Distribution
        st.subheader("Fund Returns Distribution")
        if "Fund Return" in performance_data.columns:
            return_data = performance_data["Fund Return"].astype(float, errors='ignore')
            fig, ax = plt.subplots()
            ax.hist(return_data, bins=20, alpha=0.7, color="blue")
            ax.set_title("Fund Return Distribution")
            ax.set_xlabel("Return")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    # Asset Allocation Visualization (if data exists)
    if "Asset Allocation" in data:
        st.header("Asset Allocation")
        asset_data = data["Asset Allocation"]

        # Display Asset Allocation as a table
        st.dataframe(asset_data)

        # Pie chart for Asset Allocation
        st.subheader("Asset Allocation Pie Chart")
        asset_alloc = asset_data.groupby("Asset Class")["Value"].sum()
        fig = px.pie(asset_alloc, names=asset_alloc.index, values=asset_alloc.values, title="Asset Allocation")
        st.plotly_chart(fig)

    # Performance over time line chart (if data exists)
    if "Performance Over Time" in data:
        st.header("Performance Over Time")
        performance_over_time = data["Performance Over Time"]

        # Line chart for performance over time
        performance_time_chart = generate_line_chart(performance_over_time, ["Fund Return", "Benchmark Return"], "Performance Over Time")
        if performance_time_chart:
            st.plotly_chart(performance_time_chart)

    # Additional visualizations like sector distribution or other metrics can be added here...
