import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from io import BytesIO
import datetime

# --- Data Preprocessing & Calculations ---

# Function to calculate annualized return
def calculate_annualized_return(data, years=1):
    return (data[-1] / data[0]) ** (1 / years) - 1

# Function to calculate max drawdown
def calculate_max_drawdown(data):
    cumulative_returns = (1 + data).cumprod()
    peak = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - peak) / peak
    max_drawdown = drawdowns.min()
    return max_drawdown

# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    return (returns.mean() - risk_free_rate) / returns.std()

# Function to calculate CAGR
def calculate_cagr(initial_value, final_value, years):
    return (final_value / initial_value) ** (1 / years) - 1

# Function to calculate the total AUM
def calculate_total_aum(data):
    return data['AUM'].sum()

# Function to calculate excess return over benchmark
def calculate_excess_return(fund_returns, benchmark_returns):
    return fund_returns - benchmark_returns

# --- PDF Generation ---

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Fund Factsheet', border=False, ln=True, align='C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, border=False, ln=True, align='L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_table(self, headers, rows):
        self.set_font('Arial', 'B', 10)
        for header in headers:
            self.cell(40, 10, header, 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 10)
        for row in rows:
            for cell in row:
                self.cell(40, 10, cell, 1, 0, 'C')
            self.ln()

    def add_image(self, image_path, x, y, w, h):
        self.image(image_path, x, y, w, h)

# --- Visualization & Interactive Elements ---

# Function to generate pie chart
def generate_pie_chart(data, column, title):
    pie_data = data[column].value_counts().reset_index()
    pie_data.columns = [column, 'Count']
    fig = px.pie(pie_data, names=column, values='Count', title=title)
    return fig

# Function to generate line chart
def generate_line_chart(data, columns, title):
    fig = px.line(data, x=data.index, y=columns, title=title)
    return fig

# Function to generate a bar chart
def generate_bar_chart(data, x_column, y_column, title):
    fig = px.bar(data, x=x_column, y=y_column, title=title)
    return fig

# --- Streamlit App Code ---

st.title('Dynamic Fund Factsheet Generator')

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
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

    # Generate factsheet when button is clicked
    if st.button("Generate Factsheet"):
        output_file = "dynamic_factsheet.pdf"
        pdf = PDF()
        pdf.add_page()

        # Add Fund Details section
        if "Fund Details" in data:
            pdf.chapter_title('Fund Details')
            fund_details = data["Fund Details"]
            for sub_section, sub_content in fund_details.items():
                pdf.chapter_body(f"{sub_section}: {sub_content}")

        # Add Total AUM and other metrics
        if "Portfolio Holdings" in data:
            pdf.chapter_title('Total AUM')
            portfolio_data = data["Portfolio Holdings"]
            total_aum = calculate_total_aum(portfolio_data)
            pdf.chapter_body(f"Total AUM: {total_aum}")

        # Add Fund Returns Metrics
        if "Performance Metrics" in data:
            pdf.chapter_title('Performance Metrics')
            performance_data = data["Performance Metrics"]
            headers = list(performance_data.columns)
            rows = performance_data.values.tolist()
            pdf.add_table(headers, rows)

        # Add Visualizations
        if "Visualizations" in data:
            # Example: Pie chart for sector allocation
            pdf.chapter_title("Sector Allocation")
            sector_chart = generate_pie_chart(data["Visualizations"], "Sector", "Sector Allocation")
            sector_chart.write_image("sector_chart.png")
            pdf.add_image("sector_chart.png", 10, pdf.get_y(), 180, 120)

        pdf.output(output_file)

        with open(output_file, "rb") as pdf_file:
            st.download_button(label="Download Factsheet", data=pdf_file, file_name=output_file, mime="application/pdf")

    # Data Visualization: Generate Charts
    if "Performance Metrics" in selected_sheets:
        st.header("Data Visualization")
        performance_data = data["Performance Metrics"]

        # Line Chart for Performance
        performance_chart = generate_line_chart(performance_data, ["Fund Return", "Benchmark Return"], "Performance Comparison")
        st.plotly_chart(performance_chart)

        # Bar Chart for Fund Returns
        bar_chart = generate_bar_chart(performance_data, "Date", "Fund Return", "Fund Returns Over Time")
        st.plotly_chart(bar_chart)

        # Risk vs Return Scatter Plot
        st.subheader("Risk vs Return")
        risk_return_data = performance_data[["Risk", "Return"]]
        fig = px.scatter(risk_return_data, x="Risk", y="Return", title="Risk vs Return")
        st.plotly_chart(fig)

        # Portfolio Allocation Pie Chart
        if "Portfolio Holdings" in selected_sheets:
            portfolio_data = data["Portfolio Holdings"]
            allocation_pie_chart = generate_pie_chart(portfolio_data, "Asset Class", "Portfolio Allocation")
            st.plotly_chart(allocation_pie_chart)

        # Generate Histogram of Returns
        st.subheader("Return Distribution")
        return_data = performance_data["Fund Return"].astype(float)
        plt.hist(return_data, bins=20, alpha=0.7, color="blue")
        plt.title("Return Distribution")
        st.pyplot(plt)

