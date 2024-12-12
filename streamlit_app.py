import streamlit as st
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import os

# Function to generate a PDF factsheet dynamically
def generate_factsheet(data, output_file):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'Fund Factsheet', 0, 1, 'C')
            self.ln(5)

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, border=False, ln=True, align='L')
            self.ln(5)

        def chapter_body(self, body):
            self.set_font('Arial', '', 10)
            self.multi_cell(0, 10, body)
            self.ln()

        def add_table(self, headers, rows, col_widths=None):
            self.set_font('Arial', 'B', 10)
            if not col_widths:
                col_widths = [40] * len(headers)  # Equal column width by default

            # Header Row
            for header, width in zip(headers, col_widths):
                self.cell(width, 10, header, 1, 0, 'C')
            self.ln()

            # Data Rows
            self.set_font('Arial', '', 10)
            for row in rows:
                for cell, width in zip(row, col_widths):
                    self.cell(width, 10, str(cell), 1, 0, 'C')
                self.ln()

        def add_graph(self, image_path):
            self.image(image_path, x=10, y=self.get_y(), w=190)
            self.ln(65)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, 'Legal Disclaimer: Past performance is not indicative of future results.', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()

    # Fund Details Section
    if 'Fund Details' in data:
        pdf.chapter_title('Fund Overview')
        fund_details = data['Fund Details']
        for key, value in fund_details.items():
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(60, 10, key + ":", border=False)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, str(value), ln=True)

    # Portfolio Holdings Section
    if 'Portfolio Holdings' in data:
        pdf.chapter_title('Portfolio Holdings')
        portfolio = data['Portfolio Holdings']
        headers = list(portfolio.columns)
        rows = portfolio.values.tolist()
        pdf.add_table(headers, rows)

    # Performance Metrics Section
    if 'Performance Metrics' in data:
        pdf.chapter_title('Performance Metrics')
        performance = data['Performance Metrics']
        for key, value in performance.items():
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(60, 10, key + ":", border=False)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, str(value), ln=True)

    # Risk Metrics Section
    if 'Risk Metrics' in data:
        pdf.chapter_title('Risk Metrics')
        risk = data['Risk Metrics']
        for key, value in risk.items():
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(60, 10, key + ":", border=False)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, str(value), ln=True)

    # Add Graph if available
    if 'Graph' in data:
        pdf.chapter_title('Fund Performance Chart')
        pdf.add_graph(data['Graph'])

    pdf.footer()
    pdf.output(output_file)

# Streamlit application
st.title('Dynamic Fund Factsheet Generator')

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    sheets = pd.ExcelFile(uploaded_file).sheet_names
    selected_sheets = st.multiselect("Select Sheets to Include", sheets, default=sheets)

    if selected_sheets:
        data = {}

        # Parse the data and organize it
        for sheet in selected_sheets:
            sheet_data = pd.read_excel(uploaded_file, sheet_name=sheet)
            data[sheet] = sheet_data

        # Parse specific sections from the Excel file
        if st.button("Generate Factsheet"):
            # Example of how the data dictionary should be structured for real fund factsheet:
            # This is an example; modify based on the Excel sheet's contents
            factsheet_data = {
                "Fund Details": {
                    "Fund Name": "XYZ Growth Fund",
                    "Launch Date": "01-Jan-2010",
                    "Investment Objective": "The fund aims to achieve long-term capital growth by investing in a diversified portfolio of equities.",
                    "Fund Manager": "John Doe"
                },
                "Portfolio Holdings": pd.DataFrame({
                    "Asset": ["Stock A", "Stock B", "Bond C", "Cash"],
                    "Allocation (%)": [40, 35, 15, 10]
                }),
                "Performance Metrics": {
                    "1 Year Return": "8%",
                    "YTD Return": "5%",
                    "3 Year Return": "12%",
                    "Since Inception": "60%"
                },
                "Risk Metrics": {
                    "Standard Deviation": "15%",
                    "Sharpe Ratio": "1.2"
                },
                "Graph": "path_to_performance_graph.png"  # Path to graph image if needed
            }

            # Generating the PDF
            output_file = "dynamic_factsheet.pdf"
            generate_factsheet(factsheet_data, output_file)

            with open(output_file, "rb") as pdf_file:
                st.download_button(label="Download Factsheet", data=pdf_file, file_name=output_file, mime="application/pdf")

# Dynamic plotting for visualization
if uploaded_file:
    st.header("Data Visualization")
    sheet_name = st.selectbox("Select a sheet for visualization", sheets)
    sheet_data = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    st.dataframe(sheet_data)

    numeric_columns = sheet_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = sheet_data.select_dtypes(include=['object']).columns

    x_axis = st.selectbox("Select X-Axis", categorical_columns)
    y_axis = st.selectbox("Select Y-Axis", numeric_columns)

    if x_axis and y_axis:
        plt.figure(figsize=(10, 6))
        plt.bar(sheet_data[x_axis], sheet_data[y_axis])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f"{y_axis} vs {x_axis}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        graph_path = "graph.png"
        plt.savefig(graph_path)
        st.pyplot(plt)

        if st.button("Add Graph to Factsheet"):
            output_file = "dynamic_factsheet_with_graph.pdf"
            generate_factsheet({"Graph": graph_path}, output_file)

            with open(output_file, "rb") as pdf_file:
                st.download_button(label="Download Factsheet with Graph", data=pdf_file, file_name=output_file, mime="application/pdf")
