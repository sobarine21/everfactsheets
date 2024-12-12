import streamlit as st
import pandas as pd
from fpdf import FPDF

# Generate PDF Function
def generate_factsheet(data, output_file):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'VC & PE Fund Factsheet', 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Investment Objective
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, 'Investment Objective:', ln=True)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10, data['investment_objective'])

    # Fund Manager Details
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, 'Fund Manager Details:', ln=True)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10, data['fund_manager'])

    # Key Metrics
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, 'Key Metrics:', ln=True)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Total Fund Size: {data['total_fund_size']}", ln=True)
    pdf.cell(0, 10, f"Vintage Year: {data['vintage_year']}", ln=True)
    pdf.cell(0, 10, f"Number of Portfolio Companies: {data['portfolio_companies']}", ln=True)

    # Top Portfolio Companies
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, 'Top Portfolio Companies:', ln=True)
    pdf.set_font('Arial', '', 10)
    for company in data['top_companies']:
        pdf.cell(0, 10, f"- {company}", ln=True)

    # Financial Metrics
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, 'Financial Metrics:', ln=True)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"IRR: {data['irr']}%", ln=True)
    pdf.cell(0, 10, f"MOIC: {data['moic']}x", ln=True)

    # Sector Breakdown
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, 'Sector Breakdown:', ln=True)
    pdf.set_font('Arial', '', 10)
    for sector, percentage in data['sector_breakdown'].items():
        pdf.cell(0, 10, f"{sector}: {percentage}%", ln=True)

    # Disclosures (on the second page)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, 'Disclosures:', ln=True)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10, data['disclosures'])

    pdf.output(output_file)

# Streamlit Web App
st.title("VC & PE Fund Factsheet Generator")

uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name=None)

    # Extract Data from Different Sheets
    overview_data = df['Overview'].iloc[0]
    top_companies = df['Top Companies']['Company Name'].tolist()
    sector_breakdown = dict(zip(df['Sector Breakdown']['Sector'], df['Sector Breakdown']['Percentage']))

    # Factsheet Data
    factsheet_data = {
        'investment_objective': overview_data['Investment Objective'],
        'fund_manager': overview_data['Fund Manager'],
        'total_fund_size': overview_data['Total Fund Size'],
        'vintage_year': overview_data['Vintage Year'],
        'portfolio_companies': overview_data['Portfolio Companies'],
        'top_companies': top_companies,
        'irr': overview_data['IRR'],
        'moic': overview_data['MOIC'],
        'sector_breakdown': sector_breakdown,
        'disclosures': df['Disclosures'].iloc[0, 0]
    }

    if st.button("Generate Factsheet"):
        output_file = "VC_PE_Fund_Factsheet.pdf"
        generate_factsheet(factsheet_data, output_file)
        with open(output_file, "rb") as file:
            st.download_button(
                label="Download Factsheet",
                data=file,
                file_name=output_file,
                mime="application/pdf"
            )

# Required Installations
# pip install streamlit pandas fpdf
