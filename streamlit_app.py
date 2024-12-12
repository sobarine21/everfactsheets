import streamlit as st
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt

# Function to generate a pie chart for sector breakdown
def generate_sector_chart(data, output_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(data['Percentage'], labels=data['Sector'], autopct='%1.1f%%', startangle=140)
    plt.title("Sector Breakdown")
    plt.savefig(output_path)
    plt.close()

# Function to generate a factsheet PDF
def generate_factsheet(data, output_file):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1 - Factsheet Content
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add Investment Objective
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Investment Objective", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, data['investment_objective'])  # Removed `ln=True`
    pdf.ln(5)

    # Add Top Holdings
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Top Holdings", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(60, 10, "Stock Name", 1, 0)
    pdf.cell(40, 10, "Percentage (%)", 1, 1)
    for index, row in data['top_holdings'].iterrows():
        pdf.cell(60, 10, row['Stock Name'], 1, 0)
        pdf.cell(40, 10, str(row['Percentage']), 1, 1)

    pdf.ln(5)

    # Add Sector Breakdown Pie Chart
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Sector Breakdown", ln=True)
    pdf.ln(5)

    pdf.image(data['sector_chart'], x=10, w=190)  # Add the saved chart

    # Page 2 - Disclosures
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Disclosures", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, data['disclosures'])  # Removed `ln=True`

    pdf.output(output_file)

# Streamlit application
def main():
    st.title("VC/PE Fund Factsheet Generator")
    st.write("Upload your Excel file with all required data points to generate a factsheet.")

    # File upload
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    
    if uploaded_file:
        # Load Excel file
        excel_data = pd.ExcelFile(uploaded_file)
        
        # Load different tabs
        investment_objective = excel_data.parse('Investment Objective')
        top_holdings = excel_data.parse('Top Holdings')
        sector_breakdown = excel_data.parse('Sector Breakdown')
        disclosures = excel_data.parse('Disclosures')

        # Extract data for the factsheet
        data = {
            'investment_objective': investment_objective.iloc[0, 0],
            'top_holdings': top_holdings,
            'sector_breakdown': sector_breakdown,
            'disclosures': disclosures.iloc[0, 0],
            'sector_chart': "sector_chart.png"
        }

        # Generate sector breakdown chart
        generate_sector_chart(sector_breakdown, data['sector_chart'])

        # Generate the factsheet
        output_file = "factsheet.pdf"
        generate_factsheet(data, output_file)

        # Provide download link
        with open(output_file, "rb") as file:
            st.download_button(
                label="Download Factsheet PDF",
                data=file,
                file_name="factsheet.pdf",
                mime="application/pdf",
            )

if __name__ == "__main__":
    main()
