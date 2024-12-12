import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

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
    pdf.multi_cell(0, 10, data['investment_objective'], ln=True)
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
    pdf.multi_cell(0, 10, data['disclosures'], ln=True)

    pdf.output(output_file)

# Streamlit App
st.title("VC/PE Fund Factsheet Generator")

# File Upload
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    # Read Excel File
    data_excel = pd.ExcelFile(uploaded_file)

    # Load data from specific tabs
    investment_objective = data_excel.parse("Investment Objective").iloc[0, 0]
    top_holdings = data_excel.parse("Top Holdings")
    sector_breakdown = data_excel.parse("Sector Breakdown")
    disclosures = data_excel.parse("Disclosures").iloc[0, 0]

    # Preview Data
    st.subheader("Investment Objective")
    st.write(investment_objective)

    st.subheader("Top Holdings")
    st.dataframe(top_holdings)

    st.subheader("Sector Breakdown")
    st.dataframe(sector_breakdown)

    # Generate Sector Breakdown Pie Chart
    st.subheader("Sector Breakdown Chart")
    fig, ax = plt.subplots()
    ax.pie(
        sector_breakdown["Percentage"],
        labels=sector_breakdown["Sector"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Save chart as image
    chart_path = "sector_chart.png"
    fig.savefig(chart_path)

    # Generate Factsheet
    if st.button("Generate Factsheet"):
        data = {
            "investment_objective": investment_objective,
            "top_holdings": top_holdings,
            "sector_chart": chart_path,
            "disclosures": disclosures,
        }
        output_file = "VC_PE_Factsheet.pdf"
        generate_factsheet(data, output_file)
        st.success(f"Factsheet generated: {output_file}")
        with open(output_file, "rb") as file:
            st.download_button(
                label="Download Factsheet",
                data=file,
                file_name=output_file,
                mime="application/pdf",
            )
