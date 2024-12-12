import streamlit as st
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import os

# Function to generate a PDF factsheet dynamically
def generate_factsheet(data, output_file):
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
                    self.cell(40, 10, str(cell), 1, 0, 'C')
                self.ln()

        def add_graph(self, image_path):
            self.image(image_path, x=10, y=self.get_y(), w=190)
            self.ln(65)

    pdf = PDF()
    pdf.add_page()

    # Dynamically add sections from the data
    for section, content in data.items():
        if isinstance(content, dict):
            pdf.chapter_title(section)
            for sub_section, sub_content in content.items():
                if isinstance(sub_content, pd.DataFrame):
                    pdf.chapter_title(sub_section)
                    headers = list(sub_content.columns)
                    rows = sub_content.values.tolist()
                    pdf.add_table(headers, rows)
                else:
                    pdf.chapter_body(f"{sub_section}: {sub_content}")
        elif isinstance(content, pd.DataFrame):
            pdf.chapter_title(section)
            headers = list(content.columns)
            rows = content.values.tolist()
            pdf.add_table(headers, rows)
        elif isinstance(content, str):
            pdf.chapter_title(section)
            pdf.chapter_body(content)

    # Save the PDF
    pdf.output(output_file)

# Streamlit application
st.title('Dynamic Fund Factsheet Generator')

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    sheets = pd.ExcelFile(uploaded_file).sheet_names
    selected_sheets = st.multiselect("Select Sheets to Include", sheets, default=sheets)

    if selected_sheets:
        data = {}
        
        # Extract data from each selected sheet
        for sheet in selected_sheets:
            sheet_data = pd.read_excel(uploaded_file, sheet_name=sheet, header=None)
            
            # Inspect the number of columns before renaming
            print(f"Columns in {sheet}: {sheet_data.columns}")
            
            # Check if the sheet has the expected structure (2 columns)
            if sheet_data.shape[1] == 2:
                sheet_data.columns = ['Data Point', 'Value']  # Set columns to expected format
            else:
                # Print the first few rows to debug the column issue
                print(f"Sheet '{sheet}' has unexpected number of columns. Data preview:\n", sheet_data.head())
                continue  # Skip this sheet or handle it differently
            
            if sheet == 'Fund Details':
                fund_details = sheet_data.set_index('Data Point').to_dict()['Value']
                data['Fund Details'] = fund_details
            elif sheet == 'Portfolio Holdings':
                data[sheet] = sheet_data
            elif sheet == 'Performance Metrics':
                data[sheet] = sheet_data
            elif sheet == 'Risk Metrics':
                data[sheet] = sheet_data
            elif sheet == 'Performance':
                data[sheet] = sheet_data

        # Generate factsheet when button is clicked
        if st.button("Generate Factsheet"):
            output_file = "dynamic_factsheet.pdf"
            generate_factsheet(data, output_file)

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
