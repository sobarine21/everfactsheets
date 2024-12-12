import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import seaborn as sns
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
                    self.cell(40, 10, cell, 1, 0, 'C')
                self.ln()

        def add_graph(self, image_path):
            self.image(image_path, x=10, y=self.get_y(), w=190)
            self.ln(65)

    pdf = PDF()
    pdf.add_page()

    # Add fund details section
    if "Fund Details" in data:
        pdf.chapter_title('Fund Details')
        fund_details = data["Fund Details"]
        for sub_section, sub_content in fund_details.items():
            pdf.chapter_body(f"{sub_section}: {sub_content}")

    # Add portfolio holdings section
    if "Portfolio Holdings" in data:
        pdf.chapter_title('Portfolio Holdings')
        portfolio_data = data["Portfolio Holdings"]
        headers = list(portfolio_data.columns)
        rows = portfolio_data.values.tolist()
        pdf.add_table(headers, rows)

    # Add performance metrics section
    if "Performance Metrics" in data:
        pdf.chapter_title('Performance Metrics')
        performance_data = data["Performance Metrics"]
        headers = list(performance_data.columns)
        rows = performance_data.values.tolist()
        pdf.add_table(headers, rows)

    # Add risk metrics section
    if "Risk Metrics" in data:
        pdf.chapter_title('Risk Metrics')
        risk_data = data["Risk Metrics"]
        headers = list(risk_data.columns)
        rows = risk_data.values.tolist()
        pdf.add_table(headers, rows)

    # Add graphs to the PDF
    if "Graphs" in data:
        for graph in data["Graphs"]:
            pdf.add_graph(graph)

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
            sheet_data = pd.read_excel(uploaded_file, sheet_name=sheet)

            # Convert all datetime columns to string (to handle Timestamps)
            for column in sheet_data.columns:
                if sheet_data[column].dtype == 'datetime64[ns]':
                    sheet_data[column] = sheet_data[column].astype(str)

            # Apply str conversion to all other columns to avoid Timestamp issues
            sheet_data = sheet_data.applymap(str)

            # Handle Fund Details tab: Process rows for each fund
            if sheet == 'Fund Details':
                fund_details = sheet_data.set_index('Fund Name').T.to_dict('dict')
                data['Fund Details'] = fund_details
            
            # Handle Portfolio Holdings tab: Table format
            elif sheet == 'Portfolio Holdings':
                data[sheet] = sheet_data

            # Handle Performance Metrics tab: Table format
            elif sheet == 'Performance Metrics':
                data[sheet] = sheet_data

            # Handle Risk Metrics tab: Table format
            elif sheet == 'Risk Metrics':
                data[sheet] = sheet_data

            # Handle Performance tab: Table format
            elif sheet == 'Performance':
                data[sheet] = sheet_data

        # Generate factsheet when button is clicked
        if st.button("Generate Factsheet"):
            output_file = "dynamic_factsheet.pdf"
            generate_factsheet(data, output_file)

            with open(output_file, "rb") as pdf_file:
                st.download_button(label="Download Factsheet", data=pdf_file, file_name=output_file, mime="application/pdf")

        # Dynamic plotting for visualization
        st.header("Data Visualization")
        sheet_name = st.selectbox("Select a sheet for visualization", selected_sheets)
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

            # Add Graph to PDF
            if st.button("Add Graph to Factsheet"):
                if "Graphs" not in data:
                    data["Graphs"] = []
                data["Graphs"].append(graph_path)

                output_file_with_graph = "dynamic_factsheet_with_graph.pdf"
                generate_factsheet(data, output_file_with_graph)

                with open(output_file_with_graph, "rb") as pdf_file:
                    st.download_button(label="Download Factsheet with Graph", data=pdf_file, file_name=output_file_with_graph, mime="application/pdf")

        # Create Pie Chart Visualization
        if len(numeric_columns) >= 1 and len(categorical_columns) >= 1:
            st.subheader("Pie Chart Visualization")
            pie_column = st.selectbox("Select Category for Pie Chart", categorical_columns)
            pie_value = st.selectbox("Select Value Column", numeric_columns)

            if pie_column and pie_value:
                pie_data = sheet_data.groupby(pie_column)[pie_value].sum()
                plt.figure(figsize=(8, 6))
                pie_data.plot(kind='pie', autopct='%1.1f%%', startangle=90)
                plt.title(f"{pie_value} Distribution by {pie_column}")
                plt.ylabel('')
                plt.tight_layout()

                pie_chart_path = "pie_chart.png"
                plt.savefig(pie_chart_path)
                st.pyplot(plt)

                # Add Pie Chart to PDF
                if st.button("Add Pie Chart to Factsheet"):
                    if "Graphs" not in data:
                        data["Graphs"] = []
                    data["Graphs"].append(pie_chart_path)

                    output_file_with_pie = "dynamic_factsheet_with_pie.pdf"
                    generate_factsheet(data, output_file_with_pie)

                    with open(output_file_with_pie, "rb") as pdf_file:
                        st.download_button(label="Download Factsheet with Pie Chart", data=pdf_file, file_name=output_file_with_pie, mime="application/pdf")
