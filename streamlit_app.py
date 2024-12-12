import streamlit as st
import pandas as pd
from fpdf import FPDF
import os
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO

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

        def add_image(self, image_path, x, y, w, h):
            self.image(image_path, x, y, w, h)

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

    # Generate and embed visualizations as images
    if "Visualizations" in data:
        pdf.chapter_title("Visualizations")

        # Generate Bar Chart Example
        if "bar_chart" in data["Visualizations"]:
            bar_chart = data["Visualizations"]["bar_chart"]
            bar_chart_img_path = "bar_chart.png"
            bar_chart.write_image(bar_chart_img_path)
            pdf.add_image(bar_chart_img_path, x=10, y=pdf.get_y(), w=180, h=120)

        # Generate Pie Chart Example
        if "pie_chart" in data["Visualizations"]:
            pie_chart = data["Visualizations"]["pie_chart"]
            pie_chart_img_path = "pie_chart.png"
            pie_chart.write_image(pie_chart_img_path)
            pdf.add_image(pie_chart_img_path, x=10, y=pdf.get_y(), w=180, h=120)

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
            st.subheader(f"{y_axis} vs {x_axis} (Bar Chart)")
            bar_chart_data = sheet_data.groupby(x_axis)[y_axis].sum().reset_index()
            bar_chart = px.bar(bar_chart_data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            data.setdefault("Visualizations", {})["bar_chart"] = bar_chart

        # Create Pie Chart Visualization
        if len(numeric_columns) >= 1 and len(categorical_columns) >= 1:
            st.subheader("Pie Chart Visualization")
            pie_column = st.selectbox("Select Category for Pie Chart", categorical_columns)
            pie_value = st.selectbox("Select Value Column", numeric_columns)

            if pie_column and pie_value:
                pie_data = sheet_data.groupby(pie_column)[pie_value].sum().reset_index()
                pie_chart = px.pie(pie_data, names=pie_column, values=pie_value, title=f"{pie_value} Distribution by {pie_column}")
                data["Visualizations"]["pie_chart"] = pie_chart

        # Create Line Chart for Performance Metrics if applicable
        if 'Performance Metrics' in selected_sheets and len(numeric_columns) > 1:
            st.subheader("Performance Metrics - Line Chart")
            performance_data = sheet_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
            st.line_chart(performance_data)

