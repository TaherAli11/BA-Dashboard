# 📊  Business Analyst Dashboard

A modern, single-file **Streamlit** application designed to automatically ingest data, detect semantic patterns, and generate actionable business insights without writing code.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b) ![Plotly](https://img.shields.io/badge/Plotly-23F1E3?style=flat&logo=plotly&logoColor=white)

## 🚀 Overview

This dashboard acts as an automated analyst. It accepts raw CSV data (or uses a built-in synthetic generator), cleans it, maps columns to business concepts (like "Revenue", "Date", or "Category"), and performs statistical analysis to find trends, correlations, and outliers. 



**Key Design Goals:**
* **Zero-Config:** Automatically detects Date and Numeric Target columns.
* **Modern UI:** Custom dark theme with soft blue/purple accents.
* **Robust:** Handles missing values and messy date formats gracefully.

## ✨ Key Features

* **📂 Smart Data Ingestion:**
    * Supports CSV upload.
    * **Heuristic column mapping** (automatically identifies Sales/Revenue columns via Regex).
    * Robust date parsing (handles mixed formats).
* **📈 Dynamic Visualizations:**
    * Interactive Plotly charts (Zoom, Pan, Hover).
    * Switch between **Time Series, Bar, Scatter, and Histograms** instantly.
* **🧠 Automated Insights:**
    * **KPI Calculation:** Total Revenue, Averages, and Counts.
    * **Trend Analysis:** Uses Linear Regression to determine if trends are increasing/decreasing significantly.
    * **Correlation Engine:** Identifies strong positive/negative relationships between variables.
    * **Anomaly Detection:** Uses Z-Score analysis to flag statistical outliers.


## 🛠️ Installation

1.  **Clone this repository** (or download the `.py` file):
    ```bash
    git clone https://github.com/TaherAli11/BA-Dashboard.git
    cd ai-business-analyst
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

Run the application locally using the Streamlit CLI:

```bash
streamlit run business_analyst.py