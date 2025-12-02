"""
Business Analyst Dashboard  
Single-file Streamlit app with robust data handling, smart column detection,
and automated insight generation.

Usage:
1. pip install streamlit pandas numpy plotly scikit-learn
2. streamlit run business_analyst_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
import re

# -----------------------
# Configuration & Styling
# -----------------------
ACCENT_A = "#5BC0FF"   # Soft Blue
ACCENT_B = "#7C7FFF"   # Soft Purple
BG_COLOR = "#0f1720"
CARD_BG = "#111827"
TEXT_COLOR = "#E6EEF3"
MUTED_COLOR = "#98A0AB"
CHART_COLORS = [ACCENT_A, ACCENT_B, "#9BE7FF", "#C7B3FF", "#6EE7B7", "#F472B6"]

st.set_page_config(page_title="Business Analyst", layout="wide", page_icon="")

# Modern CSS with specific targeting
st.markdown(f"""
<style>
    /* Global Variables */
    :root {{
        --accent-a: {ACCENT_A};
        --accent-b: {ACCENT_B};
        --bg: {BG_COLOR};
        --card: {CARD_BG};
        --text: {TEXT_COLOR};
        --muted: {MUTED_COLOR};
    }}
    
    /* Base App Styling */
    .stApp {{
        background-color: var(--bg);
        color: var(--text);
    }}
    
    /* Custom Headers */
    .dashboard-header {{
        padding: 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 24px;
    }}
    .main-title {{
        font-size: 32px;
        font-weight: 700;
        background: -webkit-linear-gradient(0deg, var(--accent-a), var(--accent-b));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    .sub-title {{
        font-size: 16px;
        color: var(--muted);
        margin-top: 5px;
    }}

    /* Insight Cards */
    .insight-card {{
        background-color: var(--card);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s;
    }}
    .insight-card:hover {{
        border-color: var(--accent-a);
    }}
    
    /* KPI Metrics */
    .metric-value {{
        font-size: 24px;
        font-weight: 700;
        color: var(--text);
    }}
    .metric-label {{
        font-size: 13px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Streamlit Overrides */
    div[data-testid="stExpander"] {{
        background-color: var(--card);
        border: none;
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Utility & Logic Functions
# -----------------------

@st.cache_data
def load_sample_data():
    """Generates a synthetic sales dataset."""
    rng = np.random.default_rng(42)
    n = 1000
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n)
    
    data = {
        "Date": dates,
        "Region": rng.choice(["North", "South", "East", "West"], size=n),
        "Category": rng.choice(["Electronics", "Furniture", "Office", "Software"], size=n),
        "Sales_Channel": rng.choice(["Online", "Direct", "Retail"], size=n),
        "Units_Sold": rng.integers(1, 50, size=n),
        "Unit_Price": np.round(rng.lognormal(mean=3, sigma=0.5, size=n), 2),
    }
    df = pd.DataFrame(data)
    
    # Calculate dependent variables
    df["Total_Revenue"] = df["Units_Sold"] * df["Unit_Price"]
    df["Cost"] = df["Total_Revenue"] * rng.uniform(0.5, 0.8, size=n)
    df["Profit"] = df["Total_Revenue"] - df["Cost"]
    
    # Add some noise/outliers/missing
    mask = rng.random(n) < 0.02
    df.loc[mask, "Total_Revenue"] = np.nan
    
    return df

@st.cache_data
def clean_and_parse_data(df):
    """
    Intelligent column typing and basic cleaning.
    """
    df = df.copy()
    
    # 1. Date Parsing
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try parsing as date
            try:
                # Use flexible parsing
                temp = pd.to_datetime(df[col], errors='coerce')
                # If > 80% successfully parsed and it wasn't just numbers
                if temp.notna().mean() > 0.8:
                     df[col] = temp
            except Exception:
                pass
                
    return df

def identify_semantic_columns(df):
    """
    Heuristics to identify Date, Numeric (Target), and Categorical columns.
    """
    cols = df.columns
    semantic = {
        "date_col": None,
        "num_cols": [],
        "cat_cols": [],
        "target_col": None  # Best guess for 'Revenue' or 'Sales'
    }
    
    # Find Date
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns
    if len(date_cols) > 0:
        semantic['date_col'] = date_cols[0]
    
    # Find Numerics
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    semantic['num_cols'] = num_cols
    
    # Find Categories (Object or Low Cardinality Integers)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    semantic['cat_cols'] = cat_cols
    
    # Guess Target (Revenue/Sales/Amount)
    if num_cols:
        # Priority regex for finance terms
        priority = [r'rev', r'sale', r'amount', r'price', r'profit', r'cost']
        for p in priority:
            for c in num_cols:
                if re.search(p, c.lower()):
                    semantic['target_col'] = c
                    break
            if semantic['target_col']: break
        
        # Fallback to the numeric column with highest sum if no name match
        if not semantic['target_col']:
            sums = df[num_cols].sum(numeric_only=True).sort_values(ascending=False)
            if not sums.empty:
                semantic['target_col'] = sums.index[0]
                
    return semantic

def get_outliers(series, z_thresh=3):
    """Robust Z-score outlier detection."""
    clean_s = series.dropna()
    if clean_s.empty or clean_s.std() == 0:
        return clean_s.index[:0] # Empty index
    z = np.abs(stats.zscore(clean_s))
    return clean_s.index[z > z_thresh]

# -----------------------
# Layout & Main Execution
# -----------------------

with st.sidebar:
    st.markdown("##  Data Source")
    source_opt = st.radio("Choose source:", ["Use Sample Data", "Upload CSV"])
    
    df_raw = None
    if source_opt == "Upload CSV":
        up_file = st.file_uploader("Upload CSV", type=['csv'])
        if up_file:
            try:
                df_raw = pd.read_csv(up_file)
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        df_raw = load_sample_data()

    st.markdown("---")
    st.markdown("## âš™ï¸ Settings")
    outlier_thresh = st.slider("Outlier Sensitivity (Z-Score)", 2.0, 5.0, 3.0, 0.1)
    
    st.markdown("---")
    st.caption("Business Analyst  ")

# Main Page Logic
if df_raw is None:
    st.markdown("<div class='dashboard-header'><h1 class='main-title'>Welcome</h1><p class='sub-title'>Upload data to begin analysis</p></div>", unsafe_allow_html=True)
    st.info("ðŸ‘ˆ Please upload a CSV file or select 'Use Sample Data' in the sidebar.")
    st.stop()

# Clean Data
df = clean_and_parse_data(df_raw)
semantics = identify_semantic_columns(df)

# Header
st.markdown(f"""
<div class='dashboard-header'>
    <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div>
            <h1 class='main-title'>Business   Dashboard</h1>
            <p class='sub-title'>Analyzing {df.shape[0]:,} rows across {df.shape[1]} columns</p>
        </div>
        <div style='text-align:right;'>
            <span style='color:{ACCENT_A}; font-weight:bold;'>{len(semantics['num_cols'])}</span> Numeric &nbsp;|&nbsp; 
            <span style='color:{ACCENT_B}; font-weight:bold;'>{len(semantics['cat_cols'])}</span> Categorical
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------
# 1. KPI Section
# -----------------------
# Dynamically pick columns for KPIs
kpi_cols = semantics['num_cols'][:4] # Take first 4 numeric columns if specific ones aren't found
if semantics['target_col'] and semantics['target_col'] not in kpi_cols:
    kpi_cols.insert(0, semantics['target_col'])
    kpi_cols = kpi_cols[:4]

if kpi_cols:
    cols = st.columns(len(kpi_cols))
    for idx, col_name in enumerate(kpi_cols):
        total = df[col_name].sum()
        avg = df[col_name].mean()
        
        # Format large numbers
        if total > 1_000_000:
            fmt_total = f"{total/1_000_000:.1f}M"
        elif total > 1_000:
            fmt_total = f"{total/1_000:.1f}K"
        else:
            fmt_total = f"{total:,.0f}"

        with cols[idx]:
            st.markdown(f"""
            <div class='insight-card' style='text-align:center;'>
                <div class='metric-label'>Total {col_name}</div>
                <div class='metric-value'>{fmt_total}</div>
                <div style='font-size:12px; color:{ACCENT_A}'>Avg: {avg:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("No numeric columns found for KPIs.")

# -----------------------
# 2. Main Tabs
# -----------------------
tab_viz, tab_data, tab_insight = st.tabs(["ðŸ“ˆ Visualizations", "ðŸ“„ Data & Stats", "ðŸ§  Automated Insights"])

with tab_viz:
    col_viz_1, col_viz_2 = st.columns([1, 3])
    
    with col_viz_1:
        st.markdown("**Chart Controls**")
        
        chart_type = st.selectbox("Chart Type", ["Time Series", "Bar / Category", "Scatter / Relationship", "Histogram"])
        
        # Smart Defaults based on Semantics
        default_y = semantics['target_col'] if semantics['target_col'] else (semantics['num_cols'][0] if semantics['num_cols'] else None)
        default_x_cat = semantics['cat_cols'][0] if semantics['cat_cols'] else None
        
        y_axis = st.selectbox("Y-Axis (Numeric)", semantics['num_cols'], index=semantics['num_cols'].index(default_y) if default_y else 0)
        
        x_axis = None
        if chart_type in ["Bar / Category", "Scatter / Relationship"]:
            options = semantics['cat_cols'] if chart_type == "Bar / Category" else semantics['num_cols']
            if options:
                x_axis = st.selectbox("X-Axis", options, index=0)
            else:
                st.warning(f"No columns suitable for {chart_type}")
        
        color_dim = st.selectbox("Color By (Optional)", ["None"] + semantics['cat_cols'])
        
    with col_viz_2:
        if not y_axis:
            st.info("No numeric data to visualize.")
        else:
            color_arg = None if color_dim == "None" else color_dim
            
            if chart_type == "Time Series":
                if semantics['date_col']:
                    # Resample logic
                    agg_df = df.set_index(semantics['date_col']).resample('W')[y_axis].sum().reset_index()
                    fig = px.line(agg_df, x=semantics['date_col'], y=y_axis, title=f"Weekly {y_axis} Trend", template="plotly_dark")
                    fig.update_traces(line_color=ACCENT_A, line_width=3)
                    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No Date column detected. Try 'Bar' or 'Histogram'.")

            elif chart_type == "Bar / Category" and x_axis:
                agg_bar = df.groupby(x_axis)[y_axis].sum().reset_index().sort_values(y_axis, ascending=False).head(15)
                fig = px.bar(agg_bar, x=x_axis, y=y_axis, color=x_axis if not color_arg else color_arg, 
                             template="plotly_dark", color_discrete_sequence=CHART_COLORS)
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Scatter / Relationship" and x_axis:
                # Sample down if too large for scatter
                plot_df = df if len(df) < 2000 else df.sample(2000)
                fig = px.scatter(plot_df, x=x_axis, y=y_axis, color=color_arg, 
                                 template="plotly_dark", color_discrete_sequence=CHART_COLORS, opacity=0.7)
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Histogram":
                fig = px.histogram(df, x=y_axis, nbins=30, color=color_arg,
                                   template="plotly_dark", color_discrete_sequence=CHART_COLORS)
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.markdown("### Raw Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("### Numeric Statistics")
        if semantics['num_cols']:
            st.dataframe(df[semantics['num_cols']].describe().T.style.format("{:.2f}"))
    with col_d2:
        st.markdown("### Missing Values")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.dataframe(missing, use_container_width=True)
        else:
            st.success("No missing values found.")

with tab_insight:
    st.markdown("###  Auto-Generated Analysis")
    
    # 1. Correlation Analysis
    if len(semantics['num_cols']) > 1:
        corr_matrix = df[semantics['num_cols']].corr()
        
        # Find high correlations
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.5:
                    pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], val))
        
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        st.markdown(f"<div class='insight-card'><h4>ðŸ”— Key Correlations</h4>", unsafe_allow_html=True)
        if pairs:
            for c1, c2, v in pairs[:5]:
                strength = "Positive" if v > 0 else "Negative"
                st.markdown(f"- **{c1}** and **{c2}**: Strong {strength} correlation ({v:.2f})")
        else:
            st.write("No strong linear correlations detected.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 2. Outlier Analysis
    st.markdown(f"<div class='insight-card'><h4>âš ï¸ Anomaly Detection</h4>", unsafe_allow_html=True)
    outlier_found = False
    for col in semantics['num_cols'][:5]: # check top 5 numeric
        idxs = get_outliers(df[col], z_thresh=outlier_thresh)
        if len(idxs) > 0:
            outlier_found = True
            st.markdown(f"- **{col}**: {len(idxs)} potential anomalies detected (Values > {outlier_thresh} Z-score)")
            
    if not outlier_found:
        st.write("No significant statistical outliers detected based on current threshold.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Trend Analysis (Simple Slope)
    if semantics['date_col'] and semantics['target_col']:
        st.markdown(f"<div class='insight-card'><h4>ðŸ“ˆ Trend Analysis</h4>", unsafe_allow_html=True)
        temp = df.dropna(subset=[semantics['date_col'], semantics['target_col']])
        # Aggregate to weekly to smooth noise
        trend_data = temp.set_index(semantics['date_col']).resample('W')[semantics['target_col']].sum()
        
        if len(trend_data) > 2:
            x = np.arange(len(trend_data))
            y = trend_data.values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            direction = "Increasing" if slope > 0 else "Decreasing"
            sig = "Statistically Significant" if p_value < 0.05 else "Not Significant"
            
            st.markdown(f"""
            - **Direction:** {direction}
            - **Confidence:** {sig} (p={p_value:.4f})
            - **Average Change:** {slope:,.2f} per week
            """)
        else:
            st.write("Not enough time-series data to calculate trends.")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(" Business Analyst Dashboard | Built with Streamlit & Plotly")