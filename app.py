# ==========================================================
# CUSTOMER SEGMENTATION DASHBOARD
# Developed by Rohit Kumar Prajapati
# ==========================================================

# -------------------- IMPORT LIBRARIES --------------------

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

page = st.sidebar.selectbox(
    "Select Dashboard",
    [
        "🏠 Home",
        "📊 Sales Overview",
        "📈 Forecast",
        "🚨 Anomaly Detection",
        "📦 Product Analysis",
        "💼 Business Insights"
    ]
)
# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("📊 Sales Forecasting Dashboard")

    st.markdown("""
Welcome to the **Sales Forecasting Dashboard**.

This dashboard provides insights into sales performance, customer segments,
regional performance, and future sales forecasting using Machine Learning.
""")

    st.divider()

    # -----------------------------
    # Data Preparation
    # -----------------------------

    df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce",
    format="mixed"
   )
    df = df.dropna(subset=["Order Date"])

    total_sales = df["Sales"].sum()

    total_orders = df["Order ID"].nunique()

    total_customers = df["Customer ID"].nunique()

    total_categories = df["Category"].nunique()

    # -----------------------------
    # KPI Cards
    # -----------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "💰 Total Sales",
        f"${total_sales:,.2f}"
    )

    c2.metric(
        "📦 Orders",
        total_orders
    )

    c3.metric(
        "👥 Customers",
        total_customers
    )

    c4.metric(
        "🛒 Categories",
        total_categories
    )

    st.divider()

    # -----------------------------
    # Dataset Preview
    # -----------------------------

    left, right = st.columns([2,1])

    with left:

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    with right:

        st.subheader("📊 Dataset Summary")

        st.write(f"Rows : {df.shape[0]}")
        st.write(f"Columns : {df.shape[1]}")

        st.write(f"Regions : {df['Region'].nunique()}")
        st.write(f"States : {df['State'].nunique()}")

        st.write(f"Products : {df['Product ID'].nunique()}")

    st.divider()

    # -----------------------------
    # Sales by Category
    # -----------------------------

    sales_category = (
        df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        sales_category,
        x="Category",
        y="Sales",
        color="Category",
        title="Sales by Category",
        text_auto=".2s"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------
    # Sales by Region
    # -----------------------------

    sales_region = (
        df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        sales_region,
        values="Sales",
        names="Region",
        hole=0.5,
        title="Regional Sales Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce",
    format="mixed"
    )

    df = df.dropna(subset=["Order Date"])
    # -----------------------------
    # Monthly Sales Trend
    # -----------------------------

    monthly_sales = (
        df.groupby(
            df["Order Date"].dt.to_period("M")
        )["Sales"]
        .sum()
        .reset_index()
    )

    monthly_sales["Order Date"] = (
        monthly_sales["Order Date"]
        .astype(str)
    )

    fig = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Sales Overview":

    st.title("📊 Sales Overview Dashboard")

    st.markdown("Analyze sales performance using interactive filters and charts.")

    st.divider()

    # ==========================
    # FILTERS
    # ==========================

    col1, col2, col3 = st.columns(3)

    with col1:
        region = st.selectbox(
            "🌍 Select Region",
            ["All"] + sorted(df["Region"].unique().tolist())
        )

    with col2:
        category = st.selectbox(
            "📦 Select Category",
            ["All"] + sorted(df["Category"].unique().tolist())
        )

    with col3:
        segment = st.selectbox(
            "👥 Select Segment",
            ["All"] + sorted(df["Segment"].unique().tolist())
        )

    filtered_df = df.copy()

    if region != "All":
        filtered_df = filtered_df[
            filtered_df["Region"] == region
        ]

    if category != "All":
        filtered_df = filtered_df[
            filtered_df["Category"] == category
        ]

    if segment != "All":
        filtered_df = filtered_df[
            filtered_df["Segment"] == segment
        ]

    st.divider()

    # ==========================
    # KPI CARDS
    # ==========================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "💰 Total Sales",
        f"${filtered_df['Sales'].sum():,.0f}"
    )

    c2.metric(
        "📦 Orders",
        filtered_df["Order ID"].nunique()
    )

    c3.metric(
        "👥 Customers",
        filtered_df["Customer ID"].nunique()
    )

    c4.metric(
        "🛒 Products",
        filtered_df["Product ID"].nunique()
    )

    st.divider()

    # ==========================
    # SALES BY CATEGORY
    # ==========================

    col1, col2 = st.columns(2)

    with col1:

        sales_cat = (
            filtered_df.groupby("Category")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            sales_cat,
            x="Category",
            y="Sales",
            color="Category",
            text_auto=".2s",
            title="Sales by Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==========================
    # SALES BY REGION
    # ==========================

    with col2:

        sales_reg = (
            filtered_df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            sales_reg,
            values="Sales",
            names="Region",
            hole=0.45,
            title="Regional Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ==========================
    # MONTHLY SALES
    # ==========================
    
    monthly = filtered_df.copy()
    
    # Order Date ko datetime me convert karo
    monthly["Order Date"] = pd.to_datetime(
    monthly["Order Date"],
    errors="coerce"
    )
    
    # Invalid dates hata do
    monthly = monthly.dropna(subset=["Order Date"])
        
    # Month column banao
    monthly["Month"] = monthly["Order Date"].dt.to_period("M").astype(str)
        
    # Monthly sales
    monthly = (
        monthly.groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================
    # TOP 10 PRODUCTS
    # ==========================

    st.subheader("🏆 Top 10 Products")

    top_products = (
        filtered_df.groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_products,
        x="Sales",
        y="Product Name",
        orientation="h",
        color="Sales",
        title="Top 10 Products"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ==========================
    # TOP STATES
    # ==========================

    st.subheader("🏙 Top 10 States")

    top_states = (
        filtered_df.groupby("State")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_states,
        x="State",
        y="Sales",
        color="Sales",
        title="Top Performing States",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ==========================
    # DATA TABLE
    # ==========================

    st.subheader("📄 Filtered Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=450
    )

# ==========================================================
# FORECAST DASHBOARD
# ==========================================================

elif page == "📈 Forecast":

    st.title("📈 Sales Forecast Dashboard")

    st.markdown("Predict future monthly sales using historical sales data.")

    st.divider()

    # -----------------------------
    # Prepare Monthly Sales
    # -----------------------------

    forecast_df = df.copy()

    forecast_df["Order Date"] = pd.to_datetime(
    forecast_df["Order Date"],
    errors="coerce",
    format="mixed"
)

forecast_df = forecast_df.dropna(subset=["Order Date"])

    monthly_sales = (
        forecast_df
        .groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
        .sum()
        .reset_index()
    )

    st.subheader("Monthly Sales")

    fig = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Historical Monthly Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------
    # Forecast Horizon
    # -----------------------------

    forecast_months = st.slider(
        "Forecast Months",
        min_value=1,
        max_value=12,
        value=3
    )

    # -----------------------------
    # Simple Forecast
    # -----------------------------

    avg_sales = monthly_sales["Sales"].tail(6).mean()

    last_date = monthly_sales["Order Date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="M"
    )

    future_df = pd.DataFrame({
        "Order Date": future_dates,
        "Forecast Sales": [avg_sales] * forecast_months
    })

    st.subheader("Forecast Results")

    st.dataframe(
        future_df,
        use_container_width=True
    )

    st.divider()

    # -----------------------------
    # Forecast Chart
    # -----------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly_sales["Order Date"],
            y=monthly_sales["Sales"],
            mode="lines+markers",
            name="Historical Sales"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_df["Order Date"],
            y=future_df["Forecast Sales"],
            mode="lines+markers",
            name="Forecast",
            line=dict(dash="dash")
        )
    )

    fig.update_layout(
        title="Historical vs Forecast Sales",
        xaxis_title="Month",
        yaxis_title="Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------
    # Forecast Summary
    # -----------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Monthly Sales",
        f"${avg_sales:,.2f}"
    )

    c2.metric(
        "Forecast Months",
        forecast_months
    )

    c3.metric(
        "Expected Sales",
        f"${future_df['Forecast Sales'].sum():,.2f}"
    )

    st.divider()

    # -----------------------------
    # Download Forecast
    # -----------------------------

    csv = future_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Forecast",
        data=csv,
        file_name="sales_forecast.csv",
        mime="text/csv"
    )

# ==========================================================
# ANOMALY DETECTION
# ==========================================================

elif page == "🚨 Anomaly Detection":

    st.title("🚨 Sales Anomaly Detection")

    st.markdown("""
Detect unusual sales patterns using **Isolation Forest**.
This helps identify abnormal sales transactions that may require investigation.
""")

    st.divider()

    from sklearn.ensemble import IsolationForest

    # ---------------------------------------
    # Prepare Data
    # ---------------------------------------

    anomaly_df = df.copy()

    anomaly_df["Order Date"] = pd.to_datetime(anomaly_df["Order Date"])

    anomaly_df = anomaly_df.sort_values("Order Date")

    # ---------------------------------------
    # Isolation Forest
    # ---------------------------------------

    model = IsolationForest(

        contamination=0.03,

        random_state=42

    )

    anomaly_df["Anomaly"] = model.fit_predict(

        anomaly_df[["Sales"]]

    )

    anomaly_df["Anomaly"] = anomaly_df["Anomaly"].replace({

        1:"Normal",

        -1:"Anomaly"

    })

    st.success("Anomaly Detection Completed Successfully")

    st.divider()

    # ---------------------------------------
    # KPI Cards
    # ---------------------------------------

    total = len(anomaly_df)

    anomalies = len(

        anomaly_df[
            anomaly_df["Anomaly"]=="Anomaly"
        ]

    )

    normal = total-anomalies

    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Total Records",

        total

    )

    c2.metric(

        "Normal Records",

        normal

    )

    c3.metric(

        "Anomalies",

        anomalies

    )

    st.divider()

    # ---------------------------------------
    # Scatter Plot
    # ---------------------------------------

    st.subheader("Sales Anomaly Visualization")

    fig = px.scatter(

        anomaly_df,

        x="Order Date",

        y="Sales",

        color="Anomaly",

        hover_data=[

            "Category",

            "Region",

            "State"

        ],

        title="Detected Sales Anomalies"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ---------------------------------------
    # Monthly Sales Trend
    # ---------------------------------------

    monthly = anomaly_df.copy()

    monthly["Month"] = monthly["Order Date"].dt.to_period("M").astype(str)

    monthly = monthly.groupby(

        "Month"

    )["Sales"].sum().reset_index()

    fig = px.line(

        monthly,

        x="Month",

        y="Sales",

        markers=True,

        title="Monthly Sales Trend"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ---------------------------------------
    # Only Anomalies
    # ---------------------------------------

    st.subheader("Detected Anomalies")

    anomaly_only = anomaly_df[

        anomaly_df["Anomaly"]=="Anomaly"

    ]

    st.dataframe(

        anomaly_only,

        use_container_width=True,

        height=400

    )

    st.divider()

    # ---------------------------------------
    # Download Report
    # ---------------------------------------

    csv = anomaly_only.to_csv(index=False)

    st.download_button(

        "📥 Download Anomaly Report",

        csv,

        file_name="anomaly_report.csv",

        mime="text/csv"

    )

# ==========================================================
# PRODUCT ANALYSIS
# ==========================================================

elif page == "📦 Product Analysis":

    st.title("📦 Product Analysis Dashboard")

    st.markdown("""
Analyze product performance across Categories, Sub-Categories,
Regions and Top Selling Products.
""")

    st.divider()

    # ==========================================
    # FILTERS
    # ==========================================

    col1,col2=st.columns(2)

    with col1:

        category=st.selectbox(

            "Select Category",

            ["All"]+sorted(df["Category"].unique().tolist())

        )

    with col2:

        region=st.selectbox(

            "Select Region",

            ["All"]+sorted(df["Region"].unique().tolist())

        )

    filtered=df.copy()

    if category!="All":

        filtered=filtered[
            filtered["Category"]==category
        ]

    if region!="All":

        filtered=filtered[
            filtered["Region"]==region
        ]

    st.divider()

    # ==========================================
    # KPI
    # ==========================================

    c1,c2,c3,c4=st.columns(4)

    c1.metric(

        "💰 Total Sales",

        f"${filtered['Sales'].sum():,.0f}"

    )

    c2.metric(

        "📦 Products",

        filtered["Product ID"].nunique()

    )

    c3.metric(

        "📂 Categories",

        filtered["Category"].nunique()

    )

    c4.metric(

        "🛍 Sub Categories",

        filtered["Sub-Category"].nunique()

    )

    st.divider()

    # ==========================================
    # CATEGORY SALES
    # ==========================================

    st.subheader("📊 Category-wise Sales")

    category_sales=filtered.groupby(

        "Category"

    )["Sales"].sum().reset_index()

    fig=px.bar(

        category_sales,

        x="Category",

        y="Sales",

        color="Category",

        text_auto=".2s"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ==========================================
    # SUBCATEGORY SALES
    # ==========================================

    st.subheader("📈 Top Sub-Categories")

    sub=filtered.groupby(

        "Sub-Category"

    )["Sales"].sum()

    sub=sub.sort_values(

        ascending=False

    ).reset_index()

    fig=px.bar(

        sub,

        x="Sales",

        y="Sub-Category",

        orientation="h",

        color="Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ==========================================
    # TOP PRODUCTS
    # ==========================================

    st.subheader("🏆 Top 10 Products")

    top_products=filtered.groupby(

        "Product Name"

    )["Sales"].sum()

    top_products=top_products.sort_values(

        ascending=False

    ).head(10).reset_index()

    fig=px.bar(

        top_products,

        x="Sales",

        y="Product Name",

        orientation="h",

        color="Sales",

        text_auto=".2s"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ==========================================
    # REGION SALES
    # ==========================================

    st.subheader("🌍 Region-wise Sales")

    region_sales=filtered.groupby(

        "Region"

    )["Sales"].sum().reset_index()

    fig=px.pie(

        region_sales,

        names="Region",

        values="Sales",

        hole=0.45

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ==========================================
    # TREEMAP
    # ==========================================

    st.subheader("🌳 Category Hierarchy")

    fig=px.treemap(

        filtered,

        path=[

            "Category",

            "Sub-Category"

        ],

        values="Sales",

        color="Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ==========================================
    # PRODUCT TABLE
    # ==========================================

    st.subheader("📋 Product Details")

    st.dataframe(

        filtered[[

            "Product Name",

            "Category",

            "Sub-Category",

            "Region",

            "Sales"

        ]],

        use_container_width=True,

        height=400

    )

    st.divider()

    # ==========================================
    # DOWNLOAD
    # ==========================================

    csv=filtered.to_csv(index=False)

    st.download_button(

        "📥 Download Product Report",

        csv,

        "product_report.csv",

        "text/csv"

    )


# ==========================================================
# BUSINESS INSIGHTS DASHBOARD
# ==========================================================

elif page == "💼 Business Insights":

    st.title("💼 Executive Business Insights")

    st.markdown(
        "Business performance summary with actionable insights and recommendations."
    )

    st.divider()

    # ===========================
    # KPI CALCULATIONS
    # ===========================

    total_sales = df["Sales"].sum()

    total_orders = df["Order ID"].nunique()

    total_customers = df["Customer ID"].nunique()

    avg_sales = df["Sales"].mean()

    best_region = (
        df.groupby("Region")["Sales"]
        .sum()
        .idxmax()
    )

    best_category = (
        df.groupby("Category")["Sales"]
        .sum()
        .idxmax()
    )

    best_product = (
        df.groupby("Product Name")["Sales"]
        .sum()
        .idxmax()
    )

    # ===========================
    # KPI CARDS
    # ===========================

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "💰 Total Revenue",
        f"${total_sales:,.0f}"
    )

    c2.metric(
        "📦 Orders",
        total_orders
    )

    c3.metric(
        "👥 Customers",
        total_customers
    )

    c4.metric(
        "📈 Avg Sales",
        f"${avg_sales:,.2f}"
    )

    st.divider()

    # ===========================
    # TOP INSIGHTS
    # ===========================

    st.subheader("📌 Key Business Insights")

    st.success(f"🏆 Best Performing Region : **{best_region}**")

    st.success(f"📦 Highest Revenue Category : **{best_category}**")

    st.success(f"⭐ Top Selling Product : **{best_product}**")

    st.info(f"💰 Total Revenue Generated : **${total_sales:,.2f}**")

    st.divider()

    # ===========================
    # REGION PERFORMANCE
    # ===========================

    st.subheader("🌍 Region-wise Sales")

    region_sales = (
        df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        color="Sales",
        text_auto=".2s",
        title="Regional Performance"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===========================
    # CATEGORY PERFORMANCE
    # ===========================

    st.subheader("📦 Category Performance")

    category_sales = (
        df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        category_sales,
        values="Sales",
        names="Category",
        hole=0.5,
        title="Sales Contribution by Category"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===========================
    # TOP PRODUCTS
    # ===========================

    st.subheader("🏆 Top 10 Products")

    top_products = (
        df.groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(
        top_products,
        use_container_width=True
    )

    st.divider()

    # ===========================
    # RECOMMENDATIONS
    # ===========================

    st.subheader("💡 Business Recommendations")

    st.markdown("""

### ✅ Recommendation 1
Increase inventory for top-selling products to avoid stock shortages.

### ✅ Recommendation 2
Focus marketing campaigns in high-performing regions to maximize revenue.

### ✅ Recommendation 3
Promote low-performing categories using discounts and bundled offers.

### ✅ Recommendation 4
Analyze seasonal demand trends to improve sales forecasting accuracy.

### ✅ Recommendation 5
Use customer segmentation and personalized marketing to improve customer retention.

""")

    st.divider()

    # ===========================
    # EXECUTIVE SUMMARY
    # ===========================

    st.subheader("📄 Executive Summary")

    st.write(f"""
- Total Revenue Generated: **${total_sales:,.2f}**
- Total Orders: **{total_orders}**
- Total Customers: **{total_customers}**
- Best Performing Region: **{best_region}**
- Highest Revenue Category: **{best_category}**
- Top Selling Product: **{best_product}**

Overall, the business is performing well. Strengthening inventory management,
regional marketing strategies, and product-level planning can further improve
sales performance and customer satisfaction.
""")

    st.divider()

    # ===========================
    # DOWNLOAD REPORT
    # ===========================

    report = pd.DataFrame({
        "Metric": [
            "Total Revenue",
            "Total Orders",
            "Total Customers",
            "Average Sales",
            "Best Region",
            "Best Category",
            "Top Product"
        ],
        "Value": [
            total_sales,
            total_orders,
            total_customers,
            avg_sales,
            best_region,
            best_category,
            best_product
        ]
    })

    csv = report.to_csv(index=False)

    st.download_button(
        "📥 Download Executive Report",
        csv,
        "Executive_Report.csv",
        "text/csv"
    )

    st.markdown("---")
    st.markdown(
        "<center><h5>📊 Sales Forecasting Dashboard | Developed by Rohit Kumar Prajapati</h5></center>",
        unsafe_allow_html=True
    )
