#!/usr/bin/env python
# coding: utf-8

# In[24]:


#=============================================================
#Task 7 — Deployment: Interactive Dashboard using Streamlit
#=============================================================
#Import Libraries
#=============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Superstore Sales Dataset
df = pd.read_csv(r"D:\Project\SalesForecasting_[Rohit Kumar Prajapati]\train.csv")
# Display first five rows
df.head()


# In[16]:


# ==========================================
# Task 7 - Streamlit Dashboard
# ==========================================

import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(
    page_title="Sales Overview Dashboard",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Sales Overview Dashboard")
# Load Dataset
df = pd.read_csv(r"D:\Project\SalesForecasting_[Rohit Kumar Prajapati]\train.csv")
# Convert Date
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
# Create Year and Month Columns
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.strftime("%Y-%m")
# Sidebar Filters
st.sidebar.header("Filters")
region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)
category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)
filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category))
]
# Total Sales by Year
st.subheader("Total Sales by Year")
year_sales = (
    filtered_df.groupby("Year")["Sales"]
    .sum()
    .reset_index()
)
fig1 = px.bar(
    year_sales,
    x="Year",
    y="Sales",
    color="Sales",
    text_auto=".2s"
)
st.plotly_chart(fig1, width="stretch")
# Monthly Sales Trend
st.subheader("Monthly Sales Trend")
monthly_sales = (
    filtered_df.groupby("Month")["Sales"]
    .sum()
    .reset_index()
)
fig2 = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    markers=True
)
st.plotly_chart(fig2, width="stretch")
st.success("Dashboard Loaded Successfully")


# In[28]:


#=====================================================
#Model 3 — XGBoost for Time Series (ML-based Approach)
#=====================================================


# -----------------------------------------
# Import Required Libraries
# -----------------------------------------
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error)
import warnings
warnings.filterwarnings("ignore")

# Prepare Monthly Sales Data

df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

monthly_sales = (
    df.set_index("Order Date")["Sales"]
    .resample("M")
    .sum()
    .reset_index()
)
monthly_sales.columns = ["Date", "Sales"]

# Rename columns
monthly_sales.columns = ["Date", "Sales"]
monthly_sales.head()

# Create Lag Features

# Previous month's sales
monthly_sales["Lag_1"] = monthly_sales["Sales"].shift(1)
# Sales from two months ago
monthly_sales["Lag_2"] = monthly_sales["Sales"].shift(2)
# Sales from three months ago
monthly_sales["Lag_3"] = monthly_sales["Sales"].shift(3)

# Create Rolling Mean Feature

# Three-month moving average
monthly_sales["Rolling_Mean_3"] = (
    monthly_sales["Sales"]
    .rolling(window=3)
    .mean()
)
# Create Time Features
# Month
monthly_sales["Month"] = monthly_sales["Date"].dt.month
# Quarter
monthly_sales["Quarter"] = monthly_sales["Date"].dt.quarter

# Create Season Feature
def season(month):
    if month in [12,1,2]:
        return 1

    elif month in [3,4,5]:
        return 2

    elif month in [6,7,8]:
        return 3

    else:
        return 4
monthly_sales["Season"] = monthly_sales["Month"].apply(season)

# Remove rows with NaN values
monthly_sales.dropna(inplace=True)
monthly_sales.head()

# Define Features and Target
X = monthly_sales[["Lag_1","Lag_2","Lag_3","Rolling_Mean_3","Month","Quarter","Season"]]
y = monthly_sales["Sales"]

# Split Data
X_train = X.iloc[:-3]
X_test = X.iloc[-3:]
y_train = y.iloc[:-3]
y_test = y.iloc[-3:]

# Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=3,random_state=42)
xgb_model.fit(X_train,y_train)

# Predict Sales
prediction = xgb_model.predict(X_test)
print(prediction)

# Plot Actual vs Predicted
plt.figure(figsize=(12,6))

plt.plot(y_test.index,y_test,marker="o",label="Actual")
plt.plot(y_test.index,prediction,marker="o",label="Predicted")
plt.title("XGBoost Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.savefig("chart-11.png")
plt.show()

# Evaluate XGBoost Model
mae_xgb = mean_absolute_error(y_test,prediction)
rmse_xgb = np.sqrt(mean_squared_error(y_test,prediction))
mape_xgb = np.mean(np.abs((y_test-prediction)/y_test))*100

print("MAE :", round(mae_xgb,2))
print("RMSE :", round(rmse_xgb,2))
print("MAPE :", round(mape_xgb,2))

# Forecast Table
forecast_table = pd.DataFrame({"Actual Sales":y_test,"Predicted Sales":prediction})
forecast_table



# In[29]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae_xgb = mean_absolute_error(y_test, prediction)
rmse_xgb = np.sqrt(mean_squared_error(y_test, prediction))

print("MAE:", mae_xgb)
print("RMSE:", rmse_xgb)


# In[30]:


# ==========================================
# PAGE 2 - Forecast Explorer
# ==========================================

st.header("📈 Forecast Explorer")
# Select forecast type
forecast_type = st.selectbox(
    "Select Forecast Type",
    ["Category", "Region"]
)
# Select Category or Region
if forecast_type == "Category":
    selected = st.selectbox(
        "Select Category",
        df["Category"].unique()
    )
    filtered = df[df["Category"] == selected]
else:
    selected = st.selectbox(
        "Select Region",
        df["Region"].unique()
    )
    filtered = df[df["Region"] == selected]
# Forecast horizon
months = st.slider(
    "Forecast Horizon (Months)",
    min_value=1,
    max_value=3,
    value=3
)
# Monthly sales
forecast_data = (
    filtered
    .groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
    .sum()
    .reset_index()
)
# -----------------------------
# Dummy Forecast
# Replace with your best model
# -----------------------------
last_value = forecast_data["Sales"].iloc[-1]
future_sales = []
for i in range(months):
    future_sales.append(last_value)

future_dates = pd.date_range(
    forecast_data["Order Date"].max() + pd.offsets.MonthBegin(),
    periods=months,
    freq="M"
)
forecast_df = pd.DataFrame({
    "Order Date": future_dates,
    "Forecast": future_sales
})
# Plot
fig = px.line(
    forecast_data,
    x="Order Date",
    y="Sales",
    title=f"{selected} Sales Forecast"
)
fig.add_scatter(
    x=forecast_df["Order Date"],
    y=forecast_df["Forecast"],
    mode="lines+markers",
    name="Forecast"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Model Accuracy
# -----------------------------
st.subheader("Model Performance")
st.metric("MAE", round(mae_xgb,2))
st.metric("RMSE", round(rmse_xgb,2))


# In[31]:


'''Page 3 — Anomaly Report
Display the anomaly chart from Task 5
List detected anomaly dates in a table with their sales values
'''
# ==========================================
# PAGE 3 - Anomaly Report
# ==========================================

st.header("🚨 Anomaly Report")
from sklearn.ensemble import IsolationForest
# Weekly Sales
weekly_sales = (
    df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
      .sum()
      .reset_index()
)
# Isolation Forest
model = IsolationForest(
    contamination=0.05,
    random_state=42
)
weekly_sales["Anomaly"] = model.fit_predict(
    weekly_sales[["Sales"]]
)
# Extract anomalies
anomalies = weekly_sales[
    weekly_sales["Anomaly"] == -1
]
# ----------------------------
# Plot
# ----------------------------
fig = px.line(
    weekly_sales,
    x="Order Date",
    y="Sales",
    title="Weekly Sales with Detected Anomalies"
)
fig.add_scatter(
    x=anomalies["Order Date"],
    y=anomalies["Sales"],
    mode="markers",
    marker=dict(color="red", size=10),
    name="Anomaly"
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Anomaly Table
# ----------------------------
st.subheader("Detected Anomalies")
st.dataframe(
    anomalies[
        [
            "Order Date",
            "Sales"
        ]],
    use_container_width=True
)


# In[47]:


'''Label each cluster meaningfully, for example:
    High Volume, Stable Demand
    Low Volume, High Volatility
    Growing Demand
    Declining Demand
'''
from sklearn.cluster import KMeans



'''Aggregate data at the product sub-category level with features like:
     Total sales volume
     Sales growth rate (year-over-year)
     Sales volatility (standard deviation of monthly sales)
     Average order value
'''
print("----------Total sales volume-------------\n")
# Calculate total sales for each sub-category
total_sales = df.groupby("Sub-Category")["Sales"].sum().reset_index()
# Rename column
total_sales.rename(columns={"Sales": "Total Sales Volume"}, inplace=True)
# Display result
total_sales


# Convert Order Date to datetime (only if it isn't already)
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# Create Year column
df["Year"] = df["Order Date"].dt.year

# Check that it was created
print(df[["Order Date", "Year"]].head())



print("\n----------Sales growth rate (year-over-year)-------------\n")

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])
# Calculate yearly sales
growth = df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack()
# Calculate growth rate
growth["Growth Rate (%)"] = ((growth.iloc[:, -1] - growth.iloc[:, 0]) / growth.iloc[:, 0]) * 100
# Display growth rate
growth[["Growth Rate (%)"]]


print("\n----------Sales volatility (standard deviation of monthly sales)-------------\n")

# Calculate monthly sales volatility
volatility = (
    df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="M")])["Sales"]
      .sum()
      .groupby("Sub-Category")
      .std()
      .reset_index(name="Sales Volatility")
)
# Display result
volatility


print("\n----------Average order value-------------\n")

# Calculate average order value
average_order = (
    df.groupby("Sub-Category")["Sales"]
      .mean()
      .reset_index(name="Average Order Value")
)
# Display result
average_order



# Merge all features into one table
cluster_data = (
    total_sales
    .merge(growth[["Growth Rate (%)"]], left_on="Sub-Category", right_index=True)
    .merge(volatility, on="Sub-Category")
    .merge(average_order, on="Sub-Category")
)

# Display final dataset
cluster_data

#Apply K-Means Clustering to segment products into demand groups
# Apply K-Means Clustering

# Select features
X = cluster_data[
    [
        "Total Sales Volume",
        "Growth Rate (%)",
        "Sales Volatility",
        "Average Order Value"
    ]]
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
# Assign cluster labels
cluster_data["Cluster"] = kmeans.fit_predict(X_scaled)
# Display the clustered data
print(cluster_data)



# Select features
X = cluster_data[
    [
        "Total Sales Volume",
        "Growth Rate (%)",
        "Sales Volatility",
        "Average Order Value"
    ]]
#---------------------------------------
# Label Each Cluster
#--------------------------------------
# Assign meaningful names to clusters
cluster_labels = {
    0: "High Volume, Stable Demand",
    1: "Low Volume, High Volatility",
    2: "Growing Demand",
    3: "Declining Demand"
}
# Create a new column with demand segment names
cluster_data["Demand Segment"] = cluster_data["Cluster"].map(cluster_labels)
# Display the results
print(cluster_data[["Sub-Category", "Cluster", "Demand Segment"]])


# In[49]:


#=====================================================================================
#Plot clusters using a 2D scatter plot (use PCA to reduce to 2 dimensions if needed)
#=====================================================================================
#import library
from sklearn.decomposition import PCA
# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce features to 2 dimensions
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)
# Add PCA components to dataframe
cluster_data["PCA1"] = pca_data[:, 0]
cluster_data["PCA2"] = pca_data[:, 1]
# Plot clusters
plt.figure(figsize=(10,6))
plt.scatter(
    cluster_data["PCA1"],
    cluster_data["PCA2"],
    c=cluster_data["Cluster"],
    cmap="viridis",
    s=120,
    edgecolor="black"
)
# Add sub-category labels
for i in range(len(cluster_data)):
    plt.text(
        cluster_data["PCA1"].iloc[i],
        cluster_data["PCA2"].iloc[i],
        cluster_data["Sub-Category"].iloc[i],
        fontsize=8
    )
plt.title("Product Demand Segmentation using K-Means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.colorbar(label="Cluster")
plt.savefig("chart-16.png", dpi=300)
plt.show()



# In[50]:


# ==========================================================
# PAGE 4 : PRODUCT DEMAND SEGMENTS
# ==========================================================

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
st.header("📦 Product Demand Segments")
st.markdown("### Product Demand Clustering using K-Means")

# ----------------------------------------------------------
# PCA for 2D Visualization
# ----------------------------------------------------------
pca = PCA(n_components=2)
pca_features = pca.fit_transform(X_scaled)
cluster_data["PCA1"] = pca_features[:,0]
cluster_data["PCA2"] = pca_features[:,1]

# ----------------------------------------------------------
# Scatter Plot
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(10,6))
colors = [
    "red",
    "blue",
    "green",
    "orange"
]
for cluster in sorted(cluster_data["Cluster"].unique()):
    subset = cluster_data[
        cluster_data["Cluster"] == cluster
    ]
    ax.scatter(
        subset["PCA1"],
        subset["PCA2"],
        s=180,
        color=colors[cluster],
        edgecolor="black",
        label=subset["Demand Segment"].iloc[0]
    )
    # Add labels
    for i in subset.index:
        ax.text(
            cluster_data.loc[i,"PCA1"],
            cluster_data.loc[i,"PCA2"],
            cluster_data.loc[i,"Sub-Category"],
            fontsize=8
        )
ax.set_title("Product Demand Segments")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ----------------------------------------------------------
# Demand Segment Table
# ----------------------------------------------------------

st.markdown("## Demand Segment Table")

display_table = cluster_data[
    [
        "Sub-Category",
        "Demand Segment",
        "Total Sales Volume",
        "Growth Rate (%)",
        "Sales Volatility",
        "Average Order Value"
    ]]
st.dataframe(
    display_table,
    use_container_width=True
)
# ----------------------------------------------------------
# Cluster Summary
# ----------------------------------------------------------
st.markdown("## Cluster Summary")
summary = (
    cluster_data.groupby("Demand Segment")
    .size()
    .reset_index(name="Number of Sub-Categories")
)
st.dataframe(
    summary,
    use_container_width=True
)


# ## Recommended Stocking Strategy
# 
# **High Volume, Stable Demand**
# - Maintain high inventory.
# - Replenish stock regularly.
# 
# **Growing Demand**
# - Increase stock gradually.
# - Monitor future demand trends.
# 
# **Low Volume, High Volatility**
# - Keep minimum safety stock.
# - Order products in smaller batches.
# 
# **Declining Demand**
# - Reduce inventory.
# - Use discounts to clear existing stock.

# In[ ]:




