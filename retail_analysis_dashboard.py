
import pandas as pd
import numpy as np
import sqlite3
from prophet import Prophet
import plotly.express as px
import streamlit as st

# =======================
# 1. Load and Clean Data
# =======================
df = pd.read_csv("retail_sales_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.dropna(inplace=True)
df = df[(df['Quantity'] > 0) & (df['Price per Unit'] > 0)]
df['Revenue'] = df['Quantity'] * df['Price per Unit']
df_original = df.copy()

# =======================
# 2. EDA & KPI Extraction
# =======================
total_revenue = df['Revenue'].sum()
total_transactions = df.shape[0]
unique_customers = df['Customer ID'].nunique()

df.set_index('Date', inplace=True)
monthly_sales = df['Revenue'].resample('M').sum().reset_index()
top_products = df.groupby('Product Category')['Revenue'].sum().sort_values(ascending=False).reset_index()
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['<18', '18-30', '30-45', '45-60', '60+'])
age_group_revenue = df.groupby('Age_Group')['Revenue'].sum().reset_index()

# =======================
# 3. Sales Forecasting
# =======================
prophet_df = monthly_sales.rename(columns={'Date': 'ds', 'Revenue': 'y'})
model = Prophet()
model.fit(prophet_df)
future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)

# =======================
# 4. SQL Integration
# =======================
conn = sqlite3.connect("retail_sales.db")
df_original.to_sql("sales_data", conn, if_exists="replace", index=False)

# =======================
# 5. Streamlit Dashboard
# =======================
st.set_page_config(layout="wide")
st.title("🛒 Retail Sales Analysis Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"${total_revenue:,.0f}")
col2.metric("Transactions", f"{total_transactions}")
col3.metric("Unique Customers", f"{unique_customers}")

fig1 = px.line(monthly_sales, x='Date', y='Revenue', title='Monthly Revenue Trend', markers=True)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(top_products, x='Product Category', y='Revenue', title='Top Product Categories by Revenue')
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.pie(age_group_revenue, names='Age_Group', values='Revenue', title='Revenue Distribution by Age Group')
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.line(forecast, x='ds', y='yhat', title='Revenue Forecast (Next 6 Months)')
st.plotly_chart(fig4, use_container_width=True)

st.caption("Powered by Python 🐍 | Prophet 🔮 | Plotly 📊 | Streamlit 🌐")
