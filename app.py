import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Walmart Sales Forecasting", layout="wide")
st.title("🛒 Walmart M5 - Sales Demand Forecasting")
st.markdown("**ML-powered sales prediction using LightGBM**")

# Data Creation
@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=365)
    stores = ['CA_1', 'CA_2', 'TX_1']
    items = ['ITEM_001', 'ITEM_002', 'ITEM_003']
    rows = []
    for store in stores:
        for item in items:
            for date in dates:
                rows.append({
                    'date': date,
                    'store': store,
                    'item': item,
                    'sales': np.random.poisson(10)
                })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df = df.sort_values(['store', 'item', 'date'])
    df['lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
    df['lag_7'] = df.groupby(['store', 'item'])['sales'].shift(7)
    df['rolling_mean_7'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    df = df.dropna()
    return df

# Model Training
@st.cache_resource
def train_model(df):
    features = ['day_of_week', 'month', 'day', 'week', 'lag_1', 'lag_7', 'rolling_mean_7']
    X = df[features]
    y = df['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, X_test, y_test, y_pred, rmse

df = load_data()
model, X_test, y_test, y_pred, rmse = train_model(df)

# Sidebar filters
st.sidebar.header("Filters")
selected_store = st.sidebar.selectbox("Select Store", df['store'].unique())
selected_item = st.sidebar.selectbox("Select Item", df['item'].unique())

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Model RMSE", f"{rmse:.3f}")
col3.metric("Stores", df['store'].nunique())

st.markdown("---")

# Chart 1 - Actual vs Predicted
st.subheader("📈 Actual vs Predicted Sales")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(y_test.values[:60], label='Actual', color='blue')
ax1.plot(y_pred[:60], label='Predicted', color='orange')
ax1.set_xlabel("Days")
ax1.set_ylabel("Sales")
ax1.legend()
st.pyplot(fig1)

# Chart 2 - Store wise sales
st.subheader("🏪 Sales by Store")
filtered = df[df['store'] == selected_store]
store_sales = filtered.groupby('date')['sales'].sum().reset_index()
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(store_sales['date'], store_sales['sales'], color='green')
ax2.set_xlabel("Date")
ax2.set_ylabel("Total Sales")
ax2.set_title(f"Daily Sales - {selected_store}")
st.pyplot(fig2)

# Chart 3 - Feature Importance
st.subheader("🔍 Feature Importance")
fig3, ax3 = plt.subplots(figsize=(8, 4))
lgb.plot_importance(model, ax=ax3, max_num_features=7)
st.pyplot(fig3)

st.success(f"Model trained successfully! RMSE: {rmse:.3f}")