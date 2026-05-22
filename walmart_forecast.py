import pandas as pd
import numpy as np

# Create sample Walmart data
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
df.to_csv('walmart_data.csv', index=False)
print("Dataset created successfully!")
print(df.head())
print(f"Total rows: {len(df)}")

# Feature Engineering
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['week'] = df['date'].dt.isocalendar().week.astype(int)

# Lag features - previous day sales
df = df.sort_values(['store', 'item', 'date'])
df['lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
df['lag_7'] = df.groupby(['store', 'item'])['sales'].shift(7)

# Rolling mean - average of last 7 days
df['rolling_mean_7'] = df.groupby(['store', 'item'])['sales'].transform(
    lambda x: x.shift(1).rolling(7).mean()
)

# Drop nulls
df = df.dropna()

print("Features created!")
print(df.head())
print(f"Columns: {df.columns.tolist()}")



import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare data for model
features = ['day_of_week', 'month', 'day', 'week', 'lag_1', 'lag_7', 'rolling_mean_7']
target = 'sales'

X = df[features]
y = df[target]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model trained successfully!")
print(f"RMSE: {rmse:.3f}")



import matplotlib.pyplot as plt

# Feature Importance
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=7)
plt.title('Feature Importance - Which factors affect sales most?')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("Chart saved!")

# Actual vs Predicted
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:50], label='Actual Sales', color='blue')
plt.plot(y_pred[:50], label='Predicted Sales', color='orange')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Days')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()
print("All charts saved!")