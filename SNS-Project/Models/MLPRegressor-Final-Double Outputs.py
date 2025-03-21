import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from datetime import datetime

# 1. Load the dataset
df = pd.read_csv("D:/cursor/Project/SNS-Project/Data/merged_data.csv")

# 2. Data preprocessing
# Convert the date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Extract time features (day of the week and month)
df["day_of_week"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday
df["month"] = df["Date"].dt.month  # 1=Jan, 12=Dec

# Calculate the lagged features of the dollar index (1-day, 7-day, 30-day, and 60-day lag)
df["DXY_Lag1"] = df["Close_y"].shift(1)  # 1-day lag
df["DXY_Lag7"] = df["Close_y"].shift(7)  # 7-day lag
df["DXY_Lag30"] = df["Close_y"].shift(30)  # 30-day lag
df["DXY_Lag60"] = df["Close_y"].shift(60)  # 60-day lag

# Calculate moving averages for additional features
df["DXY_MA30"] = df["Close_y"].rolling(window=30).mean()  # 30-day moving average
df["DXY_MA60"] = df["Close_y"].rolling(window=60).mean()  # 60-day moving average

# Remove NaN values (due to lagged features and rolling window)
df.dropna(inplace=True)

# Select feature columns and target columns for dollar index prediction
X_dxy = df[["DXY_Lag1", "DXY_Lag7", "DXY_Lag30", "DXY_Lag60", "DXY_MA30", "DXY_MA60", "day_of_week", "month"]]
y_dxy = df["Close_y"]  # Dollar index column
dates = df["Date"]  # Save the date column

# Normalize the data for dollar index prediction
scaler_dxy = MinMaxScaler(feature_range=(0, 1))
y_dxy_scaled = scaler_dxy.fit_transform(y_dxy.values.reshape(-1, 1))

# Split the data into training and testing sets for dollar index prediction
X_train_dxy, X_test_dxy, y_train_dxy, y_test_dxy = train_test_split(
    X_dxy, y_dxy_scaled, test_size=0.2, shuffle=False)

# 3. Train the RandomForest and MLP models to predict the dollar index
rf_dxy_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_dxy_model.fit(X_train_dxy, y_train_dxy)

mlp_dxy_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp_dxy_model.fit(X_train_dxy, y_train_dxy)

# 4. Function to predict the future dollar index using both models
def predict_dxy_rf(model, X_input):
    dxy_pred_scaled = model.predict(X_input)
    return scaler_dxy.inverse_transform(dxy_pred_scaled.reshape(-1, 1))  # Inverse transform to get the original scale

def predict_dxy_mlp(model, X_input):
    dxy_pred_scaled = model.predict(X_input)
    return scaler_dxy.inverse_transform(dxy_pred_scaled.reshape(-1, 1))  # Inverse transform to get the original scale

# 5. Use both models to predict the future dollar index for a user input date
print("Please enter a future date (format: YYYY-MM-DD):")
future_date_input = input()  # Get the user's input date

# Convert the user's input string to datetime type
future_date = datetime.strptime(future_date_input, "%Y-%m-%d")

# Calculate time-related features: day_of_week and month
future_day_of_week = future_date.weekday()  # Get the day of the week, 0=Monday, 6=Sunday
future_month = future_date.month  # Get the month

# Create the feature set for the future date (assuming lag features will be 0 as placeholders)
future_features = pd.DataFrame([[0, 0, 0, 0, 0, 0, future_day_of_week, future_month]], 
                               columns=["DXY_Lag1", "DXY_Lag7", "DXY_Lag30", "DXY_Lag60", "DXY_MA30", "DXY_MA60", "day_of_week", "month"])

# Predict the future dollar index using both models (RandomForest and MLP)
future_dxy_rf = predict_dxy_rf(rf_dxy_model, future_features)
future_dxy_mlp = predict_dxy_mlp(mlp_dxy_model, future_features)

# Combine both predictions by averaging them (or any other method like weighted average)
future_dxy_pred = (future_dxy_rf + future_dxy_mlp) / 2

print(f"The predicted future dollar index (averaged) is: {future_dxy_pred[0][0]}")

# 6. Train RandomForest and MLP models to predict the gold price
# Select feature columns and target column for gold price prediction
X_gold = df[["DXY_Lag1", "DXY_Lag7", "DXY_Lag30", "DXY_Lag60", "DXY_MA30", "DXY_MA60", "day_of_week", "month"]]
y_gold = df["Close_x"]  # Gold price column

# Normalize the gold price data
scaler_gold = MinMaxScaler(feature_range=(0, 1))
y_gold_scaled = scaler_gold.fit_transform(y_gold.values.reshape(-1, 1))

# Split the data into training and testing sets for gold price prediction
X_train_gold, X_test_gold, y_train_gold, y_test_gold = train_test_split(
    X_gold, y_gold_scaled, test_size=0.2, shuffle=False)

# Train the RandomForest and MLP models to predict the gold price
rf_gold_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_gold_model.fit(X_train_gold, y_train_gold)

mlp_gold_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp_gold_model.fit(X_train_gold, y_train_gold)

# 7. Predict gold price using both models (RandomForest and MLP)
rf_gold_pred = rf_gold_model.predict(future_features)
mlp_gold_pred = mlp_gold_model.predict(future_features)

# Combine both predictions (for example, by averaging)
gold_pred_scaled = (rf_gold_pred + mlp_gold_pred) / 2

# Inverse transform the scaled prediction to get the original scale of the gold price
gold_pred = scaler_gold.inverse_transform(gold_pred_scaled.reshape(-1, 1))

print(f"The predicted future gold price is: {gold_pred[0][0]} USD")
