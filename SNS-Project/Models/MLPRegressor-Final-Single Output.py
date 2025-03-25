import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Load the dataset
df = pd.read_csv("D:/cursor/Project/SNS-Project/Data/merged_data.csv")

# 2. Data preprocessing
# Convert the 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Extract time-related features (day of the week and month)
df["day_of_week"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday
df["month"] = df["Date"].dt.month  # 1=Jan, 12=Dec

# Calculate lag features for the Dollar Index (1-day and 7-day lags)
df["DXY_Lag1"] = df["Close_y"].shift(1)  # 1-day lag
df["DXY_Lag7"] = df["Close_y"].shift(7)  # 7-day lag

# Drop NaN values (generated by the lag features)
df.dropna(inplace=True)

# Select feature columns and target column
X = df[["DXY_Lag1", "DXY_Lag7", "day_of_week", "month"]]
y = df["Close_x"]
dates = df["Date"]  # Save the 'Date' column

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, shuffle=False)

# 4. Train a neural network model
mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# 5. Make predictions using the neural network
mlp_predictions = mlp.predict(X_test)

# 6. Train a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(mlp_predictions.reshape(-1, 1), y_test)

# 7. Make predictions using the Random Forest model
rf_predictions = rf.predict(mlp_predictions.reshape(-1, 1))

# 8. Evaluate the model's performance
mse_rf = mean_squared_error(y_test, rf_predictions)
print(f"Random Forest MSE: {mse_rf}")

# Visualize the predicted vs actual results
plt.figure(figsize=(10, 6))
plt.plot(dates_test, y_test.values, label="Actual Gold Prices")
plt.plot(dates_test, rf_predictions, label="Predicted Gold Prices", linestyle="--")
plt.legend()
plt.title("Predicted vs Actual Gold Prices")
plt.xlabel("Date")
plt.ylabel("Gold Price (USD)")
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.show()

# 9. Predict future gold prices and the Dollar Index
# Allow the user to input a future date (e.g., 2023-12-01)
print("Please enter a future date (format: YYYY-MM-DD):")
future_date_input = input()  # Get user input for the date

# Convert the user input string to a datetime object
future_date = datetime.strptime(future_date_input, "%Y-%m-%d")

# Calculate features based on the input date: day_of_week and month
future_day_of_week = future_date.weekday()  # Get the day of the week, 0=Monday, 6=Sunday
future_month = future_date.month  # Get the month

# Assume future DXY_Lag1 and DXY_Lag7 are the same as historical data, or the user provides future Dollar Index input
# Here we assume a future Dollar Index value is provided by the user or predicted based on historical trends.
# For example, the user may input the predicted future Dollar Index value.
print("Please enter the predicted Dollar Index for the future date (e.g., 103.5):")
future_dxy = float(input())  # Get the user's input for the future Dollar Index

# Create a DataFrame with the user's input for features (assuming DXY_Lag1 and DXY_Lag7 are the same future Dollar Index value)
future_features = pd.DataFrame([[future_dxy, future_dxy, future_day_of_week, future_month]], 
                               columns=["DXY_Lag1", "DXY_Lag7", "day_of_week", "month"])

# Use the neural network to predict the gold price
mlp_future_pred = mlp.predict(future_features)  # Neural network prediction of the gold price

# Use the Random Forest to predict the gold price
rf_future_pred = rf.predict(mlp_future_pred.reshape(-1, 1))  # Random Forest prediction of the gold price

# Output the prediction result
print(f"Based on the input future date {future_date_input} and Dollar Index {future_dxy}, the predicted gold price is: {rf_future_pred[0]} USD")
