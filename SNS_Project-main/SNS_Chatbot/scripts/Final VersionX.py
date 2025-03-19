import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import GRU, Dense 
from tensorflow.keras.optimizers import Adam 
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# 1. Load the dataset
df = pd.read_csv("D:/cursor/Project/SNS_Project/SNS_Chatbot/Data/merged_data.csv")

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

# Reshape the features to be compatible with GRU input (3D input)
X_train_dxy_gru = np.array(X_train_dxy).reshape((X_train_dxy.shape[0], 1, X_train_dxy.shape[1]))
X_test_dxy_gru = np.array(X_test_dxy).reshape((X_test_dxy.shape[0], 1, X_test_dxy.shape[1]))

# 3. Create the GRU model for dollar index prediction
gru_dxy_model = Sequential()
gru_dxy_model.add(GRU(64, activation='relu', input_shape=(X_train_dxy_gru.shape[1], X_train_dxy_gru.shape[2])))
gru_dxy_model.add(Dense(1))  # Output layer for predicting the dollar index

# Compile the model
gru_dxy_model.compile(optimizer=Adam(), loss='mean_squared_error')

# 4. Train the GRU model
gru_dxy_model.fit(X_train_dxy_gru, y_train_dxy, epochs=100, batch_size=32, validation_data=(X_test_dxy_gru, y_test_dxy), verbose=1)

# 5. Function to predict the future dollar index using the GRU model
def predict_dxy_gru(model, X_input):
    dxy_pred_scaled = model.predict(X_input)
    return scaler_dxy.inverse_transform(dxy_pred_scaled)  # Inverse transform to get the original scale

# 6. Ask user for future date after training completes
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

# Reshape the future features for GRU input, Automatically ADAPTS the time step
future_features = future_features.reshape(1, 30, 1)  # 30 时间步


# Predict the future dollar index using the GRU model
future_dxy_gru = predict_dxy_gru(gru_dxy_model, future_features_gru)

print(f"The predicted future dollar index (GRU model) is: {future_dxy_gru[0][0]}")

# 7. Train RandomForest and MLP models to predict the gold price (with 2D input)
# Convert the features to 2D (for RandomForest and MLP)
X_dxy_2d = X_dxy.values  # Convert to 2D numpy array (without column names)

# Normalize the gold price data
scaler_gold = MinMaxScaler(feature_range=(0, 1))
y_gold = df["Close_x"]  # Gold price column
y_gold_scaled = scaler_gold.fit_transform(y_gold.values.reshape(-1, 1))

# Fix DataConversionWarning by using ravel() to convert y to a 1D array
y_gold_scaled = y_gold_scaled.ravel()

# Split the data into training and testing sets for gold price prediction
X_train_gold, X_test_gold, y_train_gold, y_test_gold = train_test_split(
    X_dxy_2d, y_gold_scaled, test_size=0.2, shuffle=False)

# Train the RandomForest and MLP models to predict the gold price (now 2D data)
rf_gold_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_gold_model.fit(X_train_gold, y_train_gold)

mlp_gold_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp_gold_model.fit(X_train_gold, y_train_gold)

# 8. Add the predicted future dollar index as a feature for gold price prediction
future_features['predicted_dxy_future'] = future_dxy_gru[0][0]

# Since RandomForest and MLP expect the same 8 features as training, 
# we should only use the original 8 features for gold price prediction
# Extract only the 8 relevant features (without 'predicted_dxy_future') for prediction
future_features_8 = future_features[["DXY_Lag1", "DXY_Lag7", "DXY_Lag30", "DXY_Lag60", "DXY_MA30", "DXY_MA60", "day_of_week", "month"]]

# Use this new feature set in RandomForest and MLP models
rf_gold_pred = rf_gold_model.predict(future_features_8.values)  # Ensure future_features now only has 8 features
mlp_gold_pred = mlp_gold_model.predict(future_features_8.values)

# 9. Combine both predictions (for example, by averaging)
gold_pred_scaled = (rf_gold_pred + mlp_gold_pred) / 2

# Inverse transform the scaled prediction to get the original scale of the gold price
gold_pred = scaler_gold.inverse_transform(gold_pred_scaled.reshape(-1, 1))

print(f"The predicted future gold price is: {gold_pred[0][0]} USD")
