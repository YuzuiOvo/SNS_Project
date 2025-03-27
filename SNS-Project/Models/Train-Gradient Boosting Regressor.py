import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv("D:/cursor/Project/SNS-Project/Data/merged_data.csv")

# Convert the date to a numerical value (e.g., convert the date to the number of days for easier use in machine learning models)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(lambda x: x.toordinal())  # Convert the date to an integer

# Select features (DXY index and date)
X = df[['Date', 'Close_y']]  # Use Date and DXY index as features
y = df['Close_x']  # Gold price is the target variable we want to predict

# Split the data into training and testing sets, 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import GradientBoostingRegressor

# Create a gradient boosting regression model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions
gb_y_pred = gb_model.predict(X_test)

# Evaluate model performance
gb_mse = mean_squared_error(y_test, gb_y_pred)
gb_r2 = r2_score(y_test, gb_y_pred)

print(f"Gradient Boosting Mean Squared Error: {gb_mse}")
print(f"Gradient Boosting R² Score: {gb_r2}")
