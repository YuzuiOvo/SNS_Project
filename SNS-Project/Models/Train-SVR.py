import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR

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

# Create the Support Vector Regression (SVR) model
svr_model = SVR(kernel='rbf', C=10000, gamma='auto')

# Train the model
svr_model.fit(X_train, y_train)

# Make predictions
svr_y_pred = svr_model.predict(X_test)

# Evaluate model performance
svr_mse = mean_squared_error(y_test, svr_y_pred)
svr_r2 = r2_score(y_test, svr_y_pred)

print(f"SVR Mean Squared Error: {svr_mse}")
print(f"SVR R² Score: {svr_r2}")

# Visualize the prediction results
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, color='blue', label='Actual Gold Price')
plt.plot(y_test.index, svr_y_pred, color='red', linestyle='dashed', label='Predicted Gold Price')
plt.title('Actual vs Predicted Gold Price')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.legend()
plt.show()
