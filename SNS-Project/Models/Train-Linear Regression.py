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

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions using the test data
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R² (Coefficient of Determination)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Visualize the prediction results
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, color='blue', label='Actual Gold Price')
plt.plot(y_test.index, y_pred, color='red', linestyle='dashed', label='Predicted Gold Price')
plt.title('Actual vs Predicted Gold Price')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.legend()
plt.show()
