import pandas as pd
from sklearn.model_selection import train_test_split

# Load the final dataset
df = pd.read_csv("../data/final_dataset.csv")

# Select features and target variable
features = ["Gold_MA7", "Gold_MA30", "DXY_Lag1", "DXY_Lag7", "day_of_week", "month"]
X = df[features]  # Feature set
y = df["Close_x"]  # Target: Gold closing price

# Split dataset (80% train, 20% test, keeping chronological order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Save train and test sets
X_train.to_csv("../data/X_train.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

print("✅ Dataset split complete. Train & test sets saved.")
