import pandas as pd

# Load merged dataset
df = pd.read_csv("../data/merged_data.csv")

# Convert date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])
df["day_of_week"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday
df["month"] = df["Date"].dt.month  # 1=Jan, 12=Dec

# Calculate moving averages
df["Gold_MA7"] = df["Close_x"].rolling(window=7).mean()
df["Gold_MA30"] = df["Close_x"].rolling(window=30).mean()

# Create lag features for DXY index
df["DXY_Lag1"] = df["Close_y"].shift(1)
df["DXY_Lag7"] = df["Close_y"].shift(7)

# Remove rows with NaN values (due to moving average calculation)
df.dropna(inplace=True)

# Save the final feature dataset
df.to_csv("../data/final_dataset.csv", index=False)

print("✅ Feature engineering complete. Data saved as 'final_dataset.csv'.")
