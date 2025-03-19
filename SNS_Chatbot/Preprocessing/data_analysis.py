import pandas as pd

# Load datasets
gold_df = pd.read_csv("../data/gold_prices.csv")
dxy_df = pd.read_csv("../data/dxy_index.csv")

# Convert date column to datetime format for consistency
gold_df["Date"] = pd.to_datetime(gold_df["Date"])
dxy_df["Date"] = pd.to_datetime(dxy_df["Date"])

# Check dataset structure and basic stats
print("\nGold Data Overview:")
print(gold_df.info())
print(gold_df.describe())

print("\nDXY Index Data Overview:")
print(dxy_df.info())
print(dxy_df.describe())

# Look for missing values
print("\nMissing Values:")
print("Gold:\n", gold_df.isnull().sum())
print("DXY:\n", dxy_df.isnull().sum())

# Merge datasets on date to align time series
merged_df = pd.merge(gold_df, dxy_df, on="Date", how="inner")

# Save merged dataset for further analysis
merged_df.to_csv("../data/merged_data.csv", index=False)

print("\nData check complete. Merged dataset saved.")
