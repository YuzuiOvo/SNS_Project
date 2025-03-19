import pandas as pd
import matplotlib.pyplot as plt

# Load merged dataset
# df = pd.read_csv("../data/merged_data.csv")
df = pd.read_csv("D:/cursor/Project/SNS_Chatbot/Data/merged_data.csv")


# Convert date column
df["Date"] = pd.to_datetime(df["Date"])

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(12,5))

# Plot Gold Price on primary y-axis (left)
ax1.plot(df["Date"], df["Close_x"], label="Gold Price", color="gold")
ax1.set_xlabel("Date")
ax1.set_ylabel("Gold Price (USD)", color="gold")
ax1.tick_params(axis="y", labelcolor="gold")

# Create secondary y-axis (right) for DXY Index
ax2 = ax1.twinx()
ax2.plot(df["Date"], df["Close_y"], label="DXY Index", color="blue", linestyle="dashed")
ax2.set_ylabel("DXY Index", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")

# Title and grid
plt.title("Gold Price vs DXY Index")
ax1.grid(True)

# Show plot
plt.show()
