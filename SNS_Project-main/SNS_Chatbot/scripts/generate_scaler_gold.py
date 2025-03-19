import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取数据
data = pd.read_csv("D:/cursor/Project/SNS_Project-main/SNS_Chatbot/Data/gold_prices.csv")

# 确保数据无 NaN 值
data = data.dropna()

# 归一化多个特征（如果 LSTM 需要多个输入）
features = ["Close", "Open", "High", "Low"]  # 确保选取所有相关特征
scaler_gold = MinMaxScaler()

gold_values = data[features].values  # 提取多个特征
scaler_gold.fit(gold_values)  # 训练 `MinMaxScaler`

# 确保存储路径正确
scaler_path = "D:/cursor/Project/SNS_Project-main/Models/scaler_gold.pkl"

# 保存 scaler
with open(scaler_path, "wb") as f:
    pickle.dump(scaler_gold, f)

print(f"Scaler saved successfully at {scaler_path}")
