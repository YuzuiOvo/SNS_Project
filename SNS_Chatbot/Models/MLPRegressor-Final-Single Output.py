import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 加载数据集
df = pd.read_csv("D:/cursor/Project/SNS_Chatbot/Data/merged_data.csv")

# 2. 数据预处理
# 转换日期列为datetime格式
df["Date"] = pd.to_datetime(df["Date"])

# 提取时间特征（星期几和月份）
df["day_of_week"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday
df["month"] = df["Date"].dt.month  # 1=Jan, 12=Dec

# 计算美元指数的滞后特征（1天和7天滞后）
df["DXY_Lag1"] = df["Close_y"].shift(1)  # 1天滞后
df["DXY_Lag7"] = df["Close_y"].shift(7)  # 7天滞后

# 删除NaN值（由于滞后特征产生）
df.dropna(inplace=True)

# 选择特征列和目标列
X = df[["DXY_Lag1", "DXY_Lag7", "day_of_week", "month"]]
y = df["Close_x"]
dates = df["Date"]  # 保存日期列

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, shuffle=False)

# 4. 使用神经网络进行训练
mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# 5. 使用神经网络进行预测
mlp_predictions = mlp.predict(X_test)

# 6. 使用随机森林进行训练
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(mlp_predictions.reshape(-1, 1), y_test)

# 7. 使用随机森林进行预测
rf_predictions = rf.predict(mlp_predictions.reshape(-1, 1))

# 8. 评估模型性能
mse_rf = mean_squared_error(y_test, rf_predictions)
print(f"Random Forest MSE: {mse_rf}")

# 可视化预测结果与实际结果
plt.figure(figsize=(10, 6))
plt.plot(dates_test, y_test.values, label="Actual Gold Prices")
plt.plot(dates_test, rf_predictions, label="Predicted Gold Prices", linestyle="--")
plt.legend()
plt.title("Predicted vs Actual Gold Prices")
plt.xlabel("Date")
plt.ylabel("Gold Price (USD)")
plt.xticks(rotation=45)  # 使日期标签倾斜，方便阅读
plt.show()

# 9. 预测未来黄金价格和美元指数
# 让用户输入未来的日期（例如：2023-12-01）
print("请输入未来日期（格式：YYYY-MM-DD）：")
future_date_input = input()  # 获取用户输入的日期

# 将用户输入的字符串转换为datetime类型
future_date = datetime.strptime(future_date_input, "%Y-%m-%d")

# 根据输入日期计算特征：day_of_week 和 month
future_day_of_week = future_date.weekday()  # 获取星期几，0=Monday, 6=Sunday
future_month = future_date.month  # 获取月份

# 假设未来日期的DXY_Lag1和DXY_Lag7与历史数据相同，或者用户提供未来美元指数的输入
# 这里假设我们给定一个未来的美元指数值，可以从外部输入获取，或根据历史趋势预测。
# 例如，用户可以输入未来美元指数的预测值
print("请输入未来日期的预测美元指数（例如：103.5）：")
future_dxy = float(input())  # 用户输入未来美元指数

# 创建包含用户输入的特征（DXY_Lag1和DXY_Lag7都使用相同的未来美元指数值）
future_features = pd.DataFrame([[future_dxy, future_dxy, future_day_of_week, future_month]], 
                               columns=["DXY_Lag1", "DXY_Lag7", "day_of_week", "month"])

# 使用神经网络预测黄金价格
mlp_future_pred = mlp.predict(future_features)  # 神经网络预测黄金价格

# 使用随机森林预测黄金价格
rf_future_pred = rf.predict(mlp_future_pred.reshape(-1, 1))  # 随机森林预测黄金价格

# 输出预测结果
print(f"根据输入的未来日期 {future_date_input} 和美元指数 {future_dxy}，预测的黄金价格为：{rf_future_pred[0]} USD")
