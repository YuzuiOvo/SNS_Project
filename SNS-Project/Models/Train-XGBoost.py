import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb

# 读取数据
df = pd.read_csv("D:/cursor/Project/SNS-Project/Data/merged_data.csv")

# 转换日期为数值（比如将日期转换为天数，方便机器学习模型使用）
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(lambda x: x.toordinal())  # 将日期转换为整数

# 选择特征（DXY指数和日期）
X = df[['Date', 'Close_y']]  # 使用Date和DXY指数作为特征
y = df['Close_x']  # 黄金价格是我们要预测的目标变量

# 将数据分为训练集和测试集，80%的数据用于训练，20%的数据用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建 XGBoost 回归模型
xgb_model = xgb.XGBRegressor(n_estimators=30, random_state=20)

# 训练模型
xgb_model.fit(X_train, y_train)

# 预测
xgb_y_pred = xgb_model.predict(X_test)

# 评估模型性能
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)

print(f"XGBoost Mean Squared Error: {xgb_mse}")
print(f"XGBoost R² Score: {xgb_r2}")

# 可视化预测结果
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, color='blue', label='Actual Gold Price')
plt.plot(y_test.index, xgb_y_pred, color='red', linestyle='dashed', label='Predicted Gold Price')
plt.title('Actual vs Predicted Gold Price')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.legend()
plt.show()
