import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# 读取数据
data = pd.read_csv('D:/cursor/Project/SNS_Project/SNS_Chatbot/Data/dxy_index.csv')
dxy_values = data['Close'].values.reshape(-1, 1)  # 变成列向量

# 归一化数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dxy_values)

# 修正 `create_dataset()`
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])  # 30 天历史数据
        y.append(data[i + time_step, 0])  # 预测值
    return np.array(X), np.array(y).reshape(-1, 1)  # 修正 y 形状

# 设置时间步长
time_step = 30
X, y = create_dataset(scaled_data, time_step)

# 确保 `X` 形状为 `(samples, time_steps, features)`
X = X.reshape(X.shape[0], X.shape[1], 1)  # 修正 reshape

# 🔹 打印形状检查
print("X shape:", X.shape)  # 期望: (samples, 30, 1)
print("y shape:", y.shape)  # 期望: (samples, 1)

# 构建模型
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(time_step, 1)))  # 修正 input_shape
model.add(Dropout(0.2))
model.add(GRU(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# 编译和训练模型
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=32)

# 保存模型
model.save('Models/gru_dxy_model.h5')
