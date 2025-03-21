from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# 加载已训练的模型和Scaler
gru_dxy_model = load_model('gru_dxy_model.h5')

# 使用你训练时保存的MinMaxScaler来避免未拟合的问题
with open('scaler_dxy.pkl', 'rb') as f:
    scaler_dxy = pickle.load(f)

with open('scaler_gold.pkl', 'rb') as f:
    scaler_gold = pickle.load(f)

# 加载RandomForest和MLP模型
with open('rf_gold_model.pkl', 'rb') as f:
    rf_gold_model = pickle.load(f)

with open('mlp_gold_model.pkl', 'rb') as f:
    mlp_gold_model = pickle.load(f)

# 预测美元指数的函数
def predict_dxy_gru(model, future_features):
    dxy_pred_scaled = model.predict(future_features)
    return scaler_dxy.inverse_transform(dxy_pred_scaled)  # Inverse transform to get the original scale

# 预测黄金价格的函数
def predict_gold(future_features, predicted_dxy):
    rf_gold_pred = rf_gold_model.predict(future_features)
    mlp_gold_pred = mlp_gold_model.predict(future_features)
    gold_pred_scaled = (rf_gold_pred + mlp_gold_pred) / 2
    
    # 确保对预测结果进行逆缩放
    gold_pred = scaler_gold.inverse_transform(gold_pred_scaled.reshape(-1, 1))
    
    return gold_pred[0][0]  # 返回逆缩放后的黄金预测价格

# 处理用户请求的预测逻辑
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 从前端获取用户输入的日期
        date_input = request.form['date_input']
        
        # 将日期转为 datetime 格式
        future_date = datetime.strptime(date_input, "%Y-%m-%d")
        
        # 计算日期相关特征
        future_day_of_week = future_date.weekday()
        future_month = future_date.month
        
        # 创建未来的特征（假设滞后特征为 0）
        future_features = pd.DataFrame([[0, 0, 0, 0, 0, 0, future_day_of_week, future_month]],
                                       columns=["DXY_Lag1", "DXY_Lag7", "DXY_Lag30", "DXY_Lag60", "DXY_MA30", "DXY_MA60", "day_of_week", "month"])
        
        future_features_gru = np.array(future_features).reshape((future_features.shape[0], 1, future_features.shape[1]))
        
        # 预测美元指数
        future_dxy = predict_dxy_gru(gru_dxy_model, future_features_gru)
        
        # 预测黄金价格
        predicted_gold = predict_gold(future_features, future_dxy[0][0])
        
        # 将预测结果转换为标准 Python float 类型，避免 JSON 序列化错误
        result = {
            "dxy_predicted": float(future_dxy[0][0]),  # 转换为 float 类型
            "gold_predicted": float(predicted_gold)    # 转换为 float 类型
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# 页面路由，渲染 HTML 页面
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
