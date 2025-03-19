import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(current_dir), 'Models')

# 加载模型和scaler
model = load_model(os.path.join(models_dir, 'gru_dxy_model.h5'))
scaler_dxy = pickle.load(open(os.path.join(models_dir, 'scaler_dxy.pkl'), 'rb'))
scaler_gold = pickle.load(open(os.path.join(models_dir, 'scaler_gold.pkl'), 'rb'))

async def predict_future_price(future_date_str):
    try:
        # 转换日期
        future_date = datetime.strptime(future_date_str, "%Y-%m-%d")
        
        # 计算时间特征
        future_day_of_week = future_date.weekday()
        future_month = future_date.month
        
        # 创建特征集
        future_features = pd.DataFrame([[0, 0, 0, 0, 0, 0, future_day_of_week, future_month]], 
                                     columns=["DXY_Lag1", "DXY_Lag7", "DXY_Lag30", "DXY_Lag60", 
                                            "DXY_MA30", "DXY_MA60", "day_of_week", "month"])
        
        # 预测美元指数
        future_features_gru = np.array(future_features).reshape((future_features.shape[0], 1, future_features.shape[1]))
        future_dxy_gru = model.predict(future_features_gru)
        future_dxy = scaler_dxy.inverse_transform(future_dxy_gru)
        
        # 预测黄金价格
        future_features_8 = future_features[["DXY_Lag1", "DXY_Lag7", "DXY_Lag30", "DXY_Lag60", 
                                           "DXY_MA30", "DXY_MA60", "day_of_week", "month"]]
        
        # 使用随机森林和MLP模型预测
        rf_model = pickle.load(open('Models/rf_gold_model.pkl', 'rb'))
        mlp_model = pickle.load(open('Models/mlp_gold_model.pkl', 'rb'))
        
        rf_gold_pred = rf_model.predict(future_features_8.values)
        mlp_gold_pred = mlp_model.predict(future_features_8.values)
        
        # 组合预测结果
        gold_pred_scaled = (rf_gold_pred + mlp_gold_pred) / 2
        gold_pred = scaler_gold.inverse_transform(gold_pred_scaled.reshape(-1, 1))
        
        return {
            "status": "success",
            "dxy_prediction": float(future_dxy[0][0]),
            "gold_prediction": float(gold_pred[0][0]),
            "date": future_date_str
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

async def handle_websocket(websocket, path):
    try:
        async for message in websocket:
            data = json.loads(message)
            
            if data["type"] == "predict":
                result = await predict_future_price(data["date"])
                await websocket.send(json.dumps(result))
            else:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Invalid request type"
                }))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    server = await websockets.serve(handle_websocket, "localhost", 8765)
    print("Server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main()) 