import asyncio
import websockets
import json
from datetime import datetime, timedelta

async def connect_to_server():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        print("已连接到服务器")
        print("请输入要预测的日期 (格式: YYYY-MM-DD)，或输入 'q' 退出：")
        
        while True:
            user_input = input()
            
            if user_input.lower() == 'q':
                break
                
            try:
                # 验证日期格式
                datetime.strptime(user_input, "%Y-%m-%d")
                
                # 发送预测请求
                request = {
                    "type": "predict",
                    "date": user_input
                }
                
                await websocket.send(json.dumps(request))
                response = await websocket.recv()
                result = json.loads(response)
                
                if result["status"] == "success":
                    print(f"\n预测结果：")
                    print(f"日期: {result['date']}")
                    print(f"美元指数预测值: {result['dxy_prediction']:.2f}")
                    print(f"黄金价格预测值: ${result['gold_prediction']:.2f}\n")
                else:
                    print(f"错误: {result['message']}\n")
                    
            except ValueError:
                print("日期格式错误，请使用 YYYY-MM-DD 格式\n")
            except Exception as e:
                print(f"发生错误: {str(e)}\n")

if __name__ == "__main__":
    asyncio.run(connect_to_server()) 