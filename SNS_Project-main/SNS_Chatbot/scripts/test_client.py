import websockets
import asyncio
import json

async def test_client():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"type": "predict", "date": "2025-04-30"}))
        response = await websocket.recv()
        print("Server response:", response)

asyncio.run(test_client())
