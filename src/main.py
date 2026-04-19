from fastapi import FastAPI, WebSocket
import asyncio
import base64
import json
from helper.message_helper import MessageHelper
from helper.tts_helper import TTSHelper

app = FastAPI()
clients = set()
messageHelper = MessageHelper()
ttsHelper = TTSHelper()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            asyncio.create_task(messageHelper.handle_message(message, websocket))

    except Exception as e:
        print("Client disconnected:", e)
        clients.remove(websocket)