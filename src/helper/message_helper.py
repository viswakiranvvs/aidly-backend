import numpy as np
from PIL import Image
import io
import base64
from faster_whisper import WhisperModel
from datetime import datetime
import os
import asyncio
from helper.tts_helper import TTSHelper
from helper.process_frame import FrameDiffProcessor
import json
from helper.context_helper import ContextManager
import re
import asyncio
from helper.call_respond_helper import CallAndRespond
from models.new_reasoning_model import QwenReasonHelperText

class MessageHelper:
    def __init__(self):
        self.stt_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        self.image_dir = "frames"
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        self.contextManag = ContextManager()
        self.qwenTextModel = QwenReasonHelperText()
        self.call_respond = CallAndRespond(self.qwenTextModel)
        self.frame_diff = FrameDiffProcessor(self.qwenTextModel,max_frames=4)

    async def handle_message(self, message, websocket):
        msg_type = message.get("type")

        if msg_type == "frame":
            #  await asyncio.to_thread(self.process_frame, message["data"], websocket)
            asyncio.create_task(
                self.process_frame(message["data"], websocket)
            )

        elif msg_type == "audio":
            asyncio.create_task(
                self.process_audio(message["data"], websocket)
            )
        
        elif msg_type == "text":
            asyncio.create_task(
                self.process_text(message["data"], websocket)
            )
    
    async def process_text(self,text,websocket):
        try:
            print(text)
            hallucinations = ["[Music]", "Thank you.", "Subtitle by", "Thanks for watching", "Thank you for watching!", "Thank you for watching"]
            if text and not any(h in text for h in hallucinations):
                audioQues = await self.contextManag.get_audio_context()
                await self.contextManag.add_audio(text)
                context = await self.contextManag.get_context_summary()
                print(context)
                # response = self.qwenTextModel.predict(context,text)
                # await TTSHelper.text_to_mp3(response,"response.mp3")
                asyncio.create_task(
                    self.process_and_send_response(websocket,context,text,audioQues)
                )
            # websocket.send()
        except Exception as e:
            print("Audio error:", e)

    async def process_frame(self,frame_data, websocket): 
        try:
            print("Received frame")
            self.frame_diff.add_frame(frame_data,websocket)

            # image_bytes = base64.b64decode(frame_data)
            # image = Image.open(io.BytesIO(image_bytes))
            # image = image.resize((224, 224))
            # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # filename = f"f_{timestamp}.png"
            # image.save(os.path.join(self.image_dir,filename))
        except Exception as e:
            print("Frame error:", e)
    
    async def process_audio(self,audio_chunks, websocket):
        try:
            print("Received Audio")
            audio_bytes = b''.join([base64.b64decode(c) for c in audio_chunks])
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0
            segments, info = self.stt_model.transcribe(audio_np, beam_size=1)
            full_text = ""
            for segment in segments:
                full_text += segment.text + " "     
            text = full_text.strip()
            print(text)
            hallucinations = ["[Music]", "Thank you.", "Subtitle by", "Thanks for watching", "Thank you for watching!", "Thank you for watching"]
            if text and not any(h in text for h in hallucinations):
                audioQuesCtx = await self.contextManag.get_audio_context()
                await self.contextManag.add_audio(text)
                context = await self.contextManag.get_context_summary()
                print(context)
                # response = self.qwenTextModel.predict(context,text)
                # await TTSHelper.text_to_mp3(response,"response.mp3")
                # await self.send_audio(websocket,"response.mp3")
                asyncio.create_task(
                    self.call_respond.process_and_send_response(websocket,context,text,audioQuesCtx)
                )
            # websocket.send()
        except Exception as e:
            print("Audio error:", e)
    
    