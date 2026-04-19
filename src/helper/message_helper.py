import numpy as np
from PIL import Image
import io
import base64
from faster_whisper import WhisperModel
from datetime import datetime
import os
import asyncio
from models.new_reasoning_model import QwenReasonHelperText
from helper.tts_helper import TTSHelper
from helper.process_frame import FrameDiffProcessor
import json
from helper.context_helper import ContextManager
import re
import asyncio

class MessageHelper:
    def __init__(self):
        self.stt_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        self.image_dir = "frames"
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        self.qwenTextModel = QwenReasonHelperText()
        self.contextManag = ContextManager()
        self.frame_diff = FrameDiffProcessor(max_frames=4)

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
                # await self.send_audio(websocket,"response.mp3")
                asyncio.create_task(
                    self.process_and_send_response(websocket,context,text,audioQues)
                )
            # websocket.send()
        except Exception as e:
            print("Audio error:", e)

    async def process_frame(self,frame_data, websocket): 
        try:
            print("Received frame")
            self.frame_diff.add_frame(frame_data)

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
                audioQues = await self.contextManag.get_audio_context()
                await self.contextManag.add_audio(text)
                context = await self.contextManag.get_context_summary()
                print(context)
                # response = self.qwenTextModel.predict(context,text)
                # await TTSHelper.text_to_mp3(response,"response.mp3")
                # await self.send_audio(websocket,"response.mp3")
                asyncio.create_task(
                    self.process_and_send_response(websocket,context,text,audioQues)
                )
            # websocket.send()
        except Exception as e:
            print("Audio error:", e)
    
    async def send_audio(self, websocket, file_path):
        try:
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            await websocket.send_text(json.dumps({
                "type": "audio",
                "data": audio_base64
            }))
            print("Response audio sent")
            os.remove(file_path)
        except Exception as e:
            print("Audio send Error: ",e)
    
    async def process_and_send_response(self, websocket, context, text,audioQues):
        try:
            response_generator = self.qwenTextModel.predict_stream(text, context,audioQues)
            sentence_buffer = ""
            word_threshold = 30 # ⚡ Send to TTS after ~7 words even if no punctuation

            for chunk in response_generator:
                sentence_buffer += chunk
                
                # Logic A: Split on sentence boundaries
                # Logic B: Split if the buffer is getting too long (to reduce latency)
                words = sentence_buffer.split()
                
                should_split = re.search(r'[.!?\n]', sentence_buffer) or len(words) >= word_threshold
                
                if should_split:
                    # If we split by length, we find the last space to avoid cutting words
                    if not re.search(r'[.!?\n]', sentence_buffer):
                        # Split at the last space
                        split_idx = sentence_buffer.rfind(" ")
                        if split_idx == -1: continue # Wait for a space
                        current_payload = sentence_buffer[:split_idx].strip()
                        sentence_buffer = sentence_buffer[split_idx:].strip()
                    else:
                        # Standard sentence split
                        parts = re.split(r'([.!?\n])', sentence_buffer)
                        if len(parts) >= 2:
                            current_payload = (parts.pop(0) + parts.pop(0)).strip()
                            sentence_buffer = "".join(parts)
                        else:
                            continue

                    # Process the payload
                    if current_payload and any(c.isalnum() for c in current_payload):
                        try:
                            safe_hash = abs(hash(current_payload))
                            audio_path = f"temp_{safe_hash}.mp3"
                            
                            print(f"TTS Chunk: {current_payload}")
                            await TTSHelper.text_to_mp3(current_payload, audio_path)
                            await self.send_audio(websocket, audio_path)
                            
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                        except Exception as e:
                            print(f"TTS Error: {e}")

            # Handle any remaining text in the buffer after the generator ends
            if sentence_buffer.strip():
                audio_path = "temp_final.mp3"
                print(sentence_buffer)
                await TTSHelper.text_to_mp3(sentence_buffer.strip(), audio_path)
                await self.send_audio(websocket, audio_path)

        except Exception as e:
            print("Audio error:", e)
