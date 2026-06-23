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
import json
import re
import asyncio

class CallAndRespond:
    def __init__(self, qwenModel):
        self.qwenTextModel = qwenModel

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
    
    async def process_and_send_response(self, websocket, context, text,audioQuesCtx):
        try:
            response_generator = self.qwenTextModel.predict_stream(text, context,audioQuesCtx)
            sentence_buffer = ""
            word_threshold = 40 # ⚡ Send to TTS after ~7 words even if no punctuation

            for chunk in response_generator:
                sentence_buffer += chunk
                
                # Logic A: Split on sentence boundaries
                # Logic B: Split if the buffer is getting too long (to reduce latency)
                words = sentence_buffer.split()
                
                should_split = re.search(r'(?<!\d)[.!?](?!\d)|\n', sentence_buffer)
                # should_split = re.search(r'[.!?\n]', sentence_buffer) or len(words) >= word_threshold
                
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
                        parts = re.split(r'((?<!\d)[.!?](?!\d)|\n)', sentence_buffer)
                        # parts = re.split(r'([.!?\n])', sentence_buffer)
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
