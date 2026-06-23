import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTConfig
from trl import SFTTrainer
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
from helper.context_helper import ContextManager
from datetime import datetime
import asyncio
from huggingface_hub import login
from dotenv import load_dotenv
import os
from transformers import TextIteratorStreamer
from threading import Thread

load_dotenv()
login(os.getenv("HF_TOKEN"))

class VisionHelper:
    def __init__(self):
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        self.device="cuda"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map=self.device,
            torch_dtype=torch.float16,
            # local_files_only=True
        )

        lora_layers = [n for n, _ in self.model.named_modules() if "lora" in n.lower()]
        print("LoRA layers: " + str(len(lora_layers)))
        # .to(device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.contextManag = ContextManager()

        """
                                        observe state context if there are any steps scheduled and help the user.

                                give response as json: 

        {
                                    "detail": "", //Image details
                                    "further_steps":[
                                        ## Get the steps from context and update status
                                        "step_1":{
                                            "content":"",
                                            "status":"Completed/Not Completed/Currently scheduled",
                                            "summary":""
                                        }
                                    ],
                                    "additional_context":"",
                                    "messageToUser":"" // only important message to user -- very very important message only regarding deviation from steps or very brief summary about completed action and next step if any
                                }

        example:
                                {
                                    "detail": "Test tubes and Hydrochloric acid bottle are visible",
                                    "further_steps": [
                                        "step_1":{
                                            "content":"Wear Gloves",
                                            "status":"Completed",
                                            "summary":""
                                        },
                                        "step_2":{
                                            "content":"Pick Hydrochloric acid using pipette",
                                            "status":"Currently scheduled",
                                            "summary":""
                                        }
                                    ]
                                    "additional_context":"Test tube stands are present and user is wearing gloves",
                                    "messageToUser": "Now, please use pipette to transfer Hydrochloric acid to test tube"
                                }
        """
        

        self.system_message=""" You are a Vision model
                                Give description about what you see in the current images. Extract all the text that you see.
                                The context contains latest vision info, which is the same info that you have observed previously.
                                Observe the steps which are essential to complete user ask in the,
                                Every detail you see visually has to be covered in brief in plain text. Do not reason about past frames. Give very precisely and briefly.
                            """        

    def create_sample(self, query,images):

        context = asyncio.run(self.contextManag.get_context_summary())
        audioQues = asyncio.run(self.contextManag.get_audio_context())

        content = [
            {
                "type": "text",
                "text": f"{self.system_message}. Analyze the change and give detail in brief"
            }
        ]

        # add all images
        for img in images:
            content.append({
                "type": "image",
                "image": img
            })

        return {
        "images": images,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
        }
    
    def _predict_sync(self, images, loop):
        sample = self.create_sample("", images)

        # start frame (safe)
        frame_id = asyncio.run(self.contextManag.start_new_frame())

        streamer = self.generate_stream(sample)

        buffer = ""
        last_update_len = 0

        for token in streamer:
            buffer += token

            if len(buffer) - last_update_len > 50:
                last_update_len = len(buffer)

                # 🔥 schedule on MAIN loop
                loop.call_soon_threadsafe(
                    asyncio.create_task,
                    self.contextManag.add_vision_partial(buffer, frame_id)
                )

        # finalize
        loop.call_soon_threadsafe(
            asyncio.create_task,
            self.contextManag.finalize_vision(frame_id)
        )

        return buffer

    async def predict(self, images):
        loop = asyncio.get_running_loop()  # ✅ capture main loop
        return await asyncio.to_thread(self._predict_sync, images, loop)
        # query=""
        # context = await self.contextManag.get_context_summary()
        # sample = self.create_sample(query,images,context)
        # output = self.generate_text_from_sample(sample,device=self.device)

                    # parsed = json.loads(output)
            # if isinstance(parsed, dict):
            #     text = parsed.get("detail", output)
            #     state = {}
            #     state["further_steps"]=parsed.get("further_steps", output)
            #     state["additional_context"]=parsed.get("additional_context", output)
            #     asyncio.run(self.contextManag.add_vision(text))
            #     asyncio.run(self.contextManag.update_state("state",state))
        

    
    def generate_text_from_sample(self,sample, max_new_tokens=512, device="cuda"):
        # image_inputs,_ = process_vision_info(sample)
        inputs = self.processor.apply_chat_template(
            sample['messages'],
            # images=image_inputs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device, torch.float16)

        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,     use_cache=True)

        # 🔹 Get only new tokens
        generated_only = generate_ids[:, inputs["input_ids"].shape[1]:]

        output = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print("Generated output:\n", output)
        return output
    
    def generate_stream(self, sample, device="cuda"):
        inputs = self.processor.apply_chat_template(
            sample['messages'],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device, torch.float16)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            use_cache=True,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer
            