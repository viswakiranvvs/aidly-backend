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
                                Every detail you see visually has to be covered. 
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
                            """        

    def create_sample(self, query,images):

        context = asyncio.run(self.contextManag.get_context_summary())
        audioQues = asyncio.run(self.contextManag.get_audio_context())

        content = [
            {
                "type": "text",
                "text": f"{self.system_message}. Analyze the change and give detail in brief, Answer any user queries if they are not answered previously: {audioQues}, take help of the context and the steps if necessary: {context}"
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
    
    def _predict_sync(self, images):
        

        sample = self.create_sample("", images)

        output = self.generate_text_from_sample(sample)

        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                text = parsed.get("detail", output)
                state = {}
                state["further_steps"]=parsed.get("further_steps", output)
                state["additional_context"]=parsed.get("additional_context", output)
                asyncio.run(self.contextManag.add_vision(text))
                asyncio.run(self.contextManag.update_state("state",state))
            return output
        except Exception as e:
            print("Error in vision model: ",e)

    async def predict(self,images):
        return await asyncio.to_thread(self._predict_sync, images)
        # query=""
        # context = await self.contextManag.get_context_summary()
        # sample = self.create_sample(query,images,context)
        # output = self.generate_text_from_sample(sample,device=self.device)
        

    
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
        