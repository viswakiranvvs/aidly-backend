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
from threading import Thread
from transformers import TextIteratorStreamer
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper.context_helper import ContextManager
import re
import asyncio
from dotenv import load_dotenv
import os
from models.crust_web_api import CrustPDF

load_dotenv()
login(os.getenv("HF_TOKEN"))
class QwenReasonHelperText:
    def __init__(self):
        model_id = "Qwen/Qwen2.5-7B-Instruct"

        self.device="cuda"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.contextManag = ContextManager()
        self.last_json_context = None
        self.crawler = CrustPDF()
        self.previousWebQueries = []
        self.webDataAvailable = False
        self.textFromWeb=""
        """
        here is an example

                                current_response: "Sure will help you in Litmus paper test, First take the Litmus paper./aaawd"
                                json_response: {
                                    "further_steps":{
                                        "step_1":{
                                            "content":"Take Litmus paper",
                                            "status":"Currently scheduled"
                                        },
                                        "step_2":{
                                            "content":"Dip in Liquid",
                                            "status":"Not Completed"
                                        },
                                    },
                                    "additional_context":"Can see test tubes and liquid"
                                }
        """
        

    def create_sample(self, query,context,audioQues):
        
        content = (
            f"Current User query/response: {query}; "
            f"Current observed environment: {context}; "
            f"previous user queries/responses: {audioQues};"
        )

        if self.webDataAvailable:
            print("Data Available from web for query: "+query)
            content += f" Web data available: {self.textFromWeb}"
            print(f"Text length {len(self.textFromWeb)}")
        return {
            "messages": [
                {
                    "role": "system",
                    "content":  """
                                    You are helpful assistant, you have visuals of what the user is doing, 
                                    You are observing the scene directly through vision in real-time.
                                    You are not given a description — you are seeing the environment yourself.
                                    Do NOT say "image", "description", or "context".
                                    Speak as if you directly see what is happening.
                                    respond to the user questions in crisp and short manner. 
                                    You have to guide the user step by step in the tasks ask for. Once the steps are completed, remove them from state.
                                    Just leave a short message for simple general questions.
                                    Do not assume unseen details.

                                    give response as
                                    current_response: "plain text ending with /aaawd"

                                    followed by
                                    json_response:
                                    {
                                        "further_steps":[
                                            ## Track all the steps of the process user asked for, what he has done and what he should be doing
                                            "step_1":{
                                                "content":"",
                                                "status":"Completed/Not Completed/Currently scheduled"
                                            }
                                        ],
                                        "additional_context":"",
                                        "needExtraInfoFromWeb" : { //If you dont know much about the product and you want to make a web call about guide or user manual example: HP Printer Laser MFP 323sdnw
                                            "needed": True/False (boolean),
                                            "query": "HP Printer Laser MFP 323sdnw User guide or manual"
                                        }
                                    }
                                """
                },
                {
                    "role": "user",
                    "content": content,
                }
            ]
        }
    
    def predict_stream(self, query, context,audioQues):
        sample = self.create_sample(query, context,audioQues)
        
        # This returns a generator
        try:
            for new_text in self.generate_stream(sample):
                yield new_text
        except Exception as e:
            print("Reason error: ",e)

    def generate_stream(self, sample, max_new_tokens=256, device="cuda"):
        # 1. Prepare Inputs
        inputs = self.tokenizer.apply_chat_template(
            sample['messages'],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 🔥 FIX: Use keyword arguments for device and dtype
        inputs = inputs.to(device=device)

        # 2. Initialize the Streamer
        # skip_prompt=True ensures you only get the NEW generated text
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 3. Define generation kwargs
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        # 4. Start generation in a separate thread
        # We use a thread because .generate() is blocking
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 5. Yield tokens as they become available
        # for new_text in streamer:
        #     print(new_text)
        #     yield new_text

        full_stream_buffer = ""
        json_started = False
        delimiter = "json_response:"
        end_delimeter = "/aaawd"

        for new_text in streamer:
            print(new_text)
            full_stream_buffer += new_text
            
            if not json_started:
                # Check if we hit the delimiter
                if delimiter in full_stream_buffer:
                    json_started = True
                    # Split: Send the last bit of text before the delimiter
                    parts = full_stream_buffer.split(delimiter)
                    text_to_send = parts[0].replace("current_response:", "").strip()
                    # yield text_to_send
                else:
                    # Still in the text phase, yield the raw token if it's not the label
                    if "current_response:" not in new_text:
                        yield new_text.replace(end_delimeter," ")
            else:
                # We are in the JSON phase. 
                # Don't yield to the user/TTS, just keep buffering or handle internally.
                pass

        # After the loop, you have the full JSON in the buffer
        if json_started:
            try:
                raw_json = full_stream_buffer.split(delimiter)[1].strip()
                # Clean up potential markdown code blocks if Qwen adds them
                raw_json = raw_json.replace("```json", "").replace("```", "").strip()
                # final_json = json.loads(raw_json)
                
                self.last_json_context = self.repair_json(raw_json)
                needed, query = self.extract_web_query()
                if (needed==True or needed=="True") and query not in self.previousWebQueries:
                    print("Doing ")
                    self.previousWebQueries.append(query)
                    self.textFromWeb = self.crawler.run(query)
                    self.webDataAvailable = True
                    print
                    # self.crawler.run(query)
                    # thread = Thread(target=self.crawler.run, kwargs=query)
                    # thread.start()
                    
                asyncio.create_task(
                    self.contextManag.update_state("state",self.last_json_context)
                )
                
                print("Context Updated with JSON:", self.last_json_context)
            except Exception as e:
                print("Failed to parse JSON context:", e)
    
    def repair_json(self, json_str):
        if not json_str:
            return None
        
        # 1. Basic Cleaning
        json_str = json_str.strip()
        
        # 2. Fix the "Key:Value" syntax inside arrays (Specific to your schema)
        # Your example shows "step_1": { ... } inside an array [ ]. 
        # Valid JSON needs these to be objects { "step_1": { ... } }
        # This regex wraps them if they aren't wrapped.
        json_str = re.sub(r'("step_\d+":\s*\{)', r'{\1', json_str)
        
        # 3. Handle unclosed quotes first
        # If the number of double quotes is odd, the string is likely cut off mid-word
        if json_str.count('"') % 2 != 0:
            json_str += '"'

        # 4. Remove dangling commas at the end of objects/arrays
        # e.g., {"a": 1, } -> {"a": 1}
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

        # 5. Fix the trailing period in your example (]. vs ])
        json_str = json_str.replace('].', ']')

        # 6. Balance Braces and Brackets
        opened_braces = json_str.count('{') - json_str.count('}')
        opened_brackets = json_str.count('[') - json_str.count(']')
        
        # Close them in the correct LIFO order
        # Note: This is a simplified heuristic. 
        # For a production drone, we close brackets then braces.
        json_str += '}' * max(0, opened_braces)
        json_str += ']' * max(0, opened_brackets)
        
        # Final check: sometimes the repair creates a "}}" if we were mid-object
        # We try to load; if it fails, we try one more brace
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                return json.loads(json_str + "}")
            except:
                return json_str
    
    def extract_web_query(self):
        data = self.last_json_context
        if not isinstance(data, dict):
            print("context not dict")
            query_match = re.search(r'"query":\s*"([^"]+)"', data)
            needed_match = re.search(r'"needed":\s*(True|False|true|false)', data)
            extracted_query=None
            val=None
            if query_match:
                extracted_query = query_match.group(1)

            if needed_match:
                # Handle both Python (True) and JSON (true) formats
                val = needed_match.group(1).lower() == 'true'
            print(val,extracted_query)
            return val, extracted_query
        

        web_info = data.get("needExtraInfoFromWeb", {})

        needed = str(web_info.get("needed", "false")).lower() == "true"
        query = web_info.get("query")
        print(needed,query)
        return needed, query
