import asyncio
from datetime import datetime


class ContextManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ContextManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # prevent re-init
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._lock = asyncio.Lock()

        # 🧠 Core state
        self.vision_events = []   # what was seen
        self.audio_events = []    # what was heard
        self.current_state = {}   # step / status
        self.llm_processed_timestamp = None
        self.vlm_processed_timestamp = None
        self.current_frame_id = None
        self.partial_vision = {}
        # self.current_vision_partial = {}

    # ---------------- VISION ----------------
    async def add_vision(self, description: str):
        async with self._lock:
            self.vision_events.append({
                "text": description,
                "time": datetime.utcnow()
            })


    async def start_new_frame(self):
        async with self._lock:
            frame_id = datetime.utcnow().isoformat()
            self.current_frame_id = frame_id

            self.partial_vision[frame_id] = {
                "buffer": "",
                "last_update": datetime.utcnow()
            }

            return frame_id

    async def add_vision_partial(self, text: str, frame_id: str):
        async with self._lock:
            if frame_id not in self.partial_vision:
                return
            self.partial_vision[frame_id]["buffer"] = text
            self.partial_vision[frame_id]["last_update"] = datetime.utcnow()
            print(text)


    async def finalize_vision(self, frame_id: str):
        async with self._lock:
            data = self.partial_vision.get(frame_id)

            if not data:
                return

            final_text = data["buffer"]

            self.vision_events.append({
                "text": final_text,
                "time": datetime.utcnow()
            })

            # cleanup
            del self.partial_vision[frame_id]

    async def get_latest_vision(self):
        async with self._lock:
            results = {}

            # 🔹 get current partial (latest streaming)
            current_partial = None
            if self.current_frame_id and self.current_frame_id in self.partial_vision:
                current_partial = self.partial_vision[self.current_frame_id]["buffer"]

            # 🔹 latest
            if current_partial:
                results["latest_seen"] = current_partial
            elif self.vision_events:
                results["latest_seen"] = self.vision_events[-1]["text"]

            if not current_partial and len(self.vision_events) >= 1:
                results["previous"] = self.vision_events[-1]["text"]

            if len(self.vision_events) >= 2:
                results["previous_2"] = self.vision_events[-2]["text"]

            return results

    # ---------------- AUDIO ----------------
    async def add_audio(self, text: str):
        async with self._lock:
            self.audio_events.append({
                "text": text,
                "time": datetime.utcnow()
            })

    async def get_latest_audio(self):
        async with self._lock:
            return self.audio_events[-1] if self.audio_events else None

    # ---------------- STATE ----------------
    async def update_state(self, key: str, value):
        async with self._lock:
            self.current_state[key] = value

    async def get_state(self):
        async with self._lock:
            return dict(self.current_state)

    # ---------------- COMBINED ----------------
    async def get_context_summary(self):
        latest_vision = await self.get_latest_vision()
        async with self._lock:
            return {
                # Returns a list of the last 3 elements (or fewer if list is short)
                "latest_vision": latest_vision, 
                # "latest_audio": self.audio_events[-6:],
                "state": self.current_state
            }

    async def get_audio_context(self):
        async with self._lock:
            return {
                # Returns a list of the last 3 elements (or fewer if list is short)
                "latest_audio": self.audio_events[-4:],
            }
    
    async def update_llm_timestamp(self):
        async with self._lock:
            self.llm_processed_timestamp=datetime.now().timestamp()
    
    async def update_vlm_timestamp(self):
        async with self._lock:
            self.vlm_processed_timestamp=datetime.now().timestamp()

    def get_last_llm_timestamp(self):
        return self.llm_processed_timestamp
    
    def get_last_vlm_timestamp(self):
        return self.vlm_processed_timestamp