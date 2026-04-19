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

    # ---------------- VISION ----------------
    async def add_vision(self, description: str):
        async with self._lock:
            self.vision_events.append({
                "text": description,
                "time": datetime.utcnow()
            })

    async def get_latest_vision(self):
        async with self._lock:
            return self.vision_events[-1] if self.vision_events else None

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
        async with self._lock:
            return {
                # Returns a list of the last 3 elements (or fewer if list is short)
                "latest_vision": self.vision_events[-1:], 
                # "latest_audio": self.audio_events[-2:],
                "state": self.current_state
            }

    async def get_audio_context(self):
        async with self._lock:
            return {
                # Returns a list of the last 3 elements (or fewer if list is short)
                "latest_audio": self.audio_events[-1:],
            }