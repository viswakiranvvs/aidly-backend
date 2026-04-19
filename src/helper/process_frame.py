import numpy as np
from PIL import Image
import io
import base64
from models.vision_model import VisionHelper
import asyncio

class FrameDiffProcessor:
    def __init__(self, max_frames=4, threshold=39.0):
        """
        max_frames: number of frames to keep in buffer
        threshold: mean pixel difference threshold (tune this)
        """
        self.max_frames = max_frames
        self.threshold = threshold
        self.frame_buffer = []
        self.vision_model=VisionHelper()
        self.frame_count = 0
        self.warmup_frames = 1  # process first 3 frames always
        self.skip_frames = 0
        self.cooldown_frames = 3

    def _preprocess(self, image_bytes):
        """
        Convert image → grayscale numpy array
        """
        image = Image.open(io.BytesIO(image_bytes))
        print(image.size)
        image = image.resize((400, 400))  # smaller = faster

        # convert to grayscale
        # image = image.convert("L")

        # to numpy
        frame_np = np.array(image, dtype=np.float32)

        return frame_np

    def add_frame(self, frame_base64):
        image_bytes = base64.b64decode(frame_base64)
        current_frame = self._preprocess(image_bytes)

        self.frame_buffer.append(current_frame)
        self.frame_count += 1

        if self.frame_count == self.warmup_frames:
            print(f"[Warmup] Processing frame {self.frame_count}")

            asyncio.create_task(
                self.vision_model.predict(self._prepare_images(self.frame_buffer.copy()))
            )
            return

        # if len(self.frame_buffer) < self.max_frames:
        #     return

        if self.skip_frames > 0:
            self.skip_frames -= 1
            print(f"[Cooldown] Skipping frame, remaining: {self.skip_frames}")
            if len(self.frame_buffer) > self.max_frames:
                self.frame_buffer.pop(0)
            return

        oldest_frame = self.frame_buffer[0]
        diff = self._compute_diff(oldest_frame, current_frame)

        print(f"[FrameDiff] diff={diff:.2f}")

        if diff > self.threshold:
            print("Triggering VLM")

            self.skip_frames = self.cooldown_frames

            asyncio.create_task(
                self.vision_model.predict(self._prepare_images(self.frame_buffer[-2:]))
            )

        if len(self.frame_buffer) > self.max_frames:
            self.frame_buffer.pop(0)

    def _compute_diff(self, frame1, frame2):
        """
        Mean absolute difference
        """
        return np.mean(np.abs(frame1 - frame2))
    
    def _prepare_images(self, frames):
        images = []
        for frame in frames:
            img = Image.fromarray(frame.astype('uint8'))
            images.append(img)
        return images