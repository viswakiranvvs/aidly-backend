import edge_tts
import asyncio

class TTSHelper:
    async def text_to_mp3(text, output_file):
        communicate = edge_tts.Communicate(text, "en-IN-PrabhatNeural")
        await communicate.save(output_file)

# asyncio.run(text_to_mp3("Target identified.", "response.mp3"))