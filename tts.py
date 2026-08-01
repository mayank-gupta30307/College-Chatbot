import asyncio
import edge_tts
import pygame
import os


async def _speak(text: str):
    communicate = edge_tts.Communicate(
        text=text,
        voice="en-US-GuyNeural"      # Male
        # voice="en-US-JennyNeural"  # Female
    )
    await communicate.save("response.mp3")


def speak(text: str):
    asyncio.run(_speak(text))
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.quit()
    os.remove("response.mp3")