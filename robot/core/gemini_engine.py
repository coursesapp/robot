import asyncio
import logging
import base64
import numpy as np
import cv2
from google import genai
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger("GeminiLiveEngine")

class GeminiLiveEngine:
    """
    Handles the Multimodal Live API stream for Gemini 2.0 Flash.
    Connects via WebSockets to send audio/vision and receive audio/text responses.
    """
    def __init__(self, api_key: str, model_id: str = "gemini-2.0-flash", voice_name: str = "Aoede"):
        self.api_key = api_key
        self.model_id = model_id
        self.voice_name = voice_name
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        self.session = None
        self.running = False
        self._on_audio_cb = None
        self._on_text_cb = None
        
    async def connect(self, system_instruction: str):
        """Establishes the live connection."""
        config = {
            "model": self.model_id,
            "system_instruction": system_instruction,
            "generation_config": {
                "speech_config": {
                    "voice_config": {"prebuilt_voice_config": {"voice_name": self.voice_name}}
                }
            }
        }
        
        try:
            logger.info(f"Connecting to Gemini Live API ({self.model_id})...")
            async with self.client.aio.live.connect(model=self.model_id, config=config) as session:
                self.session = session
                self.running = True
                logger.info("Gemini Live Session Started.")
                
                async for message in session:
                    await self._handle_message(message)
        except Exception as e:
            logger.error(f"Gemini Live Connection Error: {e}")
            self.running = False

    async def _handle_message(self, message):
        """Processes incoming messages from Gemini (Audio/Text)."""
        # The SDK response object structure for Multimodal Live:
        if hasattr(message, 'audio') and message.audio:
            if self._on_audio_cb:
                self._on_audio_cb(message.audio)
        
        if hasattr(message, 'text') and message.text:
            if self._on_text_cb:
                self._on_text_cb(message.text)
        
        # Check for server content responses
        if hasattr(message, 'server_content') and message.server_content:
            model_turn = message.server_content.model_turn
            if model_turn:
                for part in model_turn.parts:
                    if hasattr(part, 'text') and part.text:
                        if self._on_text_cb: self._on_text_cb(part.text)
                    if hasattr(part, 'inline_data') and part.inline_data:
                        if self._on_audio_cb: self._on_audio_cb(part.inline_data.data)

    async def send_audio(self, audio_bytes: bytes):
        """Streams audio data to Gemini."""
        if self.session and self.running:
            await self.session.send(input={"mime_type": "audio/pcm;rate=16000", "data": base64.b64encode(audio_bytes).decode('utf-8')}, end_of_turn=False)

    async def send_text(self, text: str):
        """Sends text context or grounding info."""
        if self.session and self.running:
            await self.session.send(input=text, end_of_turn=False)

    async def send_frame(self, frame: np.ndarray):
        """Streams a vision frame to Gemini."""
        if self.session and self.running:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_frame = base64.b64encode(buffer).decode('utf-8')
            # The exact send format for vision in Live API needs verification
            # Usually it's via Media message
            await self.session.send(input={"mime_type": "image/jpeg", "data": b64_frame}, end_of_turn=False)

    def set_callbacks(self, on_audio: Callable, on_text: Callable):
        self._on_audio_cb = on_audio
        self._on_text_cb = on_text

    def stop(self):
        self.running = False
        self.session = None
