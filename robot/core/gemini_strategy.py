import logging
import threading
import time
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np
from core.interaction import InteractionStrategy
from core.gemini_engine import GeminiLiveEngine

logger = logging.getLogger("GeminiLiveStrategy")

class GeminiLiveStrategy(InteractionStrategy):
    """
    Implements the real-time multimodal streaming interaction logic.
    User Speech -> Live Stream -> Real-time Response
    Vision Frame -> Live Stream -> Visual Grounding
    """
    def __init__(self, agent_loop):
        self.agent = agent_loop
        self.config = agent_loop.config
        
        # Engine Initialization
        g_cfg = self.config.get('interaction', {}).get('gemini', {})
        self.engine = GeminiLiveEngine(
            api_key=g_cfg.get('api_key', ''),
            model_id=g_cfg.get('model', 'gemini-2.0-flash'),
            voice_name=g_cfg.get('voice_name', 'Aoede')
        )
        
        # Async Bridging
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        
        self.look_mode = False
        self.last_frame_sent = 0
        self.frame_interval = 1.0 / g_cfg.get('look_mode_fps', 2)

    def _run_async_loop(self):
        asyncio.set_event_loop(self.loop)
        
        # Build System Instruction
        # We reuse the existing prompt engine's logic for consistency
        sys_inst = self.agent.prompt_engine.system_prompt
        
        # Set Callbacks
        self.engine.set_callbacks(
            on_audio=self._handle_gemini_audio,
            on_text=self._handle_gemini_text
        )
        
        self.loop.run_until_complete(self.engine.connect(sys_inst))

    def on_speech_start(self):
        """Zero-latency Local Mute."""
        logger.info("GeminiLive: Local Mute Triggered (User Speech Start).")
        if self.agent.tts:
            self.agent.tts.interrupt()
        
        # We also send an 'interrupt' signal to the Gemini Engine if supported
        if self.engine and self.engine.running:
            # asyncio.run_coroutine_threadsafe(self.engine.send_interrupt(), self.loop)
            pass

    def _handle_gemini_audio(self, audio_data):
        # Gemini returns audio bytes (usually PCM or similar depending on config)
        # We need to play this via TTS engine or a direct player
        # For now, we'll try to pass it to the agent's speaker
        logger.debug("GeminiLive: Received audio response.")
        # self.agent.tts.play_stream(audio_data)
        pass

    def _handle_gemini_text(self, text):
        logger.info(f"GeminiLive Text: {text}")
        self.agent.context_history.append(f"agent: {text}")

    def on_speech(self, text: str, audio_np: np.ndarray, context_metadata: Dict[str, Any]):
        # In Live mode, audio is often streamed continuously, but if we get a chunk:
        logger.info(f"GeminiLive: Handling speech/event - {text}")
        
        is_system = context_metadata.get('is_system_event', False)
        if is_system and self.engine and self.engine.running:
            asyncio.run_coroutine_threadsafe(self.engine.send_text(text), self.loop)
        
        final_pid = context_metadata.get('final_pid', 'unknown')
        if final_pid != "unknown":
            # Pass grounding info to the stream if it's a normal speech event
            if not is_system:
                grounding_msg = f"[System: The person currently speaking is {final_pid}. Please address them accordingly.]"
                asyncio.run_coroutine_threadsafe(self.engine.send_text(grounding_msg), self.loop)
        
        self.agent.context_history.append(f"user: {text}" if not is_system else f"system: {text}")

    def on_vision(self, frame: np.ndarray, metadata: Dict[str, Any]):
        now = time.time()
        if now - self.last_frame_sent >= self.frame_interval:
            self.last_frame_sent = now
            
            # Grounding with Emotions
            emotions = metadata.get('emotions', [])
            if emotions:
                # We can optionally modify the frame or send metadata
                pass

            asyncio.run_coroutine_threadsafe(self.engine.send_frame(frame), self.loop)

    def stop(self):
        if self.engine:
            self.engine.stop()
        logger.info("GeminiLiveStrategy stopped.")
