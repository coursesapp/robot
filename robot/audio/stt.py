import threading
import queue
import time
import logging
import numpy as np
import pyaudio
from typing import Optional, Callable
from faster_whisper import WhisperModel

logger = logging.getLogger("STT")

class STTEngine:
    def __init__(self, model_size: str = "tiny.en", callback: Optional[Callable[[str], None]] = None):
        self.callback = callback
        self.running = False
        self.queue = queue.Queue()
        
        logger.info(f"Loading faster-whisper ({model_size})...")
        try:
            # Run on CPU with INT8 quantization for speed/compat
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("Whisper loaded.")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise e

        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
        
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        logger.info("STT listening...")

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _listen_loop(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        SILENCE_THRESHOLD = 500
        SILENCE_DURATION = 1.0 # seconds

        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE, input=True,
                                      frames_per_buffer=CHUNK)

        frames = []
        silent_chunks = 0
        is_speaking = False
        
        while self.running:
            data = self.stream.read(CHUNK)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            energy = np.abs(audio_chunk).mean()
            
            if energy > SILENCE_THRESHOLD:
                is_speaking = True
                silent_chunks = 0
                frames.append(data)
            else:
                if is_speaking:
                    silent_chunks += 1
                    frames.append(data)
                    # End of speech detected
                    if silent_chunks > (SILENCE_DURATION * RATE / CHUNK):
                        is_speaking = False
                        audio_data = b''.join(frames)
                        self.queue.put(audio_data)
                        frames = []
                        silent_chunks = 0
                else:
                    # Keep a small buffer of pre-speech context?
                    # For now, just drop
                    pass

    def _process_queue(self):
        while self.running:
            try:
                audio_data = self.queue.get(timeout=1.0)
                # Convert raw bytes to float32 numpy array
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                segments, info = self.model.transcribe(audio_np, beam_size=1)
                text = " ".join([segment.text for segment in segments]).strip()
                
                if text and self.callback:
                    logger.info(f"Heard: {text}")
                    self.callback(text)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription error: {e}")
