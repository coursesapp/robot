import os

try:
    import nvidia
    _nvidia_base = list(nvidia.__path__)[0]
    dll_paths = []
    for _pkg in os.listdir(_nvidia_base):
        _bin = os.path.join(_nvidia_base, _pkg, "bin")
        if os.path.isdir(_bin):
            os.add_dll_directory(_bin)
            dll_paths.append(_bin)
    # Also add to PATH so worker threads can find them
    if dll_paths:
        os.environ["PATH"] = ";".join(dll_paths) + ";" + os.environ.get("PATH", "")
except Exception as e:
    print(f"Warning: could not add NVIDIA DLL dirs: {e}")

# Now safe to import CUDA-dependent libraries
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
    def __init__(self, model_size: str = "small", callback: Optional[Callable[[str], None]] = None):
        self.callback = callback
        self.running = False

        # --- SETTINGS (ported from HearingEngine) ---
        self.SAMPLE_RATE = 16000
        self.CHUNK = 2048
        self.SILENCE_THRESHOLD = 0.02        # RMS-based, same as HearingEngine
        self.SILENCE_DURATION = 3.0          # Seconds of silence before finalizing
        self.MAX_CHUNK_DURATION = 30.0       # Max seconds before auto-stitching a chunk

        # Bias the model towards these specific terms
        self.VOCAB = "Mahmoud, Abdullah, Egypt, Grad Project, Engineer, Habibi, Robot, AI, apple"
        

        logger.info(f"Loading faster-whisper ({model_size})...")
        try:
            # float32 is best than int8
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info(f"Whisper loaded on CPU ({model_size}).")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise e

        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Audio queue feeds from the listener thread into the processor thread
        self._audio_queue: queue.Queue[bytes] = queue.Queue()

    # ------------------------------------------------------------------
    # Public controls
    # ------------------------------------------------------------------

    def start(self):
        self.running = True

        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

        logger.info("STT listening (long-speech mode)...")

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    # ------------------------------------------------------------------
    # Internal: audio capture
    # ------------------------------------------------------------------

    def _listen_loop(self):
        """Continuously reads raw PCM chunks from the microphone into the queue."""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )
        while self.running:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            self._audio_queue.put(data)

    # ------------------------------------------------------------------
    # Internal: transcription helpers
    # ------------------------------------------------------------------

    def _transcribe_chunk(self, audio_np: np.ndarray) -> str:
        """Transcribe a float32 audio array; returns text or empty string."""
        try:
            segments, _ = self.model.transcribe(
                audio_np,
                beam_size=2,
                vad_filter=True,
                temperature=0.0,
                initial_prompt=self.VOCAB,
                language="en"
            )
            segments = list(segments)
            if segments:
                text = " ".join(s.text for s in segments).strip()
                # Filter Whisper hallucinations
                if len(text) > 2 and "Thank you" not in text:
                    return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        return ""

    # ------------------------------------------------------------------
    # Internal: long-speech processing loop (ported from HearingEngine.listen)
    # ------------------------------------------------------------------

    def _process_loop(self):
        """
        Accumulates raw audio frames, stitches mid-speech chunks when the
        buffer exceeds MAX_CHUNK_DURATION, and fires the callback with the
        complete utterance once SILENCE_DURATION seconds of silence are detected.
        """
        current_audio = np.array([], dtype=np.float32)
        accumulated_text: str = ""
        silence_start_time: Optional[float] = None
        has_speech = False

        # Brief grace period so the mic can settle after opening
        startup_grace = time.time() + 2.0

        while self.running:
            # --- Drain one chunk from the queue ---
            try:
                raw = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if time.time() < startup_grace:
                continue

            # Convert int16 PCM → float32 [-1, 1]
            chunk_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # 1. ACCUMULATE
            current_audio = np.concatenate((current_audio, chunk_np))
            current_duration = len(current_audio) / self.SAMPLE_RATE

            # 2. CHECK VOLUME (RMS, same metric as HearingEngine)
            volume = np.sqrt(np.mean(chunk_np ** 2))

            if volume >= self.SILENCE_THRESHOLD:
                silence_start_time = None   # Reset silence timer on speech
                has_speech = True
            else:
                if silence_start_time is None:
                    silence_start_time = time.time()

            # ----------------------------------------------------------------
            # CASE A: Buffer full while user is still talking → stitch chunk
            # ----------------------------------------------------------------
            if current_duration > self.MAX_CHUNK_DURATION:
                logger.info(f"Stitching chunk ({current_duration:.1f}s)…")
                chunk_text = self._transcribe_chunk(current_audio)
                if chunk_text:
                    accumulated_text += " " + chunk_text
                    logger.debug(f"Buffer so far: {accumulated_text[:80]}…")

                # Reset audio buffer only; keep listening
                current_audio = np.array([], dtype=np.float32)
                silence_start_time = None
                has_speech = False

            # ----------------------------------------------------------------
            # CASE B: Silence detected → finalize the full utterance
            # ----------------------------------------------------------------
            elif silence_start_time and (time.time() - silence_start_time > self.SILENCE_DURATION):

                # Transcribe whatever audio is left in the buffer
                if len(current_audio) > self.SAMPLE_RATE * 1.0 and has_speech:
                    logger.info("Finalizing speech…")
                    last_text = self._transcribe_chunk(current_audio)
                    if last_text:
                        accumulated_text += " " + last_text

                final_output = accumulated_text.strip()

                if final_output and self.callback:
                    logger.info(f"Heard: {final_output}")
                    # Provide text and the raw audio for Speaker Recognition
                    self.callback(final_output, current_audio)

                # Reset everything for the next utterance
                current_audio = np.array([], dtype=np.float32)
                accumulated_text = ""
                silence_start_time = None
                has_speech = False


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
# Hello, this is a test of the speech recognition system.
# I have 2 apples and 15 bananas.
# This system should be able to recognize speech even when I speak very fast
# without stopping between words.
# This program converts speech into text.

# Hey assistant, can you tell me what the weather is like today?
# Please remind me to finish my project tomorrow morning.
# Can you search the internet for the latest news about artificial intelligence?
# Artificial intelligence and machine learning are transforming many industries including healthcare finance transportation and robotics by enabling computers to learn from data and make intelligent decisions without explicit programming.