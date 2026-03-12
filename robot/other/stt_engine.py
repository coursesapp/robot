import queue
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from colorama import Fore, Style, init

init(autoreset=True)

class HearingEngine:
    def __init__(self, model_size="base"):
        print(f"{Fore.CYAN}--- 🧠 Loading Ear ({model_size}) ... ---{Style.RESET_ALL}")
        
        try:
            self.model = WhisperModel(model_size, device="cpu", compute_type="float32")
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {e}{Style.RESET_ALL}")
            raise e

        self.audio_queue = queue.Queue()
        self.is_running = True
        self.is_paused = False
        
        # --- SETTINGS ---
        self.SAMPLE_RATE = 16000
        self.SILENCE_THRESHOLD = 0.02 
        self.SILENCE_DURATION = 3.0     # Wait 3s of silence to finalize
        self.MAX_CHUNK_DURATION = 15.0  # Process chunks to prevent freezing
        
        self.VOCAB = "Mahmoud, Abdullah, Egypt, Grad Project, Engineer, Habibi, Robot, AI"
        
        # NEW: A hidden buffer to stitch long speeches together
        self.accumulated_text = ""

        try:
            self.stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE, 
                blocksize=2048, 
                channels=1, 
                callback=self._audio_callback
            )
            self.stream.start()
            print(f"{Fore.GREEN}--- ✅ Ear is Listening! (Long-Speech Mode) ---{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Mic Error: {e}{Style.RESET_ALL}")

    def _audio_callback(self, indata, frames, time, status):
        if self.is_paused: return 
        if status: pass
        self.audio_queue.put(indata.copy())

    def clear_memory(self):
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.accumulated_text = "" # Clear text buffer too
        print(f"{Fore.BLUE}[Memory Flushed]{Style.RESET_ALL}")

    def pause(self):
        self.is_paused = True
    
    def resume(self):
        self.is_paused = False
        self.clear_memory()

    def _transcribe_chunk(self, audio_data):
        """Helper to transcribe a piece of audio without yielding it yet"""
        try:
            segments, info = self.model.transcribe(
                audio_data, 
                beam_size=2, 
                temperature=0.0,
                initial_prompt=self.VOCAB,
                vad_filter=True
            )
            segments = list(segments)
            if segments:
                text = " ".join([s.text for s in segments]).strip()
                if len(text) > 2 and "Thank you" not in text:
                    return text
        except Exception as e:
            print(f"{Fore.RED}Transcription Error: {e}{Style.RESET_ALL}")
        return ""

    def listen(self):
        current_audio = np.array([], dtype=np.float32)
        silence_start_time = None
        has_speech = False
        startup_grace_period = time.time() + 2.0 

        print(f"{Fore.CYAN}Waiting for speech...{Style.RESET_ALL}")

        while self.is_running:
            try:
                try:
                    new_data = self.audio_queue.get(timeout=1.0)
                    new_data = new_data.flatten()
                except queue.Empty:
                    continue

                if time.time() < startup_grace_period:
                    continue

                # 1. ACCUMULATE AUDIO
                current_audio = np.concatenate((current_audio, new_data))
                current_duration = len(current_audio) / self.SAMPLE_RATE
                
                # 2. CHECK VOLUME
                volume = np.sqrt(np.mean(new_data**2))
                
                if volume < self.SILENCE_THRESHOLD:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                else:
                    silence_start_time = None 
                    has_speech = True

                # --- LOGIC SPLIT ---
                
                # CASE A: AUTO-CHUNK (User is still talking, but buffer is full)
                if current_duration > self.MAX_CHUNK_DURATION:
                    print(f"{Fore.MAGENTA}🔄 Stitching chunk ({current_duration:.1f}s)...{Style.RESET_ALL}")
                    
                    # Transcribe current part
                    chunk_text = self._transcribe_chunk(current_audio)
                    
                    # Add to hidden buffer (Do NOT Yield yet)
                    if chunk_text:
                        self.accumulated_text += " " + chunk_text
                        print(f"{Fore.BLUE}   Current Buffer: {self.accumulated_text[:50]}...{Style.RESET_ALL}")
                    
                    # Reset audio buffer ONLY (Keep listening)
                    current_audio = np.array([], dtype=np.float32)
                    silence_start_time = None
                    has_speech = False # Reset speech flag for next chunk

                # CASE B: SILENCE (User finished talking)
                elif silence_start_time and (time.time() - silence_start_time > self.SILENCE_DURATION):
                    
                    # Check if we have any pending audio to process
                    if len(current_audio) > self.SAMPLE_RATE * 1.0 and has_speech:
                         print(f"{Fore.YELLOW}⚡ Finalizing Speech...{Style.RESET_ALL}")
                         last_chunk = self._transcribe_chunk(current_audio)
                         if last_chunk:
                             self.accumulated_text += " " + last_chunk

                    # NOW we yield the Full Story
                    final_output = self.accumulated_text.strip()
                    
                    if len(final_output) > 2:
                        yield final_output
                    
                    # Reset Everything
                    current_audio = np.array([], dtype=np.float32)
                    self.accumulated_text = ""
                    silence_start_time = None
                    has_speech = False
                        
            except Exception as e:
                print(f"Loop Error: {e}")
                break