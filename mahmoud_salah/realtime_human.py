import queue
import threading
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from colorama import Fore, Style, init

init(autoreset=True)

# --- ⚙️ SETTINGS ---
MODEL_TYPE = "base.en"
SAMPLE_RATE = 16000
BLOCK_SIZE = 2048 
SILENCE_THRESHOLD = 0.08    
SILENCE_DURATION = 3.5      

# Context for the AI
CUSTOM_VOCABULARY = "Mahmoud, Abdullah, Egypt, Grad Project, Engineer, Habibi, Robot, AI"

# Hallucination Filters
BANNED_PHRASES = [
    "Thank you", "Thanks", "You", "Copyright", "Subtitle", 
    "Amara", "org", "audio", "noise", "."
]

audio_queue = queue.Queue()
is_recording = True

def audio_callback(indata, frames, time, status):
    if status: pass 
    audio_queue.put(indata.copy())

def is_valid_text(text):
    if not text or len(text.strip()) < 2: return False
    text_lower = text.lower().strip()
    for phrase in BANNED_PHRASES:
        if phrase.lower() in text_lower:
            if len(text_lower) < len(phrase) + 5: return False
    return True

def process_audio():
    print(f"{Fore.CYAN}--- 🧠 Loading AI Ear ({MODEL_TYPE}) ... ---{Style.RESET_ALL}")
    model = WhisperModel(MODEL_TYPE, device="cpu", compute_type="int8")
    
    print(f"{Fore.GREEN}--- ✅ Production Mode Ready! ---{Style.RESET_ALL}")

    current_audio = np.array([], dtype=np.float32)
    silence_start_time = None
    
    while is_recording:
        try:
            try:
                new_data = audio_queue.get(timeout=1.0)
                new_data = new_data.flatten()
            except queue.Empty:
                continue

            current_audio = np.concatenate((current_audio, new_data))
            volume = np.sqrt(np.mean(new_data**2))
            
            if volume < SILENCE_THRESHOLD:
                if silence_start_time is None:
                    silence_start_time = time.time()
                    print(f"\r{Fore.CYAN}⏳ ...{' '*20}", end="", flush=True)
            else:
                if silence_start_time is not None:
                    print(f"\r{Fore.CYAN}🎤 Hearing...{' '*10}", end="", flush=True)
                silence_start_time = None 

            if silence_start_time and (time.time() - silence_start_time > SILENCE_DURATION):
                if len(current_audio) > SAMPLE_RATE * 1.0: 
                    
                    print(f"\r{Fore.YELLOW}⚡ Processing...{' '*20}", end="", flush=True)
                    
                    # 1. Run Whisper
                    segments, info = model.transcribe(
                        current_audio, 
                        beam_size=5,
                        initial_prompt=CUSTOM_VOCABULARY,
                        vad_filter=True
                    )
                    
                    # 2. Convert generator to list so we can check it
                    segments = list(segments)

                    if segments:
                        # 3. Check the probability of the FIRST segment
                        # (no_speech_prob lives inside the segment, not info!)
                        segment_probability = segments[0].no_speech_prob
                        
                        if segment_probability < 0.6: # 60% confidence it is speech
                            full_text = " ".join([s.text for s in segments]).strip()
                            
                            if is_valid_text(full_text):
                                print(f"\r{Fore.GREEN}🗣️  Clean Output: {full_text}{' '*20}")
                            else:
                                print(f"\r{Fore.RED}🗑️  Ignored Hallucination: '{full_text}'{' '*10}")
                        else:
                            print(f"\r{Fore.BLUE}🔇 Noise Detected (Ignored){' '*10}")
                    else:
                        # VAD filtered everything out
                        print(f"\r{Fore.BLUE}🔇 Silence (VAD){' '*10}")

                    current_audio = np.array([], dtype=np.float32)
                    silence_start_time = None
            
        except Exception as e:
            print(f"{Fore.RED}Error: {e}")
            break

if __name__ == "__main__":
    processing_thread = threading.Thread(target=process_audio)
    processing_thread.daemon = True
    processing_thread.start()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, callback=audio_callback):
            while True: time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}--- 🛑 System Stopped ---")
        is_recording = False