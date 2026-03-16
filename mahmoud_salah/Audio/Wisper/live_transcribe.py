import queue
import threading
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# --- Configuration ---
MODEL_TYPE = "base.en"      # Model to use (e.g., "tiny.en", "base.en", "small.en")
SAMPLE_RATE = 16000         # Whisper's required sample rate
BLOCK_SIZE = 1024           # Number of audio frames per block (controls latency)
SILENCE_THRESHOLD = 400     # RMS threshold for detecting silence (adjust as needed)
SILENCE_DURATION = 3        # MODIFIED: Seconds of silence to trigger a transcription

# --- Global Variables ---
audio_queue = queue.Queue()
is_recording = True

def audio_callback(indata, frames, time, status):
    """This is called from a separate thread for each audio block."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def process_audio_thread():
    """Main processing loop to detect speech and transcribe."""
    print(f"Loading faster-whisper model: {MODEL_TYPE}...")
    # Using int8 quantization for better performance on CPU
    model = WhisperModel(MODEL_TYPE, device="cpu", compute_type="int8")
    print(f"Model loaded on CPU.")

    # Create a prompt with names, jargon, or difficult words.
    # Use full sentences for better results.
    prompt = "Mahmoud's phone is on the table. This is a test of the system."
    print(f"Using initial prompt: '{prompt}'")

    audio_buffer = []
    silence_start_time = None

    while is_recording:
        try:
            audio_chunk = audio_queue.get(timeout=1)
            audio_buffer.append(audio_chunk)

            # Simple Voice Activity Detection (VAD) based on RMS volume
            rms = np.sqrt(np.mean(audio_chunk**2))

            if rms < SILENCE_THRESHOLD:
                if silence_start_time is None:
                    silence_start_time = time.time()
                # If silence persists, and we have audio in the buffer, transcribe it
                elif time.time() - silence_start_time > SILENCE_DURATION:
                    if len(audio_buffer) > 10: # Avoid transcribing short noises
                        process_and_transcribe(model, audio_buffer, prompt)
                    audio_buffer = []
                    silence_start_time = None
            else:
                # Reset silence timer if speech is detected
                silence_start_time = None

        except queue.Empty:
            # If the queue is empty, check if there was a trailing silence
            if silence_start_time and time.time() - silence_start_time > SILENCE_DURATION:
                 if len(audio_buffer) > 10:
                    process_and_transcribe(model, audio_buffer, prompt)
                 audio_buffer = []
                 silence_start_time = None
            continue

def process_and_transcribe(model, audio_buffer, prompt):
    """Concatenate, convert, and transcribe the collected audio buffer."""
    print("Silence detected, processing audio for transcription...")
    
    full_audio = np.concatenate(audio_buffer)
    audio_float32 = full_audio.flatten().astype(np.float32)

    try:
        # --- ADDED VAD PARAMETERS FOR LONGER SENTENCES ---
        # This tells the model's internal VAD to be more patient with pauses.
        vad_parameters = dict(min_silence_duration_ms=3000)

        segments, info = model.transcribe(
            audio_float32,
            beam_size=5,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters=vad_parameters  # Apply our custom VAD settings
        )
        
        full_text = ""
        for segment in segments:
            full_text += segment.text + " "
            
        if full_text.strip():
            print("Transcription:", full_text.strip())
            
    except Exception as e:
        print(f"An error occurred during transcription: {e}")

if __name__ == "__main__":
    # Start the processing thread
    processing_thread = threading.Thread(target=process_audio_thread)
    processing_thread.daemon = True
    processing_thread.start()

    # Start recording from the microphone in the main thread
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE,
                             blocksize=BLOCK_SIZE,
                             channels=1,
                             dtype='float32',
                             callback=audio_callback):
            print("Recording... Speak into your microphone. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping recording.")
        is_recording = False
        print("Stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")