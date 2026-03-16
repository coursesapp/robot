from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import os

# --- Configuration ---
MODEL_TYPE = "base.en" # English-only model
AUDIO_FILE = os.path.abspath("test_audio.wav")
SAMPLE_RATE = 16000 # Whisper's required sample rate
DURATION = 5 # seconds

def create_dummy_audio_file():
    """Creates a dummy WAV file with a simple sine wave for testing."""
    print(f"Creating a dummy audio file: {AUDIO_FILE}")
    t = np.linspace(0., DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * 440. * t)
    
    sf.write(AUDIO_FILE, data.astype(np.int16), SAMPLE_RATE)
    print("Dummy audio file created.")

def main():
    """Loads the faster-whisper model and transcribes the audio file."""
    create_dummy_audio_file()
    
    print(f"\nLoading faster-whisper model: {MODEL_TYPE}...")
    # This is a more optimized way to run on CPU
    model = WhisperModel(MODEL_TYPE, device="cpu", compute_type="int8")
    print("Model loaded.")

    print(f"\nTranscribing {AUDIO_FILE}...")
    try:
        # The transcribe function is slightly different and returns an iterator
        segments, info = model.transcribe(AUDIO_FILE, beam_size=5)

        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

        print("\n--- Transcription Result ---")
        
        full_text = ""
        for segment in segments:
            # The.text attribute contains the transcribed text for each segment
            full_text += segment.text + " "
        
        # Since the audio is just a tone, the text will be empty. This is expected.
        print(full_text.strip())
        print("--------------------------")
        
    except Exception as e:
        print(f"An error occurred during transcription: {e}")

if __name__ == "__main__":
    main()