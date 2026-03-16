import vosk
import sounddevice as sd
import queue
import json

class SpeechToText:
    def __init__(self, model_lang="en-us"):
        """Initializes the STT module."""
        self.model = vosk.Model(lang=model_lang)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.audio_queue = queue.Queue()
        print(" Speech-to-Text module initialized.")

    def _audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, flush=True)
        self.audio_queue.put(bytes(indata))

    def listen_and_transcribe(self):
        """Listens for audio and prints the transcription."""
        print(" Listening... Press Ctrl+C to stop.")
        # The 'with' statement ensures the stream is properly closed.
        with sd.RawInputStream(samplerate=16000, blocksize=8000, device=None, 
                               dtype='int16', channels=1, callback=self._audio_callback):
            
            while True:
                data = self.audio_queue.get()
                if self.recognizer.AcceptWaveform(data):
                    result_json = self.recognizer.Result()
                    result_dict = json.loads(result_json)
                    final_text = result_dict.get('text', '')
                    if final_text:
                        print(f" Detected: {final_text}")

# --- Main execution block ---
if __name__ == '__main__':
    stt = SpeechToText()
    try:
        stt.listen_and_transcribe()
    except KeyboardInterrupt:
        print("\n Stopped.")