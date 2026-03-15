"""
Quick test script for STT engine (standalone).
Run from the project root:
    python test_stt.py
"""
import time
import sys

def on_speech(text: str):
    print(f"\n✅ Heard: {text}\n")

print("🎤 Loading STT engine...")
from audio.stt import STTEngine

try:
    stt = STTEngine(model_size="small", callback=on_speech)
    stt.start()
    print("✅ STT engine loaded successfully!")
    print("📌 Device: CUDA" if "cuda" in str(stt.model.__dict__) else "📌 Device: CPU")
    print("\n🗣  Start speaking (Ctrl+C to stop)...\n")
    
    # Keep the script alive while waiting for speech
    while True:
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n\n🛑 Test stopped.")
    stt.stop()
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
