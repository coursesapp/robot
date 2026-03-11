import soundfile as sf
from kokoro import KPipeline
import io
import sounddevice as sd # We will use this to play audio directly

# If you don't have sounddevice, run: pip install sounddevice
# If you prefer saving to file only, we can remove it.

print("--- Loading Kokoro AI (This takes a few seconds)... ---")

# 1. Initialize the AI
# lang_code='a' is American English.
# This automagically downloads the model (~80MB) the first time you run it.
pipeline = KPipeline(lang_code='a') 

print("\n--- AI Ready! ---")
print("Type 'exit' to quit.")

while True:
    text = input(">> ")
    
    if text.lower() == 'exit':
        break
        
    # 2. Generate Audio
    # voice='af_heart' is the most famous realistic voice
    # speed=1 is normal.
    generator = pipeline(
        text, voice='af_heart', 
        speed=1, split_pattern=r'\n+'
    )
    
    # 3. Play the audio
    for i, (gs, ps, audio) in enumerate(generator):
        # Save to a temp file if you want to keep it
        # sf.write(f'output_{i}.wav', audio, 24000)
        
        # Play immediately (requires 'pip install sounddevice')
        print(f"Speaking...")
        sd.play(audio, 24000)
        sd.wait()

print("Goodbye!")