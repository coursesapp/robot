import re
import sounddevice as sd
from kokoro import KPipeline

class SentenceBuffer:
    def __init__(self):
        self.buffer = ""
        # Defines what counts as the "End of a sentence" (. ? ! :)
        self.split_pattern = re.compile(r'(.*?[.?!:\n]+)\s*')

    def add_token(self, token):
        self.buffer += token
        sentences = []
        while True:
            match = self.split_pattern.match(self.buffer)
            if match:
                sentence = match.group(1)
                sentences.append(sentence)
                self.buffer = self.buffer[len(sentence):].lstrip()
            else:
                break
        return sentences

    def flush(self):
        # Return whatever is left (even if no punctuation)
        return [self.buffer.strip()] if self.buffer.strip() else []

class VoiceEngine:
    def __init__(self):
        print("--- 🔈 Loading Kokoro Voice... ---")
        self.pipeline = KPipeline(lang_code='a') 
        print("--- ✅ Voice Ready ---")

    def say(self, text):
        """Standard method for simple status messages"""
        print(f"🤖 BOT: {text}")
        self._generate_and_play(text)

    def speak_stream(self, token_generator):
        """
        The Streaming Method:
        Takes words from Mistral -> Buffers them -> Speaks full sentences.
        """
        buffer = SentenceBuffer()
        print("🤖 BOT (Streaming): ", end="", flush=True)

        for token in token_generator:
            print(token, end="", flush=True) # Print token to screen
            
            # Check if we have a full sentence yet
            sentences = buffer.add_token(token)
            
            for sentence in sentences:
                self._generate_and_play(sentence)

        # Flush the remainder
        for sentence in buffer.flush():
            self._generate_and_play(sentence)
        print()

    def _generate_and_play(self, text):
        if not text.strip(): return
        
        # Generate Audio
        generator = self.pipeline(
            text, voice='af_heart', speed=1.0, split_pattern=r'\n+'
        )
        
        # Play Audio
        for i, (gs, ps, audio) in enumerate(generator):
            sd.play(audio, 24000)
            sd.wait()