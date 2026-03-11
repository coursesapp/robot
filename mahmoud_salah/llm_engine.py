from gpt4all import GPT4All
import os

class BrainEngine:
    def __init__(self, model_name="Mistral"):
        # 1. EXACT FILENAME YOU DOWNLOADED
        self.model_name = "mistral-7b-instruct-v0.1.Q5_0.gguf"
        
        print(f"--- 🧠 Loading Brain ({self.model_name}) ---")
        
        # Check if file is really there
        if not os.path.exists(self.model_name):
            print(f"❌ ERROR: I cannot find '{self.model_name}' in this folder!")
            print("Please move the .gguf file to X:\\Work\\Grad_proj\\")
            raise FileNotFoundError("Model file missing")

        # 2. LOAD LOCAL FILE (allow_download=False prevents internet usage)
        self.model = GPT4All(self.model_name, model_path=".", allow_download=False)
        
        print(f"--- ✅ Brain Ready! ---")

    def think_stream(self, user_text):
        """
        Generates the answer word-by-word (Streaming).
        """
        # Mistral uses the [INST] format
        prompt = f"[INST] You are a helpful Robot assistant. Keep answers short (1-2 sentences). {user_text} [/INST]"
        
        try:
            # streaming=True allows speaking while thinking
            generator = self.model.generate(prompt, streaming=True, max_tokens=200)
            
            for token in generator:
                yield token
                
        except Exception as e:
            yield f"Error: {e}"