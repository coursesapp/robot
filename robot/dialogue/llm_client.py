import logging
import threading
from typing import Optional, Dict, Any, List
import requests

logger = logging.getLogger("LLMClient")

class LLMClient:
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {"engine": "ollama", "model_name": "qwen2.5:3b"}
            
        self.engine = config.get("engine", "ollama")
        self.model_name = config.get("model_name", "qwen2.5:3b")
        self.llama_api_url = config.get("llama_api_url", "http://127.0.0.1:8080/v1/chat/completions")
        self.groq_api_key = config.get("groq_api_key", "")
        self.groq_model = config.get("groq_model", "llama3-70b-8192")
        self.lock = threading.Lock()
        
        if self.engine == "ollama":
            import ollama
            # Check connection on init
            try:
                ollama.list()
                logger.info(f"Connected to Ollama. Using model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not connect to Ollama: {e}. Is it running?")
        elif self.engine == "llamacpp":
            logger.info(f"Configured for llama.cpp server at {self.llama_api_url} using model: {self.model_name}")
        elif self.engine == "groq":
            if not self.groq_api_key:
                logger.warning("Groq engine selected but no API key found in settings!")
            logger.info(f"Configured for Groq Cloud API using model: {self.groq_model}")

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7, stop: Optional[List[str]] = None) -> str:
        if not prompt:
            return ""

        stop_tokens = stop or ["User:", "\n\n"]
        
        try:
            if self.engine == "ollama":
                import ollama
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "stop": stop_tokens
                    }
                )
                return response['response'].strip()
                
            elif self.engine == "llamacpp" or self.engine == "groq":
                # Both use OpenAI compatible chat completions endpoint
                headers = {"Content-Type": "application/json"}
                url = self.llama_api_url
                
                if self.engine == "groq":
                    url = "https://api.groq.com/openai/v1/chat/completions"
                    headers["Authorization"] = f"Bearer {self.groq_api_key}"
                    # Groq: simple, clean payload (no custom stop tokens)
                    payload = {
                        "model": self.groq_model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                else:
                    # llama-server/OpenAI compat payload
                    if "chat/completions" in self.llama_api_url:
                        payload = {
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "stop": stop_tokens
                        }
                    else:
                        payload = {
                            "prompt": prompt,
                            "n_predict": max_tokens,
                            "temperature": temperature,
                            "stop": stop_tokens
                        }

                response = requests.post(url, headers=headers, json=payload, timeout=120)
                if response.status_code != 200:
                    logger.error(f"LLM API error ({self.engine}) {response.status_code}: {response.text}")
                    return "..."
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "message" in choice:
                        return choice["message"]["content"].strip()
                    elif "text" in choice:
                        return choice["text"].strip()
                return ""
            
        except Exception as e:
            logger.error(f"LLM generation error ({self.engine}): {e}")
            return "..."
