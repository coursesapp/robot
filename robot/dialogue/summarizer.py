import logging
from typing import List
from dialogue.llm_client import LLMClient

logger = logging.getLogger("Summarizer")

class Summarizer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.system_prompt = (
            "You are a memory assistant. Summarize the following conversation "
            "into 2-3 concise sentences capturing the key topics, facts shared, and tone. "
            "Be factual and neutral in your description. Output ONLY the summary text without introduction."
        )

    def summarize(self, history: List[str]) -> str:
        if not history:
            return ""
            
        history_text = "\n".join(history)
        prompt = f"<|system|>\n{self.system_prompt}\n<|user|>\n{history_text}\n<|assistant|>\n"
        
        logger.debug("Generating conversation summary...")
        response = self.llm.generate(prompt)
        
        summary = response.strip()
        logger.debug(f"Summary generated: {summary}")
        return summary
