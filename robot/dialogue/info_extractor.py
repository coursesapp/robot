import json
import logging
from typing import Dict, Any
from dialogue.llm_client import LLMClient

logger = logging.getLogger("InfoExtractor")

class InfoExtractor:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.system_prompt = (
            "You are a precise personal information extractor. "
            "Extract from the conversation any of the following if explicitly mentioned:\n"
            "- name: person's first name\n"
            "- job: their profession\n"
            "- interests: list of hobbies/topics they like\n"
            "- relations: people they mentioned (e.g. 'my brother Ahmed')\n"
            "- facts: any other memorable personal facts\n\n"
            "Return ONLY valid JSON. If nothing is found, return {}.\n"
            "Example output: {\"name\": \"Ahmed\", \"job\": \"engineer\", \"interests\": [\"chess\", \"hiking\"]}"
        )

    def extract(self, user_text: str) -> Dict[str, Any]:
        prompt = f"<|system|>\n{self.system_prompt}\n<|user|>\n{user_text}\n<|assistant|>\n"
        response = self.llm.generate(prompt)
        
        # Try to parse JSON from the response
        try:
            # Clean up response if it contains markdown code blocks
            clean_resp = response.strip()
            if clean_resp.startswith("```json"):
                clean_resp = clean_resp[7:]
            elif clean_resp.startswith("```"):
                clean_resp = clean_resp[3:]
            if clean_resp.endswith("```"):
                clean_resp = clean_resp[:-3]
                
            data = json.loads(clean_resp.strip())
            
            # Additional validation
            if not isinstance(data, dict):
                return {}
            
            # Filter empty values
            filtered_data = {k: v for k, v in data.items() if v and v != "unknown" and v != ["unknown"]}
            
            if filtered_data:
                logger.debug(f"Extracted info: {filtered_data}")
            return filtered_data
            
        except json.JSONDecodeError:
            logger.debug("Failed to parse extractor response as JSON.")
            return {}
        except Exception as e:
            logger.error(f"Error extracting info: {e}")
            return {}
