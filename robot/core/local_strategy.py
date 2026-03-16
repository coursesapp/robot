import logging
import threading
import json
import time
from typing import Dict, Any, List
import numpy as np
from core.interaction import InteractionStrategy

logger = logging.getLogger("LocalStrategy")

class LocalStrategy(InteractionStrategy):
    """
    Implements the current sequential interaction logic:
    STT -> LLM -> TTS -> Action
    """
    def __init__(self, agent_loop):
        self.agent = agent_loop
        self.config = agent_loop.config
        self.llm_busy = False

    def on_speech_start(self):
        # Local strategy: interrupt TTS if it's currently speaking to allow the user to talk
        if self.agent.tts:
            self.agent.tts.interrupt()

    def on_speech(self, text: str, audio_np: np.ndarray, context_metadata: Dict[str, Any]):
        if self.llm_busy:
            logger.warning("LLM is busy, skipping speech input.")
            return

        final_pid = context_metadata.get('final_pid', 'unknown')
        target_identities = [final_pid] if final_pid != "unknown" else []
        current_emotions = context_metadata.get('current_emotions', [])
        tracks = context_metadata.get('tracks', [])
        w, h = context_metadata.get('wh', (640, 480))

        # Build Context
        soc_data = [self.agent.social_memory.get(pid) for pid in target_identities]
        
        objects_spatial = []
        for t in tracks:
            if t.class_name != 'person':
                x1, y1, x2, y2 = t.box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                pos_x = "left" if cx < w/3 else "right" if cx > 2*w/3 else "center"
                pos_y = "top" if cy < h/3 else "bottom" if cy > 2*h/3 else "middle"
                objects_spatial.append({"label": t.class_name, "position": f"{pos_y}-{pos_x}"})
        
        deep_memory = []
        if target_identities and not text.startswith("[System"):
            deep_memory = self.agent.vector_memory.search_past(target_identities[0], text, n_results=5)

        context = {
            "identities": target_identities,
            "social_data": soc_data,
            "emotions": current_emotions,
            "objects": objects_spatial,
            "history": list(self.agent.context_history),
            "deep_memory": deep_memory,
            "last_thought": self.agent.last_thought,
            "available_actions": self.agent.action_library.get_available_actions_schema(),
            "time": time.ctime(),
            'identity_conflict': context_metadata.get('identity_conflict', False),
            'conflict_message': context_metadata.get('conflict_message', "")
        }

        prompt = self.agent.prompt_engine.build_prompt(text, context)
        
        self.llm_busy = True
        self.agent.context_history.append(f"user: {text}")
        
        # Start Worker Thread
        threading.Thread(target=self._llm_worker, args=(prompt, text, target_identities), daemon=True).start()

    def _llm_worker(self, prompt, user_text, tids):
        try:
            self.agent.dashboard.update_state(lambda st: setattr(st, 'llm_status', 'Thinking...') or setattr(st, 'llm_busy', True))
            # Explicitly request enough tokens for the JSON structure
            response = self.agent.llm.generate(prompt, max_tokens=1024)
            self.agent.dashboard.update_state(lambda st: setattr(st, 'llm_status', 'Idle') or setattr(st, 'llm_busy', False))
            
            response_audio = ""
            try:
                # 1. More robust JSON extraction
                json_str = response.strip()
                if "```json" in json_str: 
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str: 
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                else:
                    # Look for the first {
                    start = json_str.find("{")
                    if start != -1:
                        json_str = json_str[start:]
                
                # Pre-parse cleanup: try to repair truncated JSON
                if json_str.count('{') > json_str.count('}'):
                    logger.warning("JSON appears truncated, attempting repair...")
                    # Case 1: Cut off mid-summary or mid-save_to_memory
                    if '"summary":' in json_str:
                         if not json_str.strip().endswith('}'):
                            # Try to close the string if needed
                            if json_str.count('"') % 2 != 0: json_str += '"'
                            json_str += ', "save_to_memory": false}'
                    # Case 2: Cut off right after response
                    elif '"response":' in json_str:
                         if not json_str.strip().endswith('}'):
                            if json_str.count('"') % 2 != 0: json_str += '"'
                            json_str += ', "actions": [], "summary": "Truncated response", "save_to_memory": false}'
                    else:
                        # General rescue: close strings and objects
                        if json_str.count('"') % 2 != 0: json_str += '"'
                        while json_str.count('{') > json_str.count('}'):
                            json_str += '}'
                
                data = json.loads(json_str)
                
                if not isinstance(data, dict):
                    raise ValueError(f"Parsed JSON is {type(data).__name__}, expected dict. Content: {json_str[:100]}")

                # Update thoughts
                thought = data.get("internal_thought", "")
                if thought:
                    self.agent.last_thought = thought
                    self.agent.dashboard.update_state(lambda st, t=thought: st.add_event(f"🧠 Thought: {t}") or setattr(st, 'current_thought', t))
                
                response_audio = data.get("response", "").strip()
                
                # Execute actions
                for act in data.get("actions", []):
                    if isinstance(act, dict):
                        self.agent.action_library.execute(act.get("action"), act.get("parameters", {}))
                    else:
                        logger.warning(f"Skipping malformed action (not a dict): {act}")
                    
                # Handle memory
                save_to_memory = data.get("save_to_memory", False)
                summary_text = data.get("summary", "")
                if save_to_memory and summary_text and tids:
                    pid0 = tids[0]
                    self.agent.social_memory.update(pid0, {"summary": summary_text})
                    self.agent.vector_memory.add_interaction(pid0, f"[Key Info]: {summary_text}", role="system")
                    self.agent.dashboard.update_state(lambda st: setattr(st, 'avatar_state', 'happy'))
            except Exception as e:
                logger.warning(f"JSON Parsing failed: {e}. Raw response: {response}")
                # If parsing fails, don't speak the raw response if it looks like JSON
                if response.strip().startswith("{") or '"response":' in response:
                    response_audio = "I'm sorry, I had trouble processing that thought."
                else:
                    response_audio = response # Treat as plain text fallback
            
            self.agent.context_history.append(f"agent: {response_audio}")
            self.agent.tts.speak(response_audio)

            # Smart Context Compression
            if len(self.agent.context_history) >= 15: # Increaced from 10 as per hackathon study
                try:
                    summary = self.agent.summarizer.summarize(list(self.agent.context_history))
                    if summary:
                        self.agent.context_history = [f"[Conversation Summary]: {summary}"]
                        if tids:
                            pid0 = tids[0]
                            self.agent.social_memory.update(pid0, {"summary": summary})
                            self.agent.vector_memory.add_interaction(pid0, f"[Conversation Summary]: {summary}", role="system")
                except Exception as se:
                    logger.warning(f"Summarizer failed: {se}")
            
            if tids:
                pid0 = tids[0]
                self.agent.vector_memory.add_interaction(pid0, user_text, role="user")
                self.agent.vector_memory.add_interaction(pid0, response_audio, role="agent")
                
                facts = self.agent.info_extractor.extract(user_text)
                if facts:
                    self.agent.social_memory.update(pid0, facts)
        except Exception as le:
            logger.error(f"LLM Worker error: {le}")
        finally:
            self.llm_busy = False

    def on_vision(self, frame, metadata):
        pass

    def stop(self):
        self.llm_busy = False
