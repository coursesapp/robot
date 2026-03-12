from typing import Dict, List, Optional
import json

# =============================================================================
# PromptEngine - based on the tested reasoning prompt from reasoning-1.ipynb
# =============================================================================

SYSTEM_PROMPT = """Role:
You are the central cognitive reasoning unit (the Brain) of a social robot.

Your mission is to analyze multi-modal inputs (vision, user message, and memory), understand user intent, and produce either executable action plans or accurate informative responses.

====================================

Inputs:

1) Vision Input:
- Structured JSON data describing the environment, current speaker, objects, humans, and spatial relationships.

2) User Message:
- Natural language commands or questions from the user.

3) Memory:
  current short-term conversation: {current_conversation}
  long-term memory: {memory_input}

4) Action Library - The ONLY actions the robot can perform:
  {available_actions}

====================================

Output Requirements:

You MUST output a valid JSON object with NOTHING else outside it:

{{
  "intent": "command | question | request_clarification | conversation | other",
  "internal_thought": "Brief reasoning about what to do and why, based on memory and vision.",
  "response": "A friendly, clear, and concise response to speak out loud to the user.",
  "actions": [],
  "summary": "A concise summary of important new information not yet stored in memory.",
  "save_to_memory": true
}}

====================================

Intent Guidelines:

- command: User requests physical or digital actions. actions must contain ordered atomic steps.
- question: User requests information. actions must be [].
- request_clarification: Input is ambiguous. Ask a clear concise question.
- conversation: Casual dialogue. actions must be [].

====================================

Rules:

1) Output valid JSON only. Nothing outside the JSON object.
2) If intent != command then actions = [].
3) If information is missing or ambiguous, set intent = request_clarification.
4) response must be friendly, warm, and conversational; keep it concise.
5) save_to_memory = true only if the information is new and important.
6) If you cannot confidently identify the user (confidence < 60%), politely ask for their name.
7) Use only actions from the Action Library.

====================================

Current Inputs:

- user_message: {user_message}
- vision_input: {vision_input}
"""


class PromptEngine:
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT

    def build_prompt(self, user_text: str, context: Dict) -> str:
        """
        Constructs the full prompt from context dict.
        Expected keys: identities, social_data, emotions, objects,
                       history, deep_memory, available_actions, time
        """
        identities = context.get('identities', [])
        social_data = context.get('social_data', [])
        emotions = context.get('emotions', [])
        objects = context.get('objects', [])  # List[{label, position}]
        history = context.get('history', [])
        deep_memory = context.get('deep_memory', [])
        available_actions = context.get('available_actions', '[]')

        # --- Build Vision Input JSON (mimicking reasoning-1.ipynb format) ---
        humans_json = []
        for i, pid in enumerate(identities):
            data = social_data[i] if i < len(social_data) else {}
            emo = emotions[i] if i < len(emotions) else "neutral"
            name = data.get('name', pid)
            entry = {
                "name": name,
                "emotion": emo,
                "action": "speaking"
            }
            if 'job' in data:
                entry['job'] = data['job']
            if 'interests' in data:
                entry['interests'] = data['interests']
            if 'summary' in data and data['summary']:
                # summary is a list of past conversation summaries - show most recent
                summaries = data['summary'] if isinstance(data['summary'], list) else [data['summary']]
                if summaries:
                    entry['past_context'] = summaries[-1]
            humans_json.append(entry)

        vision_input = json.dumps({
            "current_speaker": identities[0] if identities else "unknown",
            "humans": humans_json,
            "objects": objects
        }, ensure_ascii=False, indent=2)

        # --- Short-term conversation history ---
        current_conversation = "\n".join(history) if history else "None"

        # --- Long-term memory from deep_memory (RAG results) ---
        if deep_memory:
            memory_lines = []
            for mem in deep_memory:
                memory_lines.append(
                    f"- ({mem.get('time_str', '')}) {mem.get('role', '')}: {mem.get('text', '')}"
                )
            memory_input = "\n".join(memory_lines)
        else:
            memory_input = "None"

        filled_system = self.system_prompt.format(
            current_conversation=current_conversation,
            memory_input=memory_input,
            available_actions=available_actions,
            user_message=user_text,
            vision_input=vision_input
        )

        prompt = f"""<|system|>
{filled_system}

Current Date/Time: {context.get('time', 'Unknown')}
<|assistant|>
"""
        return prompt
