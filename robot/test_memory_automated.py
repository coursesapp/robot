import yaml
import logging
import time
import json
import os
from dialogue.llm_client import LLMClient
from dialogue.prompt_engine import PromptEngine
from dialogue.info_extractor import InfoExtractor
from dialogue.summarizer import Summarizer
from memory.social_memory import SocialMemory
from memory.vector_memory import VectorMemory

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("AutoMemoryTest")

def main():
    print("=== Automated Memory & Dialogue Simulation ===\n")

    # Load config
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize Modules
    llm = LLMClient(config.get('llm', {}))
    prompt_engine = PromptEngine()
    info_extractor = InfoExtractor(llm)
    summarizer = Summarizer(llm)
    social_memory = SocialMemory(config['memory']['db_path'])
    vector_memory = VectorMemory(config['memory'].get('chroma_db_path', 'memory/chroma_db'))
    
    person_id = "test_user_automated"
    # Ensure fresh start for this test ID
    # Clear social memory for this test ID if exists
    # (In a real test we might drop tables, but here we just append)

    dialogue_script = [
        "Hi, my name is Moham and I am a software architect from Egypt.",
        "I love working on robotics and artificial intelligence.",
        "Today is a sunny day in Cairo.",
        "Do you remember what my job is?",
        "I also have a cat named 'Pixel'. She is very playful.",
        "Tell me a bit about Cairo, and mention what my favorite field is.",
        "I just remembered, I also enjoy playing Chess on weekends.",
        "Wait, what was the name of my cat again?",
        "I'm feeling very excited about our conversation today!"
    ]

    context_history = []
    last_thought = "None"

    print(f"Simulating {len(dialogue_script)} turns of conversation for ID: {person_id}...\n")

    for i, user_text in enumerate(dialogue_script):
        print(f"[{i+1}] User: {user_text}")
        
        # 1. Perception/Context Gathering (Simulated)
        soc_data = social_memory.get(person_id)
        deep_memory = vector_memory.search_past(person_id, user_text, n_results=3)
        
        context = {
            "identities": [person_id],
            "social_data": [soc_data],
            "emotions": ["neutral"],
            "objects": [],
            "history": list(context_history),
            "deep_memory": deep_memory,
            "last_thought": last_thought,
            "available_actions": [],
            "time": time.ctime()
        }
        
        # 2. Decision/Brain
        prompt = prompt_engine.build_prompt(user_text, context)
        response = llm.generate(prompt)
        
        # Parse response
        response_text = response
        try:
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            data = json.loads(json_str)
            response_text = data.get("response", "") or response
            last_thought = data.get("internal_thought", "None")
        except:
            pass

        print(f"    Robot: {response_text}")

        # 3. Memory Updates
        context_history.append(f"user: {user_text}")
        context_history.append(f"agent: {response_text}")
        
        # Vector Memory (Long-term)
        vector_memory.add_interaction(person_id, user_text, role="user")
        vector_memory.add_interaction(person_id, response_text, role="agent")
        
        # Social Memory (Facts) - Logic from agent_loop
        facts = info_extractor.extract(user_text)
        if facts:
            social_memory.update(person_id, facts)
            print(f"    [!] Fact Extracted: {facts}")

        # Context Compression (Summarization) - Trigger at every 4 turns for testing
        if len(context_history) >= 8: # 4 exchanges
            print("    [!] Compressing History...")
            summary = summarizer.summarize(list(context_history))
            if summary:
                context_history = [f"[Conversation Summary]: {summary}"]
                social_memory.update(person_id, {"summary": summary})
                vector_memory.add_interaction(person_id, f"[Summary]: {summary}", role="system")
                print(f"    [!] New Summary: {summary[:100]}...")

    print("\n=== Verification Report ===")
    
    # 1. Check Social Memory
    final_soc = social_memory.get(person_id)
    print(f"\nSocial Memory (Facts) for {person_id}:")
    print(json.dumps(final_soc, indent=2))
    
    # 2. Check Vector Memory
    print(f"\nVector Memory (Long-term search) for 'cat':")
    vector_results = vector_memory.search_past(person_id, "What is my cat's name?", n_results=3)
    print(vector_results)
    
    # 3. Check Final Summary
    print(f"\nFinal Compressed History State:")
    print(context_history)

if __name__ == "__main__":
    main()
