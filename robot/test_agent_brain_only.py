import yaml
import logging
import time
import json
from dialogue.llm_client import LLMClient
from dialogue.prompt_engine import PromptEngine
from dialogue.info_extractor import InfoExtractor
from memory.social_memory import SocialMemory
from memory.vector_memory import VectorMemory

# Setup logging
logging.basicConfig(level=logging.ERROR) # Only show errors to keep terminal clean
logger = logging.getLogger("BrainTest")

def main():
    print("--- Robot Brain-Only Test (Terminal Mode) ---")
    print("Type 'exit' to quit. This test bypasses Camera/Audio.\n")

    # Load config
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize Modules
    llm = LLMClient(config.get('llm', {}))
    prompt_engine = PromptEngine()
    info_extractor = InfoExtractor(llm)
    social_memory = SocialMemory(config['memory']['db_path'])
    vector_memory = VectorMemory(config['memory'].get('chroma_db_path', 'memory/chroma_db'))
    
    # Simulate a Person ID for the session
    person_id = input("Which Person ID should I simulate recognizing? (default: person_0): ") or "person_0"
    
    history = []
    last_thought = "None"

    try:
        while True:
            user_text = input("\nYou > ")
            if user_text.lower() in ["exit", "quit"]:
                break
            
            # --- 1. Gather Context ---
            soc_data = social_memory.get(person_id)
            deep_memory = vector_memory.search_past(person_id, user_text, n_results=3)
            
            context = {
                "identities": [person_id],
                "social_data": [soc_data],
                "emotions": ["neutral"],
                "objects": [],
                "history": history,
                "deep_memory": deep_memory,
                "last_thought": last_thought,
                "available_actions": [], # Placeholder
                "time": time.ctime()
            }
            
            # --- 2. Reason & Decide ---
            prompt = prompt_engine.build_prompt(user_text, context)
            
            print(f"\n[Thinking with {config['llm']['engine']}...]")
            response = llm.generate(prompt)
            
            # --- 3. Parse Response ---
            response_text = response
            thought = ""
            try:
                # Basic JSON cleanup
                json_str = response
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                data = json.loads(json_str)
                thought = data.get("internal_thought", "")
                response_text = data.get("response", "") or response
                
                if thought:
                    print(f"\n🧠 THOUGHT: {thought}")
                    last_thought = thought
                
            except json.JSONDecodeError:
                pass

            print(f"\nRobot > {response_text}")
            
            # --- 4. Update Memory ---
            history.append(f"user: {user_text}")
            history.append(f"agent: {response_text}")
            
            # Ingest to Vector
            vector_memory.add_interaction(person_id, user_text, role="user")
            vector_memory.add_interaction(person_id, response_text, role="agent")
            
            # Extract Facts
            facts = info_extractor.extract(user_text)
            if facts:
                social_memory.update(person_id, facts)
                print(f"💾 MEMORY: Extracted new facts: {facts}")

    except KeyboardInterrupt:
        pass
    print("\nTest Finished.")

if __name__ == "__main__":
    main()
