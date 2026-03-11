import time
from colorama import Fore, Style, init

# Import our Modules
from stt_engine import HearingEngine
from TTS.tts_engine import VoiceEngine
from llm_engine import BrainEngine     # <--- Your new Mistral file

# In main_robot.py
#from llm_engine import FakeBrainEngine 
#bot_brain = FakeBrainEngine()

init(autoreset=True)

def main():
    print(f"\n{Fore.MAGENTA}=== 🤖 STARTING MISTRAL ROBOT ==={Style.RESET_ALL}")


    try:
        # A. Start Mouth
        bot_mouth = VoiceEngine()
        bot_mouth.say("System initialized. Mistral Brain is loading.")
        
        # --- B. USE THE FAKE BRAIN FOR NOW ---
        # Change this line back to 'BrainEngine()' when the download finishes!
        bot_brain = BrainEngine()
        
        # C. Start Ear
        bot_ear = HearingEngine(model_size="base")
        
        bot_mouth.say("I am ready to test streaming.")
        
    except Exception as e:
        print(f"{Fore.RED}Startup Error: {e}{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}Ready! Speak to your robot.{Style.RESET_ALL}")
    
    try:
        # Listen Loop
        for user_text in bot_ear.listen():
            
            print(f"{Fore.GREEN}👤 USER: {user_text}{Style.RESET_ALL}")
            
            # 1. Pause Ear
            bot_ear.pause()

            # 2. Check Exit
            if "exit" in user_text.lower():
                bot_mouth.say("Goodbye.")
                break
            
            # 3. Stream from Mistral -> Kokoro
            print(f"{Fore.YELLOW}⚡ Thinking...{Style.RESET_ALL}")
            
            # Get the generator from the Brain
            token_stream = bot_brain.think_stream(user_text)
            
            # Feed it directly to the Mouth
            bot_mouth.speak_stream(token_stream)
            
            # 4. Resume Ear
            print(f"{Fore.CYAN}👂 Listening...{Style.RESET_ALL}")
            bot_ear.resume()

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Critical Error: {e}")
        input("Press Enter...")

if __name__ == "__main__":
    main()