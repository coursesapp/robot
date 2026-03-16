import pyttsx3

def speak_fresh(text):
    """
    Initializes a NEW engine instance for every sentence.
    This prevents the 'zombie' bug where the engine gets stuck.
    """
    engine = pyttsx3.init()
    
    # You have to set properties every time since it's a new engine
    engine.setProperty('rate', 150) 
    engine.setProperty('volume', 1.0)
    
    # Optional: Set voice (0 for Male, 1 for Female)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id) 

    engine.say(text)
    engine.runAndWait()
    # The engine dies here when the function ends, clearing the memory.

def main():
    print("--- Python TTS (Reliable Mode) ---")
    print("Type 'exit' to quit.")

    while True:
        user_text = input(">> ")
        
        if user_text.lower() == 'exit':
            print("Goodbye!")
            break
        
        speak_fresh(user_text)

if __name__ == "__main__":
    main()