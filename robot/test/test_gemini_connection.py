import asyncio
import os
import logging
import yaml
from core.gemini_engine import GeminiLiveEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestGemini")

async def test_connection():
    # Load config to get API key
    try:
        with open("config/settings.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    g_cfg = cfg.get('interaction', {}).get('gemini', {})
    api_key = g_cfg.get('api_key')
    
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        logger.error("Please set a valid Gemini API Key in config/settings.yaml first!")
        return

    engine = GeminiLiveEngine(
        api_key=api_key,
        model_id=g_cfg.get('model', 'gemini-2.0-flash'),
        voice_name=g_cfg.get('voice_name', 'Aoede')
    )

    def on_text(text):
        print(f"\n[Gemini Response]: {text}")

    engine.set_callbacks(on_audio=None, on_text=on_text)

    # Start connection in a background task
    connection_task = asyncio.create_task(engine.connect("You are a helpful robot testing its connection. Say 'Connection Successful' if you can hear me."))
    
    await asyncio.sleep(5) # Give it time to connect
    
    if engine.running:
        logger.info("Connection established! Sending test text...")
        await engine.send_text("Hello Gemini, can you confirm the connection?")
        await asyncio.sleep(5) # Wait for response
    else:
        logger.error("Failed to establish connection within timeout.")

    engine.stop()
    connection_task.cancel()

if __name__ == "__main__":
    asyncio.run(test_connection())
