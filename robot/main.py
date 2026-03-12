import argparse
import signal
import sys
import yaml
import logging
from pathlib import Path
from core.agent_loop import AgentLoop

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Main")

def load_config(config_path: str):
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        sys.exit(1)
    
    with open(path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config: {e}")
            sys.exit(1)

def signal_handler(sig, frame):
    logger.info("🛑 Received termination signal. Shutting down...")
    # The agent loop should handle graceful exit via a simplistic flag or event
    # We rely on the loops main `stop()` method being called below or in finally block
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Local-First Social AI Agent")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to configuration file")
    parser.add_argument("--headless", action="store_true", help="Run without GUI windows (for servers)")
    parser.add_argument("--duration", type=int, default=0, help="Run for N seconds then exit (0=infinite)")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Override headless config if flag set
    if args.headless:
        config['headless'] = True
    
    # Initialize Core Agent Loop
    agent = AgentLoop(config)
    
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, lambda s, f: agent.stop())
    signal.signal(signal.SIGTERM, lambda s, f: agent.stop())

    try:
        logger.info("🚀 Starting Agent Loop...")
        agent.run(duration=args.duration)
    except Exception as e:
        logger.critical(f"🔥 Critical error in main loop: {e}", exc_info=True)
    finally:
        agent.stop()
        logger.info("👋 Agent shutdown complete.")

if __name__ == "__main__":
    main()
