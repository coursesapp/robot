import logging
from typing import Dict, Any, Callable

logger = logging.getLogger("ActionLibrary")

class ActionLibrary:
    """
    Central registry for all physical or digital actions the robot can perform.
    The LLM outputs action intents, and this library executes the bound functions.
    """
    def __init__(self):
        self.registry: Dict[str, Callable] = {}
        
        # Register default actions
        self.register("turn_head", self._dummy_turn_head)
        self.register("play_sound", self._dummy_play_sound)
        self.register("system_status", self._dummy_system_status)

    def register(self, name: str, func: Callable):
        self.registry[name] = func

    def execute(self, action_name: str, parameters: Dict[str, Any] = None) -> bool:
        if not parameters:
            parameters = {}
            
        if action_name not in self.registry:
            logger.warning(f"Action '{action_name}' requested by LLM but not found in registry.")
            return False
            
        try:
            logger.info(f"Executing action: {action_name} with params: {parameters}")
            self.registry[action_name](**parameters)
            return True
        except Exception as e:
            logger.error(f"Error executing action '{action_name}': {e}")
            return False

    def get_available_actions_schema(self) -> str:
        """Returns a string describing available actions for the LLM prompt."""
        return """
Available Actions:
- turn_head(direction): Directions can be "left", "right", "up", "down", "center".
- play_sound(sound_type): Types can be "beep", "alarm", "happy_chime".
- system_status(): Prints current system diagnostic.
        """

    # --- Dummy Implementations (Replace with real ROS/Serial/Hardware calls later) ---
    def _dummy_turn_head(self, direction: str = "center"):
        logger.info(f"[HARDWARE SIMULATION] Turning head to: {direction}")
        
    def _dummy_play_sound(self, sound_type: str = "beep"):
        logger.info(f"[HARDWARE SIMULATION] Playing sound: {sound_type}")
        
    def _dummy_system_status(self, **kwargs):
        logger.info(f"[SYSTEM] All systems operational. Actions online.")
