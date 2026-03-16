import abc
import numpy as np
from typing import Dict, Any, List, Optional

class InteractionStrategy(abc.ABC):
    """
    Interface for interaction flow. 
    Allows switching between local sequential logic and Gemini Live streaming.
    """
    
    @abc.abstractmethod
    def on_speech_start(self):
        """Called the instant user speech is detected (for local mute)."""
        pass

    @abc.abstractmethod
    def on_speech(self, text: str, audio_np: np.ndarray, context_metadata: Dict[str, Any]):
        """Called when user speech is detected."""
        pass

    @abc.abstractmethod
    def on_vision(self, frame: np.ndarray, metadata: Dict[str, Any]):
        """Called to update visual context."""
        pass

    @abc.abstractmethod
    def stop(self):
        """Cleanup resources."""
        pass
