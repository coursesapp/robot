import queue
import logging
from typing import Any, Dict, List
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("EventBus")

@dataclass
class Event:
    topic: str
    data: Any
    timestamp: float = 0.0

    def __post_init__(self):
        import time
        if self.timestamp == 0.0:
            self.timestamp = time.time()

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[queue.Queue]] = defaultdict(list)
        self._lock = None  # Queue is thread-safe, but list operations might need care if dynamic
        # However, for this scale, simple list append/remove is atomic enough in CPython or we use a lock if needed.
        # We'll use a simple approach: copy list on publish to avoid holding locks during dispatch.

    def subscribe(self, topic: str) -> queue.Queue:
        """
        Subscribe to a topic. Returns a Queue that will receive events.
        """
        q = queue.Queue()
        self._subscribers[topic].append(q)
        logger.debug(f"New subscriber for topic: {topic}")
        return q

    def publish(self, topic: str, data: Any):
        """
        Publish an event to a topic. Distributes to all subscribers.
        """
        event = Event(topic, data)
        # Subscribers might be listening to wildcard '*' (optional feature, keeping simple for now)
        
        # Exact match
        if topic in self._subscribers:
            for q in self._subscribers[topic]:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    logger.warning(f"Subscriber queue full for topic {topic}, dropping event.")
        
        # Log heavy events? Maybe debug only
        # logger.debug(f"Published to {topic}: {str(data)[:50]}")

    def unsubscribe(self, topic: str, q: queue.Queue):
        if topic in self._subscribers:
            if q in self._subscribers[topic]:
                self._subscribers[topic].remove(q)
