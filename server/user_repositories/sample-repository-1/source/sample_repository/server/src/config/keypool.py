import threading
from src.config.settings import GROQ_KEY_POOL

class KeyPool:
    """
    Thread-safe round-robin key dispatcher.
    Each call to next() returns the next key in rotation.
    This spreads load evenly across all available keys,
    keeping each key well under Groq's per-minute token limit.
    """
    def __init__(self, keys: list[str]):
        self._keys  = keys
        self._index = 0
        self._lock  = threading.Lock()

    def next(self) -> str:
        with self._lock:
            key = self._keys[self._index % len(self._keys)]
            self._index += 1
            return key

    def __len__(self):
        return len(self._keys)


# Single shared instance — import this everywhere
pool = KeyPool(GROQ_KEY_POOL)