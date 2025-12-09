"""
Simple in-memory cache for coherence scores and feed results
"""

import time
from typing import Dict, Any, Optional
from threading import Lock


class SimpleCache:
    """Thread-safe in-memory cache with TTL"""

    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.lock = Lock()
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    return value
                else:
                    # Clean up expired entry
                    del self.cache[key]
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with TTL (seconds)"""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        with self.lock:
            self.cache[key] = (value, expiry)

    def delete(self, key: str):
        """Delete entry from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()

    def cleanup_expired(self):
        """Remove all expired entries"""
        now = time.time()
        with self.lock:
            expired_keys = [k for k, (_, expiry) in self.cache.items() if now >= expiry]
            for key in expired_keys:
                del self.cache[key]


# Global cache instances
coherence_cache = SimpleCache(default_ttl=300)  # 5 minutes for coherence scores
feed_cache = SimpleCache(default_ttl=60)  # 1 minute for feed results
