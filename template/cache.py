"""
Redis-based caching for query results.
Fallback to in-memory if Redis unavailable.
"""

import redis
import json
import hashlib
from typing import Optional, Dict, Any
from template.config import settings
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Dual-mode cache: Redis (production) or in-memory (development).
    """
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}  # Fallback
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection with fallback"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                decode_responses=True,
                socket_connect_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connected")
        
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory cache: {e}")
            self.redis_client = None
    
    def get_cached(self, query: str, conversation_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for a query.
        """
        cache_key = self._generate_key(query, conversation_id)
        
        try:
            if self.redis_client:
                cached = self.redis_client.get(cache_key)
                if cached:
                    logger.info(f"Cache HIT: {cache_key[:20]}...")
                    return json.loads(cached)
            else:
                # In-memory fallback
                return self.memory_cache.get(cache_key)
        
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def set_cached(
        self, 
        query: str, 
        response: Dict[str, Any],
        conversation_id: str = None,
        ttl: int = None
    ) -> bool:
        """
        Cache a query response.
        """
        cache_key = self._generate_key(query, conversation_id)
        ttl = ttl or settings.CACHE_TTL
        
        try:
            if self.redis_client:
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(response)
                )
                logger.info(f"Cached: {cache_key[:20]}... (TTL: {ttl}s)")
                return True
            else:
                # In-memory fallback (no TTL)
                self.memory_cache[cache_key] = response
                return True
        
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def invalidate(self, pattern: str = None):
        """
        Invalidate cache entries matching pattern.
        """
        try:
            if self.redis_client:
                if pattern:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                        logger.info(f"Invalidated {len(keys)} cache entries")
                else:
                    self.redis_client.flushdb()
                    logger.info("Flushed entire cache")
            else:
                if pattern:
                    self.memory_cache = {
                        k: v for k, v in self.memory_cache.items()
                        if pattern not in k
                    }
                else:
                    self.memory_cache.clear()
        
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
    
    @staticmethod
    def _generate_key(query: str, conversation_id: str = None) -> str:
        """Generate consistent cache key"""
        base = f"{conversation_id or 'global'}:{query}"
        return f"rag_cache:{hashlib.md5(base.encode()).hexdigest()}"


# Singleton instance
cache = CacheManager()


# Convenience functions
def get_cached(query: str, conversation_id: str = None) -> Optional[Dict[str, Any]]:
    return cache.get_cached(query, conversation_id)


def set_cached(
    query: str, 
    response: Dict[str, Any], 
    conversation_id: str = None,
    ttl: int = None
) -> bool:
    return cache.set_cached(query, response, conversation_id, ttl)