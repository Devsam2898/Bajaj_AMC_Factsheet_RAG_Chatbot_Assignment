"""
Conversation memory management for multi-turn dialogues.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from collections import deque
import json
from template.cache import cache
from template.config import settings
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    query: str
    response: str
    timestamp: float
    retrieved_docs: List[str] = None


class ConversationMemory:
    """
    Manages conversation history with sliding window.
    """
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.history = deque(maxlen=settings.MAX_HISTORY_TURNS)
        self._load_from_cache()
    
    def add_turn(
        self, 
        query: str, 
        response: str,
        retrieved_docs: List[str] = None
    ):
        """Add new turn to history"""
        import time
        
        turn = ConversationTurn(
            query=query,
            response=response,
            timestamp=time.time(),
            retrieved_docs=retrieved_docs or []
        )
        
        self.history.append(turn)
        self._save_to_cache()
        
        logger.info(f"Turn added (total: {len(self.history)})")
    
    def get_context_string(self, max_turns: int = None) -> str:
        """
        Build context string from recent history.
        Format: User: ... \nAssistant: ... \n
        """
        max_turns = max_turns or settings.MAX_HISTORY_TURNS
        recent = list(self.history)[-max_turns:]
        
        context = []
        for turn in recent:
            context.append(f"User: {turn.query}")
            context.append(f"Assistant: {turn.response}")
        
        return "\n".join(context)
    
    def get_last_query(self) -> Optional[str]:
        """Get the most recent query"""
        if self.history:
            return self.history[-1].query
        return None
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()
        self._save_to_cache()
        logger.info(f"Cleared conversation: {self.conversation_id}")
    
    def _load_from_cache(self):
        """Load history from cache"""
        cache_key = f"conversation:{self.conversation_id}"
        
        try:
            cached = cache.redis_client.get(cache_key) if cache.redis_client else None
            
            if cached:
                data = json.loads(cached)
                for turn_dict in data:
                    turn = ConversationTurn(**turn_dict)
                    self.history.append(turn)
                
                logger.info(f"Loaded {len(self.history)} turns from cache")
        
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")
   
    def _save_to_cache(self):
        """Save history to cache"""
        cache_key = f"conversation:{self.conversation_id}"
        
        try:
            data = [asdict(turn) for turn in self.history]
            
            if cache.redis_client:
                cache.redis_client.setex(
                    cache_key,
                    3600 * 24,  # 24 hour TTL
                    json.dumps(data)
                )
        
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")


class ConversationManager:
    """Managing multiple conversations"""
    
    def __init__(self):
        self.conversations: Dict[str, ConversationMemory] = {}
    
    def get_conversation(self, conversation_id: str) -> ConversationMemory:
        """Get or create conversation"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationMemory(conversation_id)
        
        return self.conversations[conversation_id]
    
    def delete_conversation(self, conversation_id: str):
        """Delete conversation"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].clear()
            del self.conversations[conversation_id]


# Singleton instance
conversation_manager = ConversationManager()