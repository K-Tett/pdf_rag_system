"""
Conversation memory service for managing session-based chat history.
"""
import structlog
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import threading

from langchain.schema import BaseMessage, HumanMessage, AIMessage

logger = structlog.get_logger()


class ConversationMemory:
    """
    In-memory conversation storage with session management and cleanup.
    """
    
    def __init__(self, session_timeout: int = 3600, max_messages_per_session: int = 50):
        """
        Initialize conversation memory.
        
        Args:
            session_timeout: Session timeout in seconds (default: 1 hour)
            max_messages_per_session: Maximum messages to keep per session
        """
        self.session_timeout = session_timeout
        self.max_messages_per_session = max_messages_per_session
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._conversations: Dict[str, List[BaseMessage]] = defaultdict(list)
        self._session_timestamps: Dict[str, datetime] = {}
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the cleanup task for expired sessions."""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_expired_sessions()
                except Exception as e:
                    logger.error("Error in cleanup task", error=str(e))
                
                # Sleep for 5 minutes between cleanups
                import time
                time.sleep(300)
        
        self._cleanup_task = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_task.start()
        logger.info("Cleanup task started")
    
    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            session_id: Session identifier
            message: Message to add
        """
        with self._lock:
            self._conversations[session_id].append(message)
            self._session_timestamps[session_id] = datetime.utcnow()
            
            # Trim conversation if it exceeds max length
            if len(self._conversations[session_id]) > self.max_messages_per_session:
                # Keep the most recent messages
                self._conversations[session_id] = self._conversations[session_id][-self.max_messages_per_session:]
            
            logger.debug(
                "Message added to conversation",
                session_id=session_id,
                message_type=type(message).__name__,
                total_messages=len(self._conversations[session_id])
            )
    
    def get_conversation_history(self, session_id: str, max_messages: Optional[int] = None) -> List[BaseMessage]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            max_messages: Maximum number of recent messages to return
            
        Returns:
            List of messages in chronological order
        """
        with self._lock:
            if session_id not in self._conversations:
                return []
            
            messages = self._conversations[session_id]
            
            if max_messages is not None and len(messages) > max_messages:
                messages = messages[-max_messages:]
            
            # Update session timestamp
            self._session_timestamps[session_id] = datetime.utcnow()
            
            return messages.copy()
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session existed and was cleared, False otherwise
        """
        with self._lock:
            if session_id in self._conversations:
                del self._conversations[session_id]
                if session_id in self._session_timestamps:
                    del self._session_timestamps[session_id]
                
                logger.info("Session cleared", session_id=session_id)
                return True
            
            return False
    
    def get_session_stats(self, session_id: str) -> Dict[str, any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
        """
        with self._lock:
            if session_id not in self._conversations:
                return {
                    "exists": False,
                    "message_count": 0,
                    "last_activity": None,
                    "is_active": False
                }
            
            messages = self._conversations[session_id]
            last_activity = self._session_timestamps.get(session_id)
            
            # Count message types
            human_messages = sum(1 for msg in messages if isinstance(msg, HumanMessage))
            ai_messages = sum(1 for msg in messages if isinstance(msg, AIMessage))
            
            # Check if session is still active
            is_active = self.is_session_active(session_id)
            
            return {
                "exists": True,
                "message_count": len(messages),
                "human_messages": human_messages,
                "ai_messages": ai_messages,
                "last_activity": last_activity,
                "is_active": is_active,
                "session_age_seconds": (datetime.utcnow() - last_activity).total_seconds() if last_activity else 0
            }
    
    def is_session_active(self, session_id: str) -> bool:
        """
        Check if a session is still active (not expired).
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session is active, False otherwise
        """
        with self._lock:
            if session_id not in self._session_timestamps:
                return False
            
            last_activity = self._session_timestamps[session_id]
            expiry_time = last_activity + timedelta(seconds=self.session_timeout)
            
            return datetime.utcnow() < expiry_time
    
    def get_last_activity(self, session_id: str) -> Optional[datetime]:
        """
        Get the last activity timestamp for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Last activity timestamp or None if session doesn't exist
        """
        with self._lock:
            return self._session_timestamps.get(session_id)
    
    def list_active_sessions(self) -> List[str]:
        """
        Get list of all active session IDs.
        
        Returns:
            List of active session IDs
        """
        with self._lock:
            active_sessions = []
            
            for session_id in self._conversations.keys():
                if self.is_session_active(session_id):
                    active_sessions.append(session_id)
            
            return active_sessions
    
    def get_memory_stats(self) -> Dict[str, any]:
        """
        Get overall memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            total_sessions = len(self._conversations)
            active_sessions = len(self.list_active_sessions())
            total_messages = sum(len(messages) for messages in self._conversations.values())
            
            # Calculate average messages per session
            avg_messages = total_messages / max(total_sessions, 1)
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "expired_sessions": total_sessions - active_sessions,
                "total_messages": total_messages,
                "average_messages_per_session": round(avg_messages, 2),
                "session_timeout_seconds": self.session_timeout,
                "max_messages_per_session": self.max_messages_per_session
            }
    
    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        with self._lock:
            current_time = datetime.utcnow()
            expired_sessions = []
            
            for session_id, last_activity in self._session_timestamps.items():
                if current_time - last_activity > timedelta(seconds=self.session_timeout):
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                if session_id in self._conversations:
                    del self._conversations[session_id]
                if session_id in self._session_timestamps:
                    del self._session_timestamps[session_id]
            
            if expired_sessions:
                logger.info(
                    "Cleaned up expired sessions",
                    expired_count=len(expired_sessions),
                    remaining_sessions=len(self._conversations)
                )
    
    def add_conversation_exchange(
        self,
        session_id: str,
        human_message: str,
        ai_message: str
    ) -> None:
        """
        Add a complete conversation exchange (human + AI message).
        
        Args:
            session_id: Session identifier
            human_message: Human message content
            ai_message: AI message content
        """
        self.add_message(session_id, HumanMessage(content=human_message))
        self.add_message(session_id, AIMessage(content=ai_message))
    
    def get_recent_context(
        self,
        session_id: str,
        max_exchanges: int = 3
    ) -> str:
        """
        Get recent conversation context as a formatted string.
        
        Args:
            session_id: Session identifier
            max_exchanges: Maximum number of exchanges to include
            
        Returns:
            Formatted conversation context
        """
        messages = self.get_conversation_history(session_id)
        
        if not messages:
            return ""
        
        # Get the most recent exchanges
        recent_messages = messages[-(max_exchanges * 2):] if len(messages) > max_exchanges * 2 else messages
        
        context_parts = []
        for i in range(0, len(recent_messages), 2):
            if i + 1 < len(recent_messages):
                human_msg = recent_messages[i]
                ai_msg = recent_messages[i + 1]
                
                if isinstance(human_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                    context_parts.append(f"Human: {human_msg.content}")
                    context_parts.append(f"Assistant: {ai_msg.content}")
        
        return "\n\n".join(context_parts)
    
    def shutdown(self) -> None:
        """Shutdown the memory service and cleanup resources."""
        logger.info("Shutting down conversation memory service")
        
        # The cleanup task will stop automatically when the main process exits
        # since it's a daemon thread
        
        with self._lock:
            self._conversations.clear()
            self._session_timestamps.clear()
        
        logger.info("Conversation memory service shutdown complete")