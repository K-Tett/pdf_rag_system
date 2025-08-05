"""
Configuration settings for the PDF RAG system.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    API_KEY: str = Field(default="pdf-rag-secret-key", env="API_KEY")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # Database Configuration
    QDRANT_URL: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    QDRANT_COLLECTION_NAME: str = Field(default="pdf_documents", env="QDRANT_COLLECTION_NAME")
    
    # LLM Configuration
    OLLAMA_URL: str = Field(default="http://localhost:11434", env="OLLAMA_URL")
    OLLAMA_MODEL: str = Field(default="mistral", env="OLLAMA_MODEL")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4-1106-preview", env="OPENAI_MODEL")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5", env="EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # Reranking Configuration
    RERANK_MODEL: str = Field(default="BAAI/bge-reranker-base", env="RERANK_MODEL")
    RERANK_TOP_K: int = Field(default=3, env="RERANK_TOP_K")
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = Field(default=10, env="RETRIEVAL_TOP_K")
    CHUNK_SIZE: int = Field(default=512, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=50, env="CHUNK_OVERLAP")
    
    # Search Configuration
    WEB_SEARCH_ENABLED: bool = Field(default=True, env="WEB_SEARCH_ENABLED")
    WEB_SEARCH_TOP_K: int = Field(default=5, env="WEB_SEARCH_TOP_K")
    
    # File Storage
    DATA_DIR: str = Field(default="./data", env="DATA_DIR")
    LOGS_DIR: str = Field(default="./logs", env="LOGS_DIR")
    
    # Session Configuration
    SESSION_TIMEOUT: int = Field(default=3600, env="SESSION_TIMEOUT")  # 1 hour
    MAX_CONVERSATION_TURNS: int = Field(default=20, env="MAX_CONVERSATION_TURNS")
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return self.OPENAI_API_KEY is not None and len(self.OPENAI_API_KEY.strip()) > 0