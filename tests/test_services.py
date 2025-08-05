"""
Service layer tests.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import tempfile
from pathlib import Path

from src.services.document_service import DocumentChunker, PDFProcessor, DocumentService
from src.services.memory_service import ConversationMemory
from src.core.config import Settings
from langchain.schema import HumanMessage, AIMessage


class TestDocumentChunker:
    """Test document chunking functionality."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        text = "This is a test sentence. This is another sentence. And this is a third sentence that should create multiple chunks."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = DocumentChunker()
        
        chunks = chunker.chunk_text("")
        assert chunks == []
    
    def test_chunk_text_with_metadata(self):
        """Test chunking with metadata."""
        chunker = DocumentChunker()
        
        text = "This is a test sentence."
        metadata = {"document_id": "test123", "title": "Test Document"}
        
        chunks = chunker.chunk_text(text, metadata)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk['metadata']['document_id'] == "test123"
            assert chunk['metadata']['title'] == "Test Document"
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        chunker = DocumentChunker()
        
        dirty_text = "This  has   excessive    whitespace.\n\nAnd\tsome\ttabs."
        cleaned = chunker._clean_text(dirty_text)
        
        assert "   " not in cleaned
        assert "\t" not in cleaned
        assert cleaned.strip() == cleaned


class TestConversationMemory:
    """Test conversation memory functionality."""
    
    def test_add_message(self):
        """Test adding messages to memory."""
        memory = ConversationMemory(session_timeout=3600)
        
        session_id = "test_session"
        message = HumanMessage(content="Hello")
        
        memory.add_message(session_id, message)
        
        history = memory.get_conversation_history(session_id)
        assert len(history) == 1
        assert history[0].content == "Hello"
    
    def test_clear_session(self):
        """Test clearing session memory."""
        memory = ConversationMemory(session_timeout=3600)
        
        session_id = "test_session"
        memory.add_message(session_id, HumanMessage(content="Hello"))
        
        assert len(memory.get_conversation_history(session_id)) == 1
        
        memory.clear_session(session_id)
        
        assert len(memory.get_conversation_history(session_id)) == 0
    
    def test_max_messages_per_session(self):
        """Test maximum messages per session limit."""
        memory = ConversationMemory(session_timeout=3600, max_messages_per_session=3)
        
        session_id = "test_session"
        
        # Add more messages than the limit
        for i in range(5):
            memory.add_message(session_id, HumanMessage(content=f"Message {i}"))
        
        history = memory.get_conversation_history(session_id)
        assert len(history) == 3  # Should be limited to max_messages_per_session
        
        # Should keep the most recent messages
        assert history[-1].content == "Message 4"
    
    def test_session_stats(self):
        """Test session statistics."""
        memory = ConversationMemory(session_timeout=3600)
        
        session_id = "test_session"
        memory.add_message(session_id, HumanMessage(content="Hello"))
        memory.add_message(session_id, AIMessage(content="Hi there"))
        
        stats = memory.get_session_stats(session_id)
        
        assert stats["exists"] is True
        assert stats["message_count"] == 2
        assert stats["human_messages"] == 1
        assert stats["ai_messages"] == 1
        assert stats["is_active"] is True
    
    def test_nonexistent_session_stats(self):
        """Test stats for non-existent session."""
        memory = ConversationMemory(session_timeout=3600)
        
        stats = memory.get_session_stats("nonexistent")
        
        assert stats["exists"] is False
        assert stats["message_count"] == 0
        assert stats["is_active"] is False
    
    def test_conversation_exchange(self):
        """Test adding complete conversation exchange."""
        memory = ConversationMemory(session_timeout=3600)
        
        session_id = "test_session"
        memory.add_conversation_exchange(
            session_id,
            "What is AI?",
            "AI stands for Artificial Intelligence."
        )
        
        history = memory.get_conversation_history(session_id)
        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)
        assert history[0].content == "What is AI?"
        assert history[1].content == "AI stands for Artificial Intelligence."


class TestPDFProcessor:
    """Test PDF processing functionality."""
    
    @pytest.fixture
    def pdf_processor(self):
        """Create PDF processor instance."""
        return PDFProcessor()
    
    def test_extract_document_info(self, pdf_processor):
        """Test document info extraction from text."""
        text = """
        Advanced RAG Techniques for Document Retrieval
        
        Author: John Doe
        University of Technology
        
        Abstract
        This paper presents novel approaches to retrieval-augmented generation...
        """
        
        info = pdf_processor._extract_document_info(text, "test.pdf")
        
        assert "title" in info
        assert "Advanced RAG Techniques for Document Retrieval" in info["title"]
        assert "author" in info
        assert "John Doe" in info["author"]


class TestDocumentService:
    """Test document service functionality."""
    
    @pytest.fixture
    async def mock_vector_service(self):
        """Create mock vector service."""
        service = Mock()
        service.add_documents = AsyncMock(return_value=True)
        service.delete_document = AsyncMock(return_value=True)
        return service
    
    @pytest.fixture
    def document_service(self, mock_vector_service, tmp_path):
        """Create document service with mocked dependencies."""
        return DocumentService(mock_vector_service, str(tmp_path))
    
    def test_generate_document_id(self, document_service, tmp_path):
        """Test document ID generation."""
        # Create a test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"test content")
        
        doc_id = document_service._generate_document_id(str(test_file))
        
        assert doc_id.startswith("doc_")
        assert len(doc_id) > 10  # Should have some hash content
    
    def test_load_metadata(self, document_service):
        """Test metadata loading."""
        # Initially should be empty
        assert document_service.documents_metadata == {}
    
    def test_save_and_load_metadata(self, document_service, tmp_path):
        """Test saving and loading metadata."""
        # Add some metadata
        test_metadata = {
            "doc123": {
                "id": "doc123",
                "filename": "test.pdf",
                "title": "Test Document"
            }
        }
        
        document_service.documents_metadata = test_metadata
        document_service._save_metadata()
        
        # Create new service instance to test loading
        new_service = DocumentService(Mock(), str(tmp_path))
        assert new_service.documents_metadata == test_metadata
    
    def test_list_documents_empty(self, document_service):
        """Test listing documents when none exist."""
        documents = document_service.list_documents()
        assert documents == []
    
    async def test_delete_document(self, document_service, mock_vector_service):
        """Test document deletion."""
        # Add a document to metadata first
        doc_id = "test_doc_123"
        document_service.documents_metadata[doc_id] = {
            "id": doc_id,
            "filename": "test.pdf",
            "status": "completed"
        }
        
        # Mock successful vector service deletion
        mock_vector_service.delete_document.return_value = True
        
        result = await document_service.delete_document(doc_id)
        
        assert result is True
        assert doc_id not in document_service.documents_metadata
        mock_vector_service.delete_document.assert_called_once_with(doc_id)


class TestSettings:
    """Test configuration settings."""
    
    def test_default_settings(self):
        """Test default settings creation."""
        settings = Settings()
        
        assert settings.API_KEY == "pdf-rag-secret-key"
        assert settings.QDRANT_URL == "http://localhost:6333"
        assert settings.ENVIRONMENT == "development"
    
    def test_environment_override(self):
        """Test environment variable override."""
        import os
        
        # Set environment variable
        os.environ["API_KEY"] = "custom-key"
        os.environ["ENVIRONMENT"] = "test"
        
        try:
            settings = Settings()
            assert settings.API_KEY == "custom-key"
            assert settings.ENVIRONMENT == "test"
        finally:
            # Clean up
            os.environ.pop("API_KEY", None)
            os.environ.pop("ENVIRONMENT", None)
    
    def test_is_production_property(self):
        """Test is_production property."""
        settings = Settings(ENVIRONMENT="production")
        assert settings.is_production is True
        
        settings = Settings(ENVIRONMENT="development")
        assert settings.is_production is False
    
    def test_has_openai_key_property(self):
        """Test has_openai_key property."""
        settings = Settings(OPENAI_API_KEY="sk-test123")
        assert settings.has_openai_key is True
        
        settings = Settings(OPENAI_API_KEY=None)
        assert settings.has_openai_key is False
        
        settings = Settings(OPENAI_API_KEY="")
        assert settings.has_openai_key is False