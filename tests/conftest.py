"""
Pytest configuration and fixtures.
"""
import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.config import Settings
from src.services.vector_service import VectorService
from src.services.document_service import DocumentService


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Test settings configuration."""
    return Settings(
        API_KEY="test-api-key",
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION_NAME="test_pdf_documents",
        OLLAMA_URL="http://localhost:11434",
        OLLAMA_MODEL="mistral",
        OPENAI_API_KEY=None,  # No OpenAI for tests
        ENVIRONMENT="test",
        DATA_DIR="./test_data",
        LOGS_DIR="./test_logs"
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
async def mock_vector_service():
    """Mock vector service for testing."""
    service = Mock(spec=VectorService)
    service.initialize = AsyncMock()
    service.add_documents = AsyncMock(return_value=True)
    service.search_similar = AsyncMock(return_value=[])
    service.delete_document = AsyncMock(return_value=True)
    service.get_document_count = AsyncMock(return_value=0)
    service.close = AsyncMock()
    return service


@pytest.fixture
def sample_pdf_content():
    """Sample PDF text content for testing."""
    return """
    # Test Research Paper

    ## Abstract
    This is a test research paper for evaluating the PDF RAG system.
    It contains various sections and technical content.

    ## Introduction
    Recent advances in retrieval-augmented generation (RAG) have shown
    significant improvements in question-answering systems.

    ## Methodology
    Our approach combines dense vector retrieval with sparse keyword matching
    to achieve better performance on academic papers.

    ## Results
    We achieved 85% accuracy on the test dataset with our hybrid approach.
    The system processed 1000 documents in under 10 seconds.

    ## Conclusion
    The proposed method shows promising results for academic document retrieval.
    """


@pytest.fixture
def sample_evaluation_pairs():
    """Sample evaluation pairs for testing."""
    return [
        {
            "question": "What is the accuracy of the proposed method?",
            "expected_answer": "The proposed method achieved 85% accuracy on the test dataset."
        },
        {
            "question": "How fast is the document processing?",
            "expected_answer": "The system processed 1000 documents in under 10 seconds."
        },
        {
            "question": "What approach does the methodology use?",
            "expected_answer": "The methodology combines dense vector retrieval with sparse keyword matching."
        }
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.ainvoke = AsyncMock()
    
    # Configure different responses for different prompts
    async def mock_response(messages):
        content = "This is a mock response from the LLM."
        response = Mock()
        response.content = content
        return response
    
    llm.ainvoke.side_effect = mock_response
    return llm


@pytest.fixture
def test_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_environment(test_settings, temp_dir):
    """Setup test environment."""
    # Create necessary directories
    (temp_dir / "data").mkdir(exist_ok=True)
    (temp_dir / "logs").mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DATA_DIR"] = str(temp_dir / "data")
    os.environ["LOGS_DIR"] = str(temp_dir / "logs")
    os.environ["API_KEY"] = "test-api-key"
    
    yield
    
    # Cleanup
    for key in ["ENVIRONMENT", "DATA_DIR", "LOGS_DIR", "API_KEY"]:
        os.environ.pop(key, None)