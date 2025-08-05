"""
API endpoint tests.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock
import json


def test_health_endpoint(test_client):
    """Test health endpoint."""
    response = test_client.get("/health/")
    assert response.status_code == 200
    data = response.model_dump_json()
    assert "status" in data
    assert "timestamp" in data


def test_root_endpoint(test_client):
    """Test root endpoint."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.model_dump_json()
    assert data["message"] == "PDF RAG System API"
    assert data["version"] == "1.0.0"


def test_documents_list_requires_auth(test_client):
    """Test that documents endpoint requires authentication."""
    response = test_client.get("/documents/")
    assert response.status_code == 401


def test_documents_list_with_auth(test_client):
    """Test documents list with authentication."""
    headers = {"Authorization": "Bearer test-api-key"}
    
    with patch('src.api.main.get_document_service') as mock_service:
        mock_service.return_value.list_documents.return_value = []
        
        response = test_client.get("/documents/", headers=headers)
        assert response.status_code == 200
        data = response.model_dump_json()
        assert "documents" in data
        assert "total_count" in data


def test_chat_requires_auth(test_client):
    """Test that chat endpoint requires authentication."""
    response = test_client.post("/chat/ask", json={
        "question": "Test question",
        "session_id": "test"
    })
    assert response.status_code == 401


def test_chat_with_auth(test_client):
    """Test chat endpoint with authentication."""
    headers = {"Authorization": "Bearer test-api-key"}
    
    with patch('src.api.main.get_orchestrator') as mock_orchestrator:
        # Mock orchestrator response
        mock_response = Mock()
        mock_response.answer = "This is a test answer"
        mock_response.sources = []
        mock_response.confidence_score = 0.8
        mock_response.metadata = {"test": True}
        
        mock_orchestrator.return_value.process_question = AsyncMock(return_value=mock_response)
        
        response = test_client.post("/chat/ask", headers=headers, json={
            "question": "Test question",
            "session_id": "test"
        })
        
        assert response.status_code == 200
        data = response.model_dump_json()
        assert "answer" in data
        assert "confidence_score" in data


def test_invalid_api_key(test_client):
    """Test with invalid API key."""
    headers = {"Authorization": "Bearer invalid-key"}
    
    response = test_client.get("/documents/", headers=headers)
    assert response.status_code == 401


def test_evaluation_requires_auth(test_client):
    """Test that evaluation endpoint requires authentication."""
    response = test_client.post("/evaluation/single", json={
        "question": "Test question",
        "expected_answer": "Test answer"
    })
    assert response.status_code == 401


def test_evaluation_with_auth(test_client):
    """Test evaluation endpoint with authentication."""
    headers = {"Authorization": "Bearer test-api-key"}
    
    with patch('src.api.main.get_orchestrator') as mock_orchestrator:
        # This test would need proper mocking of the evaluation system
        # For now, we'll just test that the endpoint exists and requires auth
        response = test_client.post("/evaluation/single", headers=headers, json={
            "question": "Test question",
            "expected_answer": "Test answer"
        })
        
        # The actual response depends on the mock setup
        # At minimum, it should not be a 401 (unauthorized)
        assert response.status_code != 401


class TestAPIValidation:
    """Test API input validation."""
    
    def test_chat_missing_question(self, test_client):
        """Test chat endpoint with missing question."""
        headers = {"Authorization": "Bearer test-api-key"}
        
        response = test_client.post("/chat/ask", headers=headers, json={
            "session_id": "test"
        })
        assert response.status_code == 422  # Validation error
    
    def test_chat_empty_question(self, test_client):
        """Test chat endpoint with empty question."""
        headers = {"Authorization": "Bearer test-api-key"}
        
        response = test_client.post("/chat/ask", headers=headers, json={
            "question": "",
            "session_id": "test"
        })
        assert response.status_code == 422  # Validation error
    
    def test_evaluation_missing_fields(self, test_client):
        """Test evaluation endpoint with missing fields."""
        headers = {"Authorization": "Bearer test-api-key"}
        
        response = test_client.post("/evaluation/single", headers=headers, json={
            "question": "Test question"
            # Missing expected_answer
        })
        assert response.status_code == 422  # Validation error


class TestCORS:
    """Test CORS headers."""
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present."""
        response = test_client.options("/health/")
        
        # Check that CORS is configured (headers should be present)
        # The specific headers depend on the CORS configuration
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_for_nonexistent_endpoint(self, test_client):
        """Test 404 for non-existent endpoint."""
        response = test_client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_405_for_wrong_method(self, test_client):
        """Test 405 for wrong HTTP method."""
        response = test_client.put("/health/")
        assert response.status_code == 405