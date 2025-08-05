"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ChatRequest(BaseModel):
    """Chat request model."""
    question: str = Field(..., description="The question to ask")
    session_id: str = Field(default="default", description="Session ID for conversation memory")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the main contribution of the paper?",
                "session_id": "user123-session1"
            }
        }


class Source(BaseModel):
    """Source document model."""
    title: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Document Title",
                "content": "Relevant content excerpt...",
                "score": 0.85,
                "metadata": {
                    "page": 5,
                    "document_id": "doc123",
                    "source_type": "pdf"
                }
            }
        }


class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str = Field(..., description="The generated answer")
    sources: List[Source] = Field(default=[], description="Source documents used")
    confidence_score: float = Field(default=0.0, description="Confidence score (0-1)")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The main contribution of this paper is...",
                "sources": [
                    {
                        "title": "Paper Title",
                        "content": "Relevant excerpt...",
                        "score": 0.9,
                        "metadata": {"page": 3}
                    }
                ],
                "confidence_score": 0.85,
                "metadata": {
                    "session_id": "user123-session1",
                    "processing_time": 2.5
                }
            }
        }


class StreamingChunk(BaseModel):
    """Streaming response chunk model."""
    type: str = Field(..., description="Chunk type: 'token', 'complete', 'error'")
    content: str = Field(..., description="Chunk content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "token",
                "content": "The main",
                "metadata": None
            }
        }


class SessionClearRequest(BaseModel):
    """Session clear request model."""
    session_id: str = Field(..., description="Session ID to clear")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user123-session1"
            }
        }


class DocumentUploadRequest(BaseModel):
    """Document upload request model."""
    filename: str = Field(..., description="Name of the uploaded file")
    process_immediately: bool = Field(default=True, description="Process document immediately")
    
    class Config:
        json_schema_extra = {
            "example": {
                "filename": "research_paper.pdf",
                "process_immediately": True
            }
        }


class DocumentInfo(BaseModel):
    """Document information model."""
    id: str
    filename: str
    title: Optional[str] = None
    author: Optional[str] = None
    upload_date: datetime
    processing_status: str  # "pending", "processing", "completed", "failed"
    num_chunks: int = 0
    metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc123",
                "filename": "research_paper.pdf",
                "title": "Advanced RAG Techniques",
                "author": "John Doe",
                "upload_date": "2024-01-15T10:30:00Z",
                "processing_status": "completed",
                "num_chunks": 45,
                "metadata": {
                    "pages": 12,
                    "file_size": 1024000
                }
            }
        }


class DocumentListResponse(BaseModel):
    """Document list response model."""
    documents: List[DocumentInfo]
    total_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "id": "doc123",
                        "filename": "paper1.pdf",
                        "title": "RAG Techniques",
                        "upload_date": "2024-01-15T10:30:00Z",
                        "processing_status": "completed",
                        "num_chunks": 45
                    }
                ],
                "total_count": 1
            }
        }


class EvaluationRequest(BaseModel):
    """Evaluation request model."""
    question: str
    expected_answer: str
    context: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is RAG?",
                "expected_answer": "Retrieval-Augmented Generation...",
                "context": "Optional context information"
            }
        }


class EvaluationResult(BaseModel):
    """Evaluation result model."""
    question: str
    generated_answer: str
    expected_answer: str
    scores: Dict[str, float]  # ROUGE, BERT, etc.
    pass_threshold: bool
    metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is RAG?",
                "generated_answer": "RAG is a technique...",
                "expected_answer": "Retrieval-Augmented Generation...",
                "scores": {
                    "rouge_l": 0.85,
                    "bert_score": 0.92,
                    "semantic_similarity": 0.88
                },
                "pass_threshold": True,
                "metadata": {
                    "processing_time": 1.5,
                    "sources_used": 3
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
    dependencies: Dict[str, str] = Field(default={})
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "dependencies": {
                    "qdrant": "healthy",
                    "ollama": "healthy"
                }
            }
        }