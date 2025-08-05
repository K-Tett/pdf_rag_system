import os
import sys
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path to fix import issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.core.config import Settings
    from src.services.vector_service import VectorService
    from src.services.document_service import DocumentService
    from src.agents.orchestrator import Orchestrator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Settings and security
settings = Settings()
security = HTTPBearer()

# Global service instances
document_service = None
vector_service = None
orchestrator = None


# Auth function
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication."""
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global document_service, vector_service, orchestrator
    
    logger.info("Starting PDF RAG application")
    
    try:
        # Initialize services
        vector_service = VectorService(settings.QDRANT_URL)
        await vector_service.initialize()
        
        document_service = DocumentService(vector_service)
        orchestrator = SimpleOrchestrator(settings, vector_service)
        
        # Load existing documents
        await document_service.load_existing_documents()
        
        logger.info("Application startup complete")
        yield
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    finally:
        logger.info("Shutting down application")
        if vector_service:
            await vector_service.close()


# Create FastAPI app
app = FastAPI(
    title="PDF RAG System",
    description="Chat with your PDF documents using AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "PDF RAG System API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    dependencies = {}
    
    # Check Qdrant
    try:
        if vector_service:
            await vector_service.get_document_count()
            dependencies["qdrant"] = "healthy"
        else:
            dependencies["qdrant"] = "not_initialized"
    except Exception:
        dependencies["qdrant"] = "unhealthy"
    
    return {
        "status": "healthy" if all(dep == "healthy" for dep in dependencies.values()) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": dependencies
    }


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    _: str = Depends(verify_api_key)
):
    """Upload a PDF document."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    if not document_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service not ready"
        )
    
    try:
        # Save file
        upload_dir = Path(document_service.data_dir)
        file_path = upload_dir / file.filename
        
        if file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"File '{file.filename}' already exists"
            )
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        document_id = await document_service.process_pdf(str(file_path), True)
        doc_info = document_service.get_document_info(document_id)
        
        if not doc_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process document"
            )
        
        # Return as plain dictionary
        return {
            "id": doc_info.id,
            "filename": doc_info.filename,
            "title": doc_info.title,
            "author": doc_info.author,
            "upload_date": doc_info.upload_date.isoformat() if doc_info.upload_date else None,
            "processing_status": doc_info.processing_status,
            "num_chunks": doc_info.num_chunks,
            "metadata": doc_info.metadata or {}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@app.get("/documents/")
async def list_documents(_: str = Depends(verify_api_key)):
    """List all documents."""
    if not document_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service not ready"
        )
    
    try:
        documents = document_service.list_documents()
        
        # Convert to plain dictionaries
        documents_data = []
        for doc in documents:
            documents_data.append({
                "id": doc.id,
                "filename": doc.filename,
                "title": doc.title,
                "author": doc.author,
                "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                "processing_status": doc.processing_status,
                "num_chunks": doc.num_chunks,
                "metadata": doc.metadata or {}
            })
        
        return {
            "documents": documents_data,
            "total_count": len(documents_data)
        }
        
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    _: str = Depends(verify_api_key)
):
    """Delete a document."""
    if not document_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service not ready"
        )
    
    try:
        success = await document_service.delete_document(document_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {"message": "Document deleted successfully", "document_id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@app.post("/chat/ask")
async def ask_question(
    request: Dict[str, Any],
    _: str = Depends(verify_api_key)
):
    """Ask a question about your documents."""
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not ready"
        )
    
    try:
        question = request.get("question", "")
        session_id = request.get("session_id", "default")
        
        if not question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question is required"
            )
        
        logger.info("Processing question", question=question)
        
        response = await orchestrator.process_question(
            question=question,
            session_id=session_id,
            stream=False
        )
        
        # Convert sources to dictionaries
        sources_data = []
        for source in response.sources:
            if hasattr(source, 'model_dump'):
                sources_data.append(source.model_dump())
            elif hasattr(source, 'dict'):
                sources_data.append(source.dict())
            else:
                sources_data.append({
                    "title": getattr(source, 'title', ''),
                    "content": getattr(source, 'content', ''),
                    "score": getattr(source, 'score', 0.0),
                    "metadata": getattr(source, 'metadata', {})
                })
        
        return {
            "answer": response.answer,
            "sources": sources_data,
            "confidence_score": response.confidence_score,
            "metadata": response.metadata or {}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )


@app.post("/chat/clear-memory")
async def clear_memory(
    request: Dict[str, Any],
    _: str = Depends(verify_api_key)
):
    """Clear conversation memory."""
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not ready"
        )
    
    try:
        session_id = request.get("session_id", "default")
        await orchestrator.clear_session_memory(session_id)
        return {"message": "Memory cleared", "session_id": session_id}
        
    except Exception as e:
        logger.error("Failed to clear memory", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear memory: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)