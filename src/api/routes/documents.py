"""
Document management API routes.
"""
import structlog
import os
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, status, UploadFile, File
from typing import List

from src.core.models import (
    DocumentUploadRequest, 
    DocumentInfo, 
    DocumentListResponse
)

logger = structlog.get_logger()
router = APIRouter()


# Get document service from global state
def get_global_document_service():
    """Get document service from global application state."""
    from src.api.main import document_service
    if document_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service not initialized. Please wait for startup to complete."
        )
    return document_service


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    process_immediately: bool = True
):
    """
    Upload a PDF document for processing.
    """
    logger.info("Uploading document", filename=file.filename)
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    try:
        document_service = get_global_document_service()
        
        # Save uploaded file
        upload_dir = Path(document_service.data_dir)
        file_path = upload_dir / file.filename
        
        # Check if file already exists
        if file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"File '{file.filename}' already exists"
            )
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        document_id = await document_service.process_pdf(
            str(file_path),
            process_immediately=process_immediately
        )
        
        # Get document info
        doc_info = document_service.get_document_info(document_id)
        
        if doc_info is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve document information after processing"
            )
        
        logger.info(
            "Document uploaded successfully",
            filename=file.filename,
            document_id=document_id
        )
        
        return doc_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload document", filename=file.filename, error=str(e))
        
        # Clean up file if it was created
        if 'file_path' in locals() and file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        ) is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve document information after processing"
            )
        
        logger.info(
            "Document uploaded successfully",
            filename=file.filename,
            document_id=document_id
        )
        
        return doc_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload document", filename=file.filename, error=str(e))
        
        # Clean up file if it was created
        if 'file_path' in locals() and file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents.
    """
    try:
        document_service = get_global_document_service()
        documents = document_service.list_documents()
        
        logger.info("Listed documents", count=len(documents))
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    """
    Get information about a specific document.
    """
    try:
        document_service = get_global_document_service()
        doc_info = document_service.get_document_info(document_id)
        
        if doc_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' not found"
            )
        
        logger.info("Retrieved document info", document_id=document_id)
        
        return doc_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document", document_id=document_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its associated data.
    """
    try:
        document_service = get_global_document_service()
        
        # Check if document exists
        doc_info = document_service.get_document_info(document_id)
        if doc_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' not found"
            )
        
        # Delete the document
        success = await document_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document"
            )
        
        logger.info("Document deleted successfully", document_id=document_id)
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document", document_id=document_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.get("/stats/overview")
async def get_documents_stats():
    """
    Get overall document statistics.
    """
    try:
        document_service = get_global_document_service()
        documents = document_service.list_documents()
        
        # Calculate statistics
        total_documents = len(documents)
        completed_documents = len([doc for doc in documents if doc.processing_status == "completed"])
        failed_documents = len([doc for doc in documents if doc.processing_status == "failed"])
        processing_documents = len([doc for doc in documents if doc.processing_status == "processing"])
        
        total_chunks = sum(doc.num_chunks for doc in documents)
        
        # Document size statistics
        file_sizes = [doc.metadata.get("file_size", 0) for doc in documents]
        total_size = sum(file_sizes)
        avg_size = total_size / max(total_documents, 1)
        
        stats = {
            "total_documents": total_documents,
            "completed_documents": completed_documents,
            "failed_documents": failed_documents,
            "processing_documents": processing_documents,
            "total_chunks": total_chunks,
            "average_chunks_per_document": total_chunks / max(completed_documents, 1),
            "total_file_size_bytes": total_size,
            "average_file_size_bytes": avg_size
        }
        
        logger.info("Retrieved document statistics", **stats)
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document statistics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document statistics: {str(e)}"
        )