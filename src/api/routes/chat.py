"""
Chat API routes for question answering.
"""
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import asyncio

# Import the global dependency functions
import sys
sys.path.append('..')
from src.api.main import get_orchestrator_service
from src.core.models import ChatRequest, ChatResponse, SessionClearRequest

logger = structlog.get_logger()
router = APIRouter()


class StreamingChatResponse(BaseModel):
    """Streaming chat response model."""
    type: str  # "token", "complete", "error"
    content: str
    metadata: Optional[Dict[str, Any]] = None


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(lambda: get_orchestrator_service())
):
    """
    Ask a question and get a complete response.
    """
    try:
        logger.info(
            "Processing question",
            question=request.question,
            session_id=request.session_id
        )
        
        response = await orchestrator.process_question(
            question=request.question,
            session_id=request.session_id,
            stream=False
        )
        
        logger.info(
            "Question processed successfully",
            session_id=request.session_id,
            response_length=len(response.answer)
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Error processing question",
            error=str(e),
            session_id=request.session_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


@router.post("/ask/stream", response_class=StreamingResponse)
async def ask_question_stream(
    request: ChatRequest,
    orchestrator: AgentOrchestrator = Depends()
):
    """
    Ask a question and get a streaming response.
    """
    async def generate_response():
        """Generate streaming response."""
        try:
            logger.info(
                "Processing streaming question",
                question=request.question,
                session_id=request.session_id
            )
            
            async for chunk in orchestrator.process_question_stream(
                question=request.question,
                session_id=request.session_id
            ):
                if chunk.type == "token":
                    # Stream token
                    response = StreamingChatResponse(
                        type="token",
                        content=chunk.content
                    )
                elif chunk.type == "complete":
                    # Stream complete response with metadata
                    response = StreamingChatResponse(
                        type="complete",
                        content=chunk.content,
                        metadata=chunk.metadata
                    )
                elif chunk.type == "error":
                    # Stream error
                    response = StreamingChatResponse(
                        type="error",
                        content=chunk.content
                    )
                else:
                    continue
                
                yield f"data: {response.model_dump_json()}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(
                "Error in streaming response",
                error=str(e),
                session_id=request.session_id
            )
            error_response = StreamingChatResponse(
                type="error",
                content=f"Error processing question: {str(e)}"
            )
            yield f"data: {error_response.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


@router.post("/clear-memory")
async def clear_session_memory(
    request: SessionClearRequest,
    orchestrator: AgentOrchestrator = Depends()
):
    """
    Clear conversation memory for a session.
    """
    try:
        logger.info("Clearing session memory", session_id=request.session_id)
        
        await orchestrator.clear_session_memory(request.session_id)
        
        logger.info("Session memory cleared", session_id=request.session_id)
        
        return {"message": "Session memory cleared successfully", "session_id": request.session_id}
        
    except Exception as e:
        logger.error(
            "Error clearing session memory",
            error=str(e),
            session_id=request.session_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing session memory: {str(e)}"
        )


@router.get("/sessions/{session_id}/status")
async def get_session_status(
    session_id: str,
    orchestrator: AgentOrchestrator = Depends()
):
    """
    Get session status and conversation history.
    """
    try:
        status_info = await orchestrator.get_session_status(session_id)
        return status_info
        
    except Exception as e:
        logger.error(
            "Error getting session status",
            error=str(e),
            session_id=session_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting session status: {str(e)}"
        )


@router.get("/health")
async def chat_health_check():
    """Chat service health check."""
    return {"status": "healthy", "service": "chat"}