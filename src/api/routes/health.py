"""
Health check API routes.
"""
import structlog
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import asyncio
import aiohttp
from qdrant_client import QdrantClient

from src.core.models import HealthResponse
from src.core.config import Settings

logger = structlog.get_logger()
router = APIRouter()

settings = Settings()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Main health check endpoint.
    """
    try:
        # Check all dependencies
        dependencies = await check_all_dependencies()
        
        # Determine overall status
        overall_status = "healthy"
        if any(status != "healthy" for status in dependencies.values()):
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/detailed")
async def detailed_health_check():
    """
    Detailed health check with more information about each service.
    """
    try:
        detailed_info = {}
        
        # Check Qdrant
        qdrant_info = await check_qdrant_detailed()
        detailed_info["qdrant"] = qdrant_info
        
        # Check Ollama
        ollama_info = await check_ollama_detailed()
        detailed_info["ollama"] = ollama_info
        
        # Check OpenAI (if configured)
        if settings.has_openai_key:
            openai_info = await check_openai_detailed()
            detailed_info["openai"] = openai_info
        
        # System info
        system_info = get_system_info()
        detailed_info["system"] = system_info
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "services": detailed_info
        }
        
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Detailed health check failed: {str(e)}"
        )


@router.get("/qdrant")
async def qdrant_health():
    """Check Qdrant health specifically."""
    try:
        status_info = await check_qdrant_detailed()
        return status_info
    except Exception as e:
        logger.error("Qdrant health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant health check failed: {str(e)}"
        )


@router.get("/ollama")
async def ollama_health():
    """Check Ollama health specifically."""
    try:
        status_info = await check_ollama_detailed()
        return status_info
    except Exception as e:
        logger.error("Ollama health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama health check failed: {str(e)}"
        )


async def check_all_dependencies() -> dict:
    """Check all service dependencies."""
    dependencies = {}
    
    # Check Qdrant
    try:
        client = QdrantClient(url=settings.QDRANT_URL)
        client.get_collections()
        dependencies["qdrant"] = "healthy"
        client.close()
    except Exception as e:
        logger.warning("Qdrant health check failed", error=str(e))
        dependencies["qdrant"] = "unhealthy"
    
    # Check Ollama
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{settings.OLLAMA_URL}/api/tags") as response:
                if response.status == 200:
                    dependencies["ollama"] = "healthy"
                else:
                    dependencies["ollama"] = "unhealthy"
    except Exception as e:
        logger.warning("Ollama health check failed", error=str(e))
        dependencies["ollama"] = "unhealthy"
    
    # Check OpenAI (if configured)
    if settings.has_openai_key:
        try:
            # Simple check - we'll assume it's healthy if we have a key
            # In practice, you might want to make a test API call
            dependencies["openai"] = "healthy"
        except Exception as e:
            logger.warning("OpenAI health check failed", error=str(e))
            dependencies["openai"] = "unhealthy"
    
    return dependencies


async def check_qdrant_detailed() -> dict:
    """Get detailed Qdrant health information."""
    try:
        client = QdrantClient(url=settings.QDRANT_URL)
        
        # Get collections info
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        # Get collection details if our collection exists
        collection_info = {}
        if settings.QDRANT_COLLECTION_NAME in collection_names:
            collection_details = client.get_collection(settings.QDRANT_COLLECTION_NAME)
            collection_info = {
                "points_count": collection_details.points_count,
                "segments_count": collection_details.segments_count,
                "status": collection_details.status.value
            }
        
        client.close()
        
        return {
            "status": "healthy",
            "url": settings.QDRANT_URL,
            "collections": collection_names,
            "target_collection": settings.QDRANT_COLLECTION_NAME,
            "collection_info": collection_info,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "url": settings.QDRANT_URL,
            "checked_at": datetime.utcnow().isoformat()
        }


async def check_ollama_detailed() -> dict:
    """Get detailed Ollama health information."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            # Check if service is running
            async with session.get(f"{settings.OLLAMA_URL}/api/tags") as response:
                if response.status == 200:
                    models_data = await response.model_dump_json()
                    models = [model["name"] for model in models_data.get("models", [])]
                    
                    # Check if our target model is available
                    target_model_available = settings.OLLAMA_MODEL in models
                    
                    return {
                        "status": "healthy",
                        "url": settings.OLLAMA_URL,
                        "available_models": models,
                        "target_model": settings.OLLAMA_MODEL,
                        "target_model_available": target_model_available,
                        "checked_at": datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status}",
                        "url": settings.OLLAMA_URL,
                        "checked_at": datetime.utcnow().isoformat()
                    }
                    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "url": settings.OLLAMA_URL,
            "checked_at": datetime.utcnow().isoformat()
        }


async def check_openai_detailed() -> dict:
    """Get detailed OpenAI health information."""
    try:
        # For OpenAI, we'll just check if we have a valid API key format
        # Making actual API calls for health checks can be expensive
        
        api_key = settings.OPENAI_API_KEY
        is_valid_format = api_key and api_key.startswith("sk-") and len(api_key) > 40
        
        return {
            "status": "healthy" if is_valid_format else "unhealthy",
            "api_key_configured": bool(api_key),
            "api_key_format_valid": is_valid_format,
            "target_model": settings.OPENAI_MODEL,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "checked_at": datetime.utcnow().isoformat()
        }


def get_system_info() -> dict:
    """Get system information."""
    import psutil
    import platform
    
    try:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    except Exception as e:
        return {
            "error": f"Failed to get system info: {str(e)}"
        }


@router.get("/readiness")
async def readiness_check():
    """
    Readiness check - determines if the service is ready to handle requests.
    """
    try:
        # Check critical dependencies
        dependencies = await check_all_dependencies()
        
        # For readiness, we need at least Qdrant to be healthy
        critical_services = ["qdrant"]
        
        for service in critical_services:
            if dependencies.get(service) != "healthy":
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Critical service {service} is not healthy"
                )
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "critical_services": critical_services,
            "all_dependencies": dependencies
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )


@router.get("/liveness")
async def liveness_check():
    """
    Liveness check - determines if the service is alive and should be restarted if not.
    """
    try:
        # Simple liveness check - just verify the service can respond
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": "unknown"  # Could track actual uptime
        }
    except Exception as e:
        logger.error("Liveness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Liveness check failed: {str(e)}"
        )