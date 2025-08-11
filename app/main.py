"""Main FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine, Base
from app.core.logging import setup_logging, get_logger
from app.api import ingestion, chat
from app.models.schemas import HealthCheck

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Assessment API")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise
    
    # Initialize services (they will be lazy-loaded when first used)
    logger.info("RAG Assessment API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Assessment API")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG Assessment API with document ingestion and conversational chat",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingestion.router)
app.include_router(chat.router)


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "RAG Assessment API with document ingestion and conversational chat",
        "endpoints": {
            "ingestion": "/ingest",
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Check database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = f"unhealthy: {str(e)}"
    
    # Check Redis connection
    try:
        import redis
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_status = f"unhealthy: {str(e)}"
    
    # Check Milvus connection
    try:
        from app.services.storage_milvus import get_milvus_service
        milvus_service = get_milvus_service()
        milvus_service.get_collection_stats()
        milvus_status = "healthy"
    except Exception as e:
        logger.error(f"Milvus health check failed: {e}")
        milvus_status = f"unhealthy: {str(e)}"
    
    # Overall status
    services = {
        "database": db_status,
        "redis": redis_status,
        "milvus": milvus_status
    }
    
    overall_status = "healthy" if all("healthy" in status for status in services.values()) else "unhealthy"
    
    from datetime import datetime
    return HealthCheck(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        services=services
    )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return {
        "error": "Not Found",
        "detail": f"The requested endpoint {request.url.path} was not found",
        "available_endpoints": [
            "/",
            "/health",
            "/ingest",
            "/chat",
            "/docs"
        ]
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "detail": "An unexpected error occurred. Please try again later."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
